#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 01 报告插图生成：训练损失曲线、PR 曲线、指标柱状图、检测结果拼图。

给实验报告直接用的图，统一输出到 reports/figures/，编号与报告正文对应。

三个数据来源（都可用 --xxx 指定，缺哪个就跳过哪部分）：
  1. 训练日志目录 —— 从 ModelCheckpoint 的**文件名**里解析损失历史。
     checkpoints 名字形如 `ep012-loss45.231-val_loss48.900.h5`，本身就带 epoch/loss/val_loss，
     所以**不需要装 TensorFlow 就能画损失曲线**，也不必去解析 events 文件。
  2. 评测指标目录 —— evaluate_detection.py 输出的 pr_curve.csv 与 detection_metrics.json。
  3. 检测结果目录 —— detect_images.py 输出的带框图片，拼成一张网格图。

用法：
    python make_report_figures.py                       # 三个都尝试
    python make_report_figures.py --logs ../logs/full_416_xxx
    python make_report_figures.py --metrics ../outputs/metrics
    python make_report_figures.py --detect ../outputs/detect/xxx --montage 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from config import (
        ANCHORS_PATH, CLASSES_PATH, IMAGE_EXTS, LOGS_DIR, OUTPUTS_DIR, VOC_ROOT,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        ANCHORS_PATH, CLASSES_PATH, IMAGE_EXTS, LOGS_DIR, OUTPUTS_DIR, VOC_ROOT,
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = REPO_ROOT / "reports" / "figures"

CKPT_RE = re.compile(r"ep(\d+)-loss([\d.]+)-val_loss([\d.]+)\.h5$", re.IGNORECASE)


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Windows 上中文用雅黑/黑体，避免出现方框
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


def parse_tb_history(log_dir: Path):
    """从 TensorBoard 事件文件里读**完整**的逐 epoch 损失历史。

    为什么优先用它：ModelCheckpoint 设了 `save_best_only=True`，
    只在 val_loss 改善时才写 checkpoint，所以文件名里只有**少数几个** epoch。
    本实验就出现过「跑了几十轮却只有 3 个 checkpoint」的情况，
    此时按文件名画出来的曲线只有 3 个点，完全看不出趋势。
    TensorBoard 回调是每个 epoch 都记录的，数据完整。

    返回 (epochs, train_losses, val_losses)，取不到则返回 None。
    **epoch 已统一成 1 基**，与训练日志里的 `Epoch N/200` 以及
    checkpoint 文件名里的 `ep{N}` 一致。
    （TensorBoard 的 step 是 0 基：step 167 对应日志的 Epoch 168，
      实测用 val_loss 值逐一对上确认过。）
    """
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import tensorflow as tf
    except Exception as e:
        # 只吞 ImportError 之外的异常是危险的：这里曾经因为模块忘了 import os，
        # `os.environ` 抛 NameError 被吞掉，函数静默返回 None，
        # 表面上只是"回退到 checkpoint 文件名"，排查了很久。
        if not isinstance(e, ImportError):
            print("  [提示] 读取 TensorBoard 事件失败（%s: %s），改用 checkpoint 文件名"
                  % (type(e).__name__, e))
        return None

    def _value(v):
        """把 Summary.Value 取成 float。

        TF 2.x 里损失是以 `tensor` 字段存的（不是 simple_value），
        且转换函数是 `tf.make_ndarray`（TF 1.x 的
        `tensorflow.python.framework.tensor_util.make_ndarray` 在 2.10 已不存在）。
        """
        try:
            if v.HasField("simple_value"):
                return float(v.simple_value)
        except Exception:
            pass
        try:
            if v.HasField("tensor"):
                return float(tf.make_ndarray(v.tensor))
        except Exception:
            return None
        return None

    series = {}
    for sub in ("train", "validation"):
        d = log_dir / sub
        if not d.is_dir():
            continue
        store = {}
        for f in sorted(d.glob("events.out.tfevents.*")):
            try:
                for e in tf.compat.v1.train.summary_iterator(str(f)):
                    for v in e.summary.value:
                        if v.tag != "epoch_loss":
                            continue
                        x = _value(v)
                        if x is not None:
                            # +1：TensorBoard 的 step 从 0 开始，日志从 Epoch 1 开始
                            store[int(e.step) + 1] = x
            except Exception:
                continue
        if store:
            series[sub] = store

    if not series:
        return None

    epochs = sorted(set().union(*[set(s) for s in series.values()]))
    tr = [series.get("train", {}).get(e) for e in epochs]
    va = [series.get("validation", {}).get(e) for e in epochs]
    return epochs, tr, va


def parse_ckpt_history(log_dir: Path):
    """从 checkpoint 文件名解析训练历史（TensorBoard 事件不可用时的兜底）。

    返回 (epochs, losses, val_losses)，按 epoch 升序。
    """
    rows = []
    for p in log_dir.glob("*.h5"):
        m = CKPT_RE.search(p.name)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    rows.sort(key=lambda r: r[0])
    if not rows:
        return [], [], []
    ep = [r[0] for r in rows]
    ls = [r[1] for r in rows]
    vl = [r[2] for r in rows]
    return ep, ls, vl


def parse_tb_history_multi(log_dirs):
    """把多个日志目录的 TensorBoard 历史合并起来。

    断点续训会产生第二个日志目录（本次是 full_catface_v2 跑到第 172 轮，
    重启后续训写在 full_catface_v3）。报告里的损失曲线应当横跨两段，
    否则只能看到续训那 33 轮——恰好是最平的尾部，看不出任何趋势。
    """
    merged = {}
    for d in log_dirs:
        d = Path(d)
        r = parse_tb_history(d)
        if r is None:
            continue
        for e, l, v in zip(*r):
            slot = merged.setdefault(int(e), [None, None])
            if l is not None:
                slot[0] = l
            if v is not None:
                slot[1] = v
    if not merged:
        return None
    eps = sorted(merged)
    return (eps,
            [merged[e][0] for e in eps],
            [merged[e][1] for e in eps])


def fig_loss_curve(plt, log_dir, out: Path, run_label: str):
    # 优先用 TensorBoard 事件（每轮都有记录），退回 checkpoint 文件名（只含改善轮次）
    dirs = ([Path(x) for x in log_dir] if isinstance(log_dir, (list, tuple))
            else [Path(log_dir)])
    tb = parse_tb_history_multi(dirs)
    source = "TensorBoard 事件"
    if tb is not None:
        ep, ls, vl = tb
        # val 可能只有部分 epoch 有值，画图时过滤掉 None
    else:
        # 兜底：从各目录的 checkpoint 文件名汇总
        allrows = []
        for d in dirs:
            e, l, v = parse_ckpt_history(d)
            allrows += list(zip(e, l, v))
        allrows.sort(key=lambda r: r[0])
        ep = [r[0] for r in allrows]
        ls = [r[1] for r in allrows]
        vl = [r[2] for r in allrows]
        source = "checkpoint 文件名"

    pairs = [(e, l, v) for e, l, v in zip(ep, ls, vl) if l is not None or v is not None]
    if not pairs:
        print("  [跳过] %s 里没有任何损失记录" % log_dir)
        return None
    ep = [p[0] for p in pairs]
    ls = [p[1] for p in pairs]
    vl = [p[2] for p in pairs]
    print("  [数据来源] %s，共 %d 个 epoch" % (source, len(ep)))

    # 训练初期 val_loss 可能比后期大十几个数量级（随机初始化 + 背景项主导），
    # 线性坐标会把后面所有点压成一条贴着 0 的直线，图就废了。
    # 跨度过大时改用对数坐标，并把这个情况标注出来。
    finite = [v for v in (list(ls) + list(vl)) if v is not None and v > 0]
    use_log = bool(finite) and (max(finite) / max(min(finite), 1e-12) > 1000)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    def series(xs, ys):
        """过滤掉 None（TensorBoard 里某个 epoch 可能只有 train 或只有 val）。"""
        p = [(x, y) for x, y in zip(xs, ys) if y is not None]
        return [q[0] for q in p], [q[1] for q in p]

    ep_t, ls_t = series(ep, ls)
    ep_v, vl_v = series(ep, vl)

    if ls_t:
        ax.plot(ep_t, ls_t, marker="o", ms=3, lw=1.6, label="train loss")
    if vl_v:
        ax.plot(ep_v, vl_v, marker="s", ms=3, lw=1.6, label="val loss")
        bi = min(range(len(vl_v)), key=lambda i: vl_v[i])
        ax.scatter([ep_v[bi]], [vl_v[bi]], s=90, facecolors="none",
                   edgecolors="red", zorder=5,
                   label="best val (ep %d, %.3f)" % (ep_v[bi], vl_v[bi]))
    else:
        bi = 0
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    title = "图 1-1 训练损失曲线（%s）" % run_label
    if use_log:
        ax.set_yscale("log")
        title += "　[对数纵轴]"
        # 标出第一轮那个异常大的点，避免读者误以为曲线是线性的
        if vl_v:
            ax.annotate("首轮 val_loss = %.3g\n（随机初始化 + 背景项主导）" % vl_v[0],
                        xy=(ep_v[0], vl_v[0]), xytext=(0.28, 0.80),
                        textcoords="axes fraction", fontsize=8, color="#b45309",
                        arrowprops=dict(arrowstyle="->", color="#b45309", lw=1))

    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(out)
    plt.close(fig)
    print("  [生成] %s  (%d 个 epoch%s)"
          % (out, len(ep), "，对数纵轴" if use_log else ""))
    return {"epochs": len(ep),
            "best_epoch": ep_v[bi] if vl_v else None,
            "best_val_loss": vl_v[bi] if vl_v else None}


def fig_pr_curve(plt, metrics_dir: Path, out: Path):
    pr = metrics_dir / "pr_curve.csv"
    if not pr.is_file():
        print("  [跳过] 找不到 %s" % pr)
        return
    import csv
    from collections import defaultdict
    data = defaultdict(list)
    with open(pr, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            data[row["class"]].append((float(row["recall"]), float(row["precision"])))
    if not data:
        print("  [跳过] %s 里没有数据" % pr)
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for cname, pts in data.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.8, label=cname)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("图 1-2 Precision-Recall 曲线")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out)
    plt.close(fig)
    print("  [生成] %s" % out)


def fig_metrics_bar(plt, metrics_dir: Path, out: Path):
    js = metrics_dir / "detection_metrics.json"
    if not js.is_file():
        print("  [跳过] 找不到 %s" % js)
        return
    # 用 utf-8-sig 读：文件被记事本之类编辑过会带上 BOM，
    # 纯 utf-8 解码会直接抛 JSONDecodeError
    try:
        d = json.loads(js.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        print("  [跳过] %s 不是合法 JSON：%s" % (js.name, e))
        return
    classes = d.get("classes", {})
    if not classes:
        print("  [跳过] 指标里没有类别")
        return

    names = list(classes)
    keys = ["precision", "recall", "f1", "ap"]
    labels = ["Precision", "Recall", "F1", "AP"]
    import numpy as np
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (k, lb) in enumerate(zip(keys, labels)):
        vals = [classes[n].get(k, 0) for n in names]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lb)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, "%.3f" % v,
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title("图 1-3 各类别检测指标（IoU=%.2f，mAP=%.4f）"
                 % (d.get("iou_threshold", 0.5), d.get("mAP", 0)))
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=4, fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    print("  [生成] %s" % out)


def fig_montage(plt, detect_dir: Path, out: Path, n: int, cols: int = 4):
    from PIL import Image
    imgs = sorted(p for p in detect_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not imgs:
        print("  [跳过] %s 里没有结果图" % detect_dir)
        return
    imgs = imgs[:n]
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, p in zip(axes, imgs):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(p.name, fontsize=9)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")
    fig.suptitle("图 1-4 猫脸检测结果示例", fontsize=13)
    fig.savefig(out)
    plt.close(fig)
    print("  [生成] %s  (%d 张)" % (out, len(imgs)))


def main():
    ap = argparse.ArgumentParser(description="生成实验 01 报告插图")
    ap.add_argument("--logs", default=str(LOGS_DIR),
                    help="训练日志目录（含 ep*-loss*-val_loss*.h5 的 checkpoint）")
    ap.add_argument("--metrics", default=str(OUTPUTS_DIR / "metrics"),
                    help="evaluate_detection.py 的输出目录")
    ap.add_argument("--detect", default=None,
                    help="detect_images.py 的结果目录（用于拼图）")
    ap.add_argument("--montage", type=int, default=8, help="拼图用几张图（默认 8）")
    ap.add_argument("--out", default=str(FIG_DIR), help="图片输出目录")
    ap.add_argument("--run-label", default=None, help="损失曲线标题里的运行名")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    try:
        plt = setup_matplotlib()
    except ImportError:
        print("[错误] 需要 matplotlib。可用 cv 环境：")
        print("  python make_report_figures.py")
        return 2

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("实验 01 · 生成报告插图")
    print("=" * 70)
    print("输出目录 : %s" % out_dir)
    print("")

    # --logs 支持逗号分隔的多个目录（断点续训会分成两段记录）
    logs = [Path(p.strip()).resolve() for p in str(args.logs).split(",") if p.strip()]
    logs = [p for p in logs if p.is_dir()]
    if logs:
        label = args.run_label or "+".join(p.name for p in logs)
        fig_loss_curve(plt, logs, out_dir / "01_loss_curve.png", label)
    else:
        print("  [跳过] --logs 里没有有效目录")

    metrics = Path(args.metrics).resolve()
    if metrics.is_dir():
        fig_pr_curve(plt, metrics, out_dir / "01_pr_curve.png")
        fig_metrics_bar(plt, metrics, out_dir / "01_metrics_bar.png")

    if args.detect:
        d = Path(args.detect).resolve()
        if d.is_dir():
            fig_montage(plt, d, out_dir / "01_detect_samples.png", args.montage)

    print("")
    print("完成。图片在 %s" % out_dir)
    print("报告正文里引用时按 图 1-1 / 图 1-2 ... 编号。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
