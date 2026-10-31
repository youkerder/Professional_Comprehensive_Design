# -*- coding: utf-8 -*-
"""实验三 · 出图与出表（对应“交付要求：轨迹图、FPS/指标对比表”）。

输入：outputs/metrics/eval_*.json（evaluate.py 产物）、outputs/traces/*.csv
输出：
  outputs/figures/*.png      —— 实验过程图
  outputs/tables/*.csv|md    —— 指标表
  reports/figures/、reports/tables/ 各存一份（报告直接引用）

图：
  1. success / precision 曲线（OTB 口径）
  2. 逐帧 IoU 曲线（遮挡序列，直观看出何时丢目标）
  3. 轨迹对比图（真值 vs 各算法）
  4. FPS / 精度柱状图、FPS-精度散点图
  5. 定性对比拼图（同一帧各算法的框）
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from common import (ANN_DIR, FIG_DIR, METRIC_DIR, OUT_DIR, REPORT_FIG_DIR,
                    REPORT_TAB_DIR, TRACE_DIR, VIDEO_OUT_DIR, ensure_dir,
                    load_otb_gt, load_trace, setup_console)

# ---------------------------------------------------------------- 中文字体
def setup_font():
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        try:
            path = font_manager.findfont(name, fallback_to_default=False)
            if path and Path(path).exists():
                font_manager.fontManager.addfont(path)
                matplotlib.rcParams["font.sans-serif"] = [name]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return name
        except Exception:
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "default"


C_GT = "#222222"
ALGO_STYLE = {
    "KCF": ("#d62728", "-"),
    "CSRT": ("#1f77b4", "-"),
    "MEDIANFLOW": ("#2ca02c", "-"),
    "MIL": ("#9467bd", "--"),
    "BOOSTING": ("#8c564b", "--"),
    "TLD": ("#e377c2", "--"),
    "MIXED": ("#ff7f0e", "-"),
}


def style(algo):
    if algo in ALGO_STYLE:
        return ALGO_STYLE[algo]
    return ("#7f7f7f", "-")


def load_evals(mdir):
    out = []
    for p in sorted(mdir.glob("eval_*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def tag_name(ev):
    algos = ev.get("algos") or [ev["tag"]]
    return "+".join(algos) if len(algos) > 1 else algos[0]


def disp(name):
    """图上用的短标签：混用实验太长，横轴放不下。"""
    if "+" in name:
        return "混合(3种)"
    return name


def group(evals):
    """{seq: [eval, ...]}，按算法顺序排序。"""
    order = {k: i for i, k in enumerate(
        ["KCF", "CSRT", "MEDIANFLOW", "MIL", "BOOSTING", "TLD"])}
    g = {}
    for ev in evals:
        g.setdefault(ev["seq"], []).append(ev)
    for seq in g:
        g[seq].sort(key=lambda e: order.get(tag_name(e).split("+")[0], 99))
    return g


# ---------------------------------------------------------------- 曲线
def fig_curves(evals, out_dir):
    g = group(evals)
    thr = np.arange(0.0, 1.0001, 0.01)
    pthr = np.arange(0.0, 51.0, 1.0)
    for seq, evs in g.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ev in evs:
            name = tag_name(ev)
            col, ls = style(name.split("+")[0])
            succ = np.mean([np.asarray(v["success_curve"]) for v in ev["otb_per_target"].values()
                            if "success_curve" in v], axis=0)
            prec = np.mean([np.asarray(v["precision_curve"]) for v in ev["otb_per_target"].values()
                            if "precision_curve" in v], axis=0)
            axes[0].plot(thr, succ, ls, color=col, lw=1.8,
                         label=f"{name} (AUC={ev['otb']['auc']:.3f})")
            axes[1].plot(pthr, prec, ls, color=col, lw=1.8,
                         label=f"{name} (P@20={ev['otb']['prec20'] * 100:.1f}%)")
        axes[0].set_xlabel("IoU 阈值")
        axes[0].set_ylabel("成功率（IoU ≥ 阈值的帧占比）")
        axes[0].set_title(f"Success plot — 序列 {seq}")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8, loc="lower left")
        axes[1].set_xlabel("中心误差阈值 / px")
        axes[1].set_ylabel("精度（误差 ≤ 阈值的帧占比）")
        axes[1].set_title(f"Precision plot — 序列 {seq}")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        p = out_dir / f"fig_curves_{seq}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"  {p}")


# ---------------------------------------------------------------- 逐帧 IoU
def fig_iou_timeline(evals, out_dir):
    from common import iou
    g = group(evals)
    for seq, evs in g.items():
        # 预载真值，避免逐行读文件
        gt_cache = {}
        for t in sorted(evs[0]["otb_per_target"]):
            p = ANN_DIR / seq / "otb" / f"{t}.txt"
            if p.exists():
                gt_cache[t] = load_otb_gt(p)
        fig, ax = plt.subplots(figsize=(10, 4))
        for ev in evs:
            name = tag_name(ev)
            col, ls = style(name.split("+")[0])
            per_frame = {}
            for r in load_trace(ev["trace"]):
                gt = gt_cache.get(r["target"], {})
                if r["frame"] in gt:
                    per_frame.setdefault(r["frame"], []).append(iou(r["box"], gt[r["frame"]]))
            fs = sorted(per_frame)
            vals = [float(np.mean(per_frame[f])) for f in fs]
            ax.plot(fs, vals, ls, color=col, lw=1.4, label=name)
        # 遮挡区间阴影
        meta_p = ANN_DIR / seq / "gt.txt"
        if meta_p.exists():
            occ_frames = {}
            for line in meta_p.read_text(encoding="utf-8").splitlines():
                p = line.split(",")
                if len(p) >= 9 and float(p[8]) < 0.05:
                    fid = int(float(p[1]))
                    occ_frames.setdefault(fid, []).append(int(float(p[0])) - 1)
            for tid, fs in occ_frames.items():
                ax.axvspan(min(fs), max(fs), color="gray", alpha=0.18,
                           label=f"target{tid} 完全遮挡")
        ax.set_xlabel("帧号")
        ax.set_ylabel("与真值的 IoU（三目标平均）")
        ax.set_title(f"逐帧 IoU — 序列 {seq}")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="lower left")
        fig.tight_layout()
        p = out_dir / f"fig_iou_timeline_{seq}.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        print(f"  {p}")


# ---------------------------------------------------------------- 轨迹
def fig_trajectories(evals, out_dir):
    g = group(evals)
    for seq, evs in g.items():
        n = len(evs)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows),
                                 squeeze=False)
        target_names = sorted(evs[0]["otb_per_target"])
        tcolors = ["#d62728", "#1f77b4", "#2ca02c"]
        for k, ev in enumerate(evs):
            ax = axes[k // cols][k % cols]
            name = tag_name(ev)
            rows_t = load_trace(ev["trace"])
            for ti, t in enumerate(target_names):
                col = tcolors[ti % len(tcolors)]
                # 真值（灰色虚线）
                gt = load_otb_gt(ANN_DIR / seq / "otb" / f"{t}.txt")
                fs = sorted(gt)
                cxs = [gt[f][0] + gt[f][2] / 2 for f in fs]
                cys = [gt[f][1] + gt[f][3] / 2 for f in fs]
                ax.plot(cxs, cys, color="#999999", lw=1.0, ls="--", alpha=0.9)
                # 预测（按目标着色；只画 update 成功的帧，丢帧处轨迹断开）
                tr = [r for r in rows_t if r["target"] == t and r["ok"]]
                cxs = [r["box"][0] + r["box"][2] / 2 for r in tr]
                cys = [r["box"][1] + r["box"][3] / 2 for r in tr]
                ax.plot(cxs, cys, color=col, lw=1.3, alpha=0.9,
                        label=t if k == 0 else None)
            ax.set_title(f"{name}  meanIoU={ev['otb']['mean_iou']:.3f}", fontsize=10)
            ax.set_xlim(0, 640)
            ax.set_ylim(360, 0)
            ax.set_aspect("equal")
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.legend(fontsize=7, loc="lower right", title="虚线=真值", title_fontsize=7)
        for k in range(n, rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(f"轨迹对比（灰虚线=真值，彩色=跟踪结果，断点处为丢帧）— 序列 {seq}", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"fig_trajectory_{seq}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  {p}")


# ---------------------------------------------------------------- 柱状图
def fig_bars(evals, out_dir):
    g = group(evals)
    tags = sorted({tag_name(e) for e in evals},
                  key=lambda t: -max(e["fps"] for e in evals if tag_name(e) == t))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # FPS
    vals = [max(e["fps"] for e in evals if tag_name(e) == t) for t in tags]
    axes[0].bar(range(len(tags)), vals,
                color=[style(t.split("+")[0])[0] for t in tags])
    axes[0].set_xticks(range(len(tags)))
    axes[0].set_xticklabels([disp(t) for t in tags], rotation=25, ha="right", fontsize=7.5)
    axes[0].set_ylabel("处理速度 / FPS")
    axes[0].set_title("速度对比（越高越快）")
    for i, v in enumerate(vals):
        axes[0].text(i, v * 1.01, f"{v:.0f}", ha="center", fontsize=8)
    # 平均 IoU / 两序列
    width = 0.38
    seqs = list(g)
    for si, seq in enumerate(seqs):
        evs = g[seq]
        vals = [e["otb"]["mean_iou"] for e in evs]
        axes[1].bar(np.arange(len(evs)) + (si - 0.5) * width, vals, width,
                    label=f"{seq}", color=["#4c72b0", "#dd8452"][si % 2])
    axes[1].set_xticks(np.arange(len(g[seqs[0]])))
    axes[1].set_xticklabels([disp(tag_name(e)) for e in g[seqs[0]]],
                            rotation=25, ha="right", fontsize=7.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("平均 IoU")
    axes[1].set_title("精度对比（OTB 平均重叠率）")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
    # FPS - AUC 散点（圆点=cross，三角=occlude；每个算法只标一次名字）
    seen = {}
    for ev in evals:
        name = tag_name(ev)
        marker = "o" if ev["seq"] == "cross" else "^"
        axes[2].scatter(ev["fps"], ev["otb"]["auc"], s=55, marker=marker,
                        color=style(name.split("+")[0])[0])
        if name not in seen:
            seen[name] = True
            axes[2].annotate(disp(name), (ev["fps"], ev["otb"]["auc"]),
                             textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    axes[2].plot([], [], "o", color="#555555", label="cross")
    axes[2].plot([], [], "^", color="#555555", label="occlude")
    axes[2].legend(fontsize=7, loc="lower right")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("FPS（对数轴）")
    axes[2].set_ylabel("Success AUC")
    axes[2].set_title("速度-精度权衡")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig_speed_accuracy.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------- 定性拼图
def fig_qualitative(evals, out_dir, frame_idx=220):
    import cv2
    g = group(evals)
    for seq, evs in g.items():
        tiles = []
        for ev in evs:
            if not ev.get("video") or not Path(ev["video"]).exists():
                continue
            cap = cv2.VideoCapture(ev["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, f = cap.read()
            cap.release()
            if ok:
                tiles.append((tag_name(ev), f))
        if len(tiles) < 2:
            continue
        cols = min(3, len(tiles))
        rows = int(np.ceil(len(tiles) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.0 * rows), squeeze=False)
        for k, (name, f) in enumerate(tiles):
            ax = axes[k // cols][k % cols]
            ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            ax.set_title(name, fontsize=9)
            ax.axis("off")
        for k in range(len(tiles), rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(f"定性对比：{seq} 第 {frame_idx} 帧", fontsize=12)
        fig.tight_layout()
        p = out_dir / f"fig_qualitative_{seq}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  {p}")


# ---------------------------------------------------------------- 遮挡鲁棒性
def fig_robustness(evals, out_dir):
    evs = [e for e in evals if e["seq"] == "occlude"]
    if not evs:
        return
    s = group(evs)["occlude"]
    levels = [("清晰", "mean_iou_clear"), ("部分遮挡", "mean_iou_partial"),
              ("完全遮挡", "mean_iou_full")]
    levels = [(n, k) for n, k in levels if any(k in e["otb"] for e in s)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.8 / len(s)
    for si, ev in enumerate(s):
        vals = [ev["otb"].get(k, np.nan) for _, k in levels]
        xs = np.arange(len(levels)) + (si - (len(s) - 1) / 2) * width
        ax.bar(xs, vals, width, label=tag_name(ev),
               color=style(tag_name(ev).split("+")[0])[0])
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([f"{n}\n(共 {s[0]['otb'].get(k.replace('mean_iou', 'n'), '?')} 帧)"
                        for n, k in levels], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("平均 IoU（与真值重叠率）")
    ax.set_title("遮挡鲁棒性：不同可见程度下的定位精度（序列 occlude）", fontsize=11)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig_robustness.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    print(f"  {p}")


# ---------------------------------------------------------------- 表格
SPEED_FIELDS = ["seq", "algos", "init_ms", "mean_ms", "p95_ms", "fps"]
ACC_FIELDS = ["seq", "algos", "auc", "mean_iou", "op50", "op75", "prec20",
              "mean_err", "fail_frames", "longest_fail"]
MOT_FIELDS = ["seq", "algos", "mota", "motp_iou", "motp_px", "idsw", "frag",
              "fp", "fn", "mt", "ml"]
ROB_FIELDS = ["algos", "n_clear", "mean_iou_clear", "n_partial",
              "mean_iou_partial", "n_full", "mean_iou_full", "keep_partial"]


def write_tables(evals, out_dir):
    ensure_dir(out_dir)
    rows = []
    for ev in evals:
        row = {"seq": ev["seq"], "algos": tag_name(ev),
               "init_ms": round(ev["init_ms"], 2), "mean_ms": round(ev["mean_ms"], 2),
               "p95_ms": round(ev["p95_ms"], 2), "fps": round(ev["fps"], 1)}
        for k in ("auc", "mean_iou", "op50", "op75", "prec20", "mean_err",
                  "fail_frames", "longest_fail"):
            v = ev["otb"].get(k)
            row[k] = round(v, 4) if isinstance(v, float) else v
        for k in ("mota", "motp_iou", "motp_px", "idsw", "frag", "fp", "fn", "mt", "ml"):
            v = ev["mot"].get(k)
            row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)

    def dump(name, fields, title, fmt=None):
        p = out_dir / f"{name}.csv"
        with open(p, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        # Markdown
        md = [f"### {title}", "",
              "| " + " | ".join(fields) + " |",
              "|" + "|".join(["---"] * len(fields)) + "|"]
        for r in rows:
            md.append("| " + " | ".join(str(r.get(k, "")) for k in fields) + " |")
        (out_dir / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print(f"  {p}")

    dump("table_speed", SPEED_FIELDS, "表 跟踪速度实测")
    dump("table_accuracy", ACC_FIELDS, "表 OTB 精度指标")
    dump("table_mot", MOT_FIELDS, "表 MOTChallenge 指标")

    # 遮挡鲁棒性表（只有 occlude 序列有可见比例差异）
    rob = []
    for r in rows:
        if r["seq"] != "occlude":
            continue
        ev = next(e for e in evals if e["seq"] == "occlude" and tag_name(e) == r["algos"])
        o = ev["otb"]
        rec = {k: (round(o[k], 4) if isinstance(o.get(k), float) else o.get(k, ""))
               for k in ROB_FIELDS if k != "keep_partial"}
        mc = o.get("mean_iou_clear")
        mp = o.get("mean_iou_partial")
        rec["keep_partial"] = round(mp / mc, 3) if mc and mp is not None else ""
        rec["algos"] = r["algos"]
        rob.append(rec)
    rob.sort(key=lambda d: -(d.get("mean_iou_partial") or 0))
    p = out_dir / "table_robustness.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=ROB_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rob)
    md = ["### 表 遮挡鲁棒性（序列 occlude，按可见比例分档）", "",
          "| " + " | ".join(ROB_FIELDS) + " |",
          "|" + "|".join(["---"] * len(ROB_FIELDS)) + "|"]
    for r in rob:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in ROB_FIELDS) + " |")
    md.append("")
    md.append("keep_partial = 部分遮挡档平均 IoU ÷ 清晰档平均 IoU，越接近 1 说明越不受遮挡影响。")
    (out_dir / "table_robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  {p}")
    return rows


def copy_to_reports(paths):
    """拷贝到工作区 reports/，加 02_ 前缀以区分实验编号。"""
    ensure_dir(REPORT_FIG_DIR)
    ensure_dir(REPORT_TAB_DIR)
    for p in paths:
        p = Path(p)
        d = REPORT_FIG_DIR if p.parent == FIG_DIR else REPORT_TAB_DIR
        (d / f"02_{p.name}").write_bytes(p.read_bytes())
        # 清掉早期没有前缀的同名文件，避免两套并存
        stale = d / p.name
        if stale.exists():
            stale.unlink()


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="生成实验三图表")
    ap.add_argument("--frame", type=int, default=220, help="定性拼图取第几帧")
    args = ap.parse_args()
    font = setup_font()
    print(f"[i] 中文字体：{font}")

    evals = load_evals(METRIC_DIR)
    if not evals:
        raise SystemExit("没有 eval_*.json，先运行 evaluate.py")
    fig_dir = ensure_dir(FIG_DIR)
    fig_curves(evals, fig_dir)
    fig_iou_timeline(evals, fig_dir)
    fig_trajectories(evals, fig_dir)
    fig_bars(evals, fig_dir)
    fig_robustness(evals, fig_dir)
    fig_qualitative(evals, fig_dir, frame_idx=args.frame)

    tab_dir = ensure_dir(OUT_DIR / "tables")
    write_tables(evals, tab_dir)

    figs = sorted(fig_dir.glob("fig_*.png")) + sorted(fig_dir.glob("seq_*.png"))
    tabs = sorted(tab_dir.glob("table_*.csv")) + sorted(tab_dir.glob("table_*.md"))
    copy_to_reports(figs + tabs)
    print(f"[OK] 图 {len(figs)} 张、表 {len(tabs)} 个，已复制到 reports/{{figures,tables}}")


if __name__ == "__main__":
    main()
