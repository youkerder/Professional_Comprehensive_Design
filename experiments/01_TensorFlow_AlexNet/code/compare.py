#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总四组对比实验（2 种激活函数 × 2 种学习率），出对比表与对比图。

输入（由 train.py / evaluate.py 产生）
-------------------------------------
    outputs/eval/<run>_history.json    训练历史（每轮的 loss / accuracy / val_*）
    outputs/eval/<run>_metrics.json    测试集指标（accuracy / top3 / macro / per_class）

输出
----
    outputs/compare/summary.csv            对比表（CSV）
    outputs/compare/summary.md             对比表（Markdown）
    outputs/compare/curves_val_acc.png     四组验证集准确率曲线
    outputs/compare/curves_loss.png        四组训练损失曲线
    outputs/compare/test_accuracy.png      测试集准确率柱状图
    reports/tables/01b_compare.md          报告用表格
    reports/figures/01b_*.png              报告用插图

用法
----
    python compare.py                 # 汇总已有结果
    python compare.py --selftest      # 用合成数据校验汇总逻辑（不需要 TensorFlow）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    from config import (ACTIVATIONS, ACT_LABELS, COMPARE_DIR, EVAL_DIR,
                        LEARNING_RATES, REPORT_FIG_DIR, REPORT_TAB_DIR, all_runs)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (ACTIVATIONS, ACT_LABELS, COMPARE_DIR, EVAL_DIR,
                        LEARNING_RATES, REPORT_FIG_DIR, REPORT_TAB_DIR, all_runs)


# --------------------------------------------------------------------------
# 载入与汇总（纯 python/numpy，便于自检）
# --------------------------------------------------------------------------

def load_run(run: str, eval_dir: Path = None):
    """读一组的训练历史与测试指标。缺文件时对应项为 None。"""
    d = Path(eval_dir) if eval_dir else EVAL_DIR
    hist_path = d / ("%s_history.json" % run)
    met_path = d / ("%s_metrics.json" % run)

    hist = None
    if hist_path.is_file():
        try:
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception:
            hist = None

    met = None
    if met_path.is_file():
        try:
            met = json.loads(met_path.read_text(encoding="utf-8"))
        except Exception:
            met = None
    return hist, met


def summarize(records):
    """把若干组记录整理成对比表的行。

    records: [(run_name, act, lr, hist, met), ...]
    返回 list[dict]，字段：run/act/lr/epochs/best_val_acc/final_val_acc/
                            test_acc/top3/macro_f1/elapsed_sec
    """
    rows = []
    for run, act, lr, hist, met in records:
        row = {
            "run": run, "act": act, "lr": lr,
            "act_label": ACT_LABELS.get(act, act),
            "epochs": None, "best_val_acc": None, "final_val_acc": None,
            "test_acc": None, "top3": None, "macro_f1": None,
            "elapsed_sec": None,
        }
        if hist:
            va = [v for v in hist.get("val_accuracy", []) if v is not None]
            if va:
                row["best_val_acc"] = float(max(va))
                row["final_val_acc"] = float(va[-1])
            ep = hist.get("loss", [])
            row["epochs"] = len(ep) if ep else None
            meta = hist.get("_meta", {})
            if meta.get("elapsed_sec") is not None:
                row["elapsed_sec"] = float(meta["elapsed_sec"])
        if met:
            row["test_acc"] = met.get("accuracy")
            row["top3"] = met.get("top3_accuracy")
            row["macro_f1"] = (met.get("macro") or {}).get("f1")
        rows.append(row)
    return rows


def fmt(v, nd=4, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return str(v)


def rows_to_md(rows) -> str:
    """生成 Markdown 对比表。"""
    lines = [
        "| 激活函数 | 学习率 | 训练轮数 | 最佳验证准确率 | 末轮验证准确率 | 测试准确率 | top-3 准确率 | macro-F1 | 训练耗时 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cost = "%.2f h" % (r["elapsed_sec"] / 3600) if r["elapsed_sec"] else "—"
        lines.append("| %s | %g | %s | %s | %s | %s | %s | %s | %s |"
                     % (r["act_label"], r["lr"], fmt(r["epochs"], 0),
                        fmt(r["best_val_acc"]), fmt(r["final_val_acc"]),
                        fmt(r["test_acc"]), fmt(r["top3"]),
                        fmt(r["macro_f1"]), cost))
    return "\n".join(lines) + "\n"


def group_effect(rows, metric="test_acc"):
    """按"激活函数"和"学习率"分别求平均，看各自的影响。"""
    out = {"by_act": {}, "by_lr": {}}
    for act in ACTIVATIONS:
        vals = [r[metric] for r in rows if r["act"] == act and r[metric] is not None]
        if vals:
            out["by_act"][ACT_LABELS.get(act, act)] = float(np.mean(vals))
    for lr in LEARNING_RATES:
        vals = [r[metric] for r in rows if r["lr"] == lr and r[metric] is not None]
        if vals:
            out["by_lr"]["%g" % lr] = float(np.mean(vals))
    return out


# --------------------------------------------------------------------------
# 画图
# --------------------------------------------------------------------------

def _plt():
    """取配置好的 matplotlib.pyplot。

    中文字体、无界面后端、DPI 等统一在 `config.setup_matplotlib()` 里设置，
    避免两个画图脚本各写一份、其中一处漏掉字体（本实验踩过这个坑）。
    """
    from config import setup_matplotlib
    return setup_matplotlib()


def fig_curves(plt, records, key, ylabel, title, out: Path):
    """画四组的某条曲线。key 为 history 里的字段名。"""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    n = 0
    for run, act, lr, hist, _met in records:
        if not hist or key not in hist:
            continue
        vals = hist[key]
        if not vals:
            continue
        ax.plot(range(1, len(vals) + 1), vals, marker="o", ms=3, lw=1.6,
                label="%s, lr=%g" % (ACT_LABELS.get(act, act), lr))
        n += 1
    if n == 0:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_bar(plt, rows, out: Path):
    """测试准确率柱状图：按激活函数分组、学习率并排。"""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(len(ACTIVATIONS))
    width = 0.8 / max(1, len(LEARNING_RATES))
    for i, lr in enumerate(LEARNING_RATES):
        vals, errs = [], []
        for act in ACTIVATIONS:
            v = [r["test_acc"] for r in rows
                 if r["act"] == act and r["lr"] == lr and r["test_acc"] is not None]
            vals.append(float(np.mean(v)) if v else 0.0)
            errs.append(0.0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label="lr = %g" % lr, yerr=errs, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([ACT_LABELS.get(a, a) for a in ACTIVATIONS])
    ax.set_ylabel("测试集准确率")
    ax.set_ylim(0, 1)
    ax.set_title("激活函数与学习率对测试准确率的影响")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def selftest() -> int:
    """用合成记录校验汇总、分组统计与表格生成。不需要 TensorFlow。"""
    fails = []

    def check(cond, msg):
        print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
        if not cond:
            fails.append(msg)

    print("compare.py 汇总逻辑自检（不需要 TensorFlow）")
    print("-" * 74)

    def mk(act, lr, accs, test_acc, f1=0.5):
        hist = {"loss": [1.0] * len(accs), "val_accuracy": accs,
                "_meta": {"elapsed_sec": 100.0}}
        met = {"accuracy": test_acc, "top3_accuracy": test_acc + 0.05,
               "macro": {"f1": f1}}
        return (act.replace("_", "") + "-lr%g" % lr, act, lr, hist, met)

    records = [
        mk("relu", 0.01, [0.5, 0.6, 0.7], 0.70, 0.69),
        mk("relu", 0.001, [0.3, 0.4, 0.45], 0.45, 0.44),
        mk("leaky_relu", 0.01, [0.51, 0.61, 0.72], 0.72, 0.71),
        mk("leaky_relu", 0.001, [0.31, 0.41, 0.46], 0.46, 0.45),
    ]
    rows = summarize(records)
    check(len(rows) == 4, "四组都进了汇总表")
    check(rows[0]["best_val_acc"] == 0.7, "最佳验证准确率取最大值")
    check(rows[0]["final_val_acc"] == 0.7, "末轮验证准确率取最后一个")
    check(rows[0]["epochs"] == 3, "训练轮数正确")
    check(rows[2]["test_acc"] == 0.72, "测试准确率正确读取")

    eff = group_effect(rows, "test_acc")
    check(abs(eff["by_act"]["ReLU"] - (0.70 + 0.45) / 2) < 1e-9, "按激活函数求平均正确")
    check(abs(eff["by_act"]["LeakyReLU"] - (0.72 + 0.46) / 2) < 1e-9,
          "LeakyReLU 的均值正确")
    check(abs(eff["by_lr"]["0.01"] - (0.70 + 0.72) / 2) < 1e-9, "按学习率求平均正确")
    check(abs(eff["by_lr"]["0.001"] - (0.45 + 0.46) / 2) < 1e-9, "lr=0.001 的均值正确")

    md = rows_to_md(rows)
    check(md.count("\n") == 6, "Markdown 表为表头+分隔+4 行")
    check("LeakyReLU" in md and "ReLU" in md, "表里包含两种激活函数")
    check("0.01" in md and "0.001" in md, "表里包含两种学习率")

    # 缺失数据不能崩
    rows2 = summarize([("x", "relu", 0.01, None, None)])
    check(rows2[0]["test_acc"] is None, "缺指标文件时字段为 None")
    check("—" in rows_to_md(rows2), "缺失值显示为占位符而不是崩溃")
    check(group_effect(rows2, "test_acc")["by_act"] == {},
          "全缺失时分组统计为空字典而不是报错")

    check(fmt(None) == "—", "fmt(None) 返回占位符")
    check(fmt(0.5) == "0.5000", "fmt 浮点保留 4 位")
    check(fmt(3.0, 0) == "3", "fmt 可指定小数位")

    print("-" * 74)
    if fails:
        print("失败 %d 项：" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("全部通过。")
    return 0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="汇总四组对比实验")
    ap.add_argument("--selftest", action="store_true", help="只跑汇总逻辑自检")
    ap.add_argument("--report", action="store_true", default=True,
                    help="同时写 reports/（默认开启）")
    ap.add_argument("--no-report", dest="report", action="store_false")
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.selftest:
        return selftest()

    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("实验一 · 四组对比汇总")
    print("=" * 74)

    records = []
    missing = []          # [(run, act, lr), ...]
    for run, act, lr in all_runs():
        hist, met = load_run(run)
        records.append((run, act, lr, hist, met))
        flags = []
        if hist is None:
            flags.append("无训练历史")
        if met is None:
            flags.append("无测试指标")
        print("  %-20s %-11s lr=%-7g %s"
              % (run, ACT_LABELS.get(act, act), lr,
                 "就绪" if not flags else "缺：" + "、".join(flags)))
        if flags:
            missing.append((run, act, lr))

    rows = summarize(records)
    md = rows_to_md(rows)

    print("")
    print("=" * 74)
    print("对比表")
    print("=" * 74)
    print(md)

    eff = group_effect(rows, "test_acc")
    if eff["by_act"]:
        print("按激活函数（测试准确率均值）：")
        for k, v in eff["by_act"].items():
            print("  %-12s %.4f" % (k, v))
    if eff["by_lr"]:
        print("按学习率（测试准确率均值）：")
        for k, v in eff["by_lr"].items():
            print("  lr=%-8s %.4f" % (k, v))

    # ---------------- 落盘 ----------------
    csv_path = COMPARE_DIR / "summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (COMPARE_DIR / "summary.md").write_text(md, encoding="utf-8")
    print("")
    print("对比表 CSV : %s" % csv_path)

    # ---------------- 画图 ----------------
    figs = []
    try:
        plt = _plt()
        figs.append(fig_curves(plt, records, "val_accuracy", "验证集准确率",
                               "四组验证集准确率曲线", COMPARE_DIR / "curves_val_acc.png"))
        figs.append(fig_curves(plt, records, "loss", "训练损失",
                               "四组训练损失曲线", COMPARE_DIR / "curves_loss.png"))
        figs.append(fig_bar(plt, rows, COMPARE_DIR / "test_accuracy.png"))
        for f in figs:
            if f:
                print("对比图     : %s" % f)
    except Exception as e:
        print("[提示] 画图失败（不影响表格）：%s: %s" % (type(e).__name__, e))

    if args.report:
        REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_TAB_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_TAB_DIR / "01b_compare.md").write_text(
            "# 四组对比汇总（激活函数 × 学习率）\n\n" + md, encoding="utf-8")
        print("报告表格   : %s" % (REPORT_TAB_DIR / "01b_compare.md"))
        import shutil
        for f in figs:
            if f and f.is_file():
                shutil.copy2(f, REPORT_FIG_DIR / ("01b_" + f.name))
        print("报告插图   : %s/01b_curves_*.png、01b_test_accuracy.png" % REPORT_FIG_DIR)

    print("")
    if missing:
        print("[待办] 以下组还缺结果，先跑 train.py / evaluate.py：")
        for run, act, lr in missing:
            # 用 config 里的 act 原名（带下划线，如 leaky_relu），
            # 不能从 run_name 反推——run_name 里是去掉下划线的 leakyrelu，
            # 直接抄给 argparse 会因为不在 choices 里而报错。
            print("  python train.py --act %s --lr %g" % (act, lr))
        return 1
    print("[ OK ] 四组结果齐全。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
