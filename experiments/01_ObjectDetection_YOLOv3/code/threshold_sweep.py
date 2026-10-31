#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阈值扫描评测：对一个训练好的模型，在多个置信度阈值下算指标，挑出最佳阈值。

======================================================================
为什么需要
======================================================================
keras-yolo3 的损失里，背景项数量（每张图约 10600 个预测）远远多于正样本
（1~2 个），模型会把置信度整体压低——本实验实测所有框的分数都 < 0.02。
后果是：**单一固定阈值（如 0.3）会一个框都检不出**，必须扫一遍阈值看哪一档
Precision / Recall / AP 的平衡最好。

做法：对每个阈值跑一次 detect_images.py + evaluate_detection.py，
汇总成对比表，并按 AP 选出最佳阈值。

用法：
    python threshold_sweep.py --weights ../models/xxx.h5
    python threshold_sweep.py --weights <权重> --anchors model_data/yolo_anchors_catface.txt
    python threshold_sweep.py --weights <权重> --thresholds 0.5,0.3,0.1,0.03,0.01,0.003,0.001
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from config import ANCHORS_PATH, CODE_DIR, OUTPUTS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ANCHORS_PATH, CODE_DIR, OUTPUTS_DIR

DEFAULT_THRESHOLDS = "0.5,0.3,0.1,0.03,0.01,0.003,0.001"


def run(cmd, cwd=None):
    return subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)


def main():
    ap = argparse.ArgumentParser(
        description="在多个置信度阈值下评测模型，挑出最佳阈值")
    ap.add_argument("--weights", required=True, help="训练好的 .h5 权重")
    ap.add_argument("--anchors", default=None, help="anchor 文件（要与训练时一致）")
    ap.add_argument("--split", default="test", help="用哪个划分评测（默认 test）")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                    help="逗号分隔的阈值列表，从大到小")
    ap.add_argument("--iou", type=float, default=0.5, help="评测的 IoU 阈值")
    ap.add_argument("--tf-python", default=None,
                    help="跑检测用的 python（需要 TensorFlow）；默认用当前解释器")
    ap.add_argument("--out", default=None, help="结果输出目录")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    weights = Path(args.weights).resolve()
    tf_py = Path(args.tf_python).resolve() if args.tf_python else Path(sys.executable)
    anchors = Path(args.anchors).resolve() if args.anchors else ANCHORS_PATH

    thresholds = []
    for t in args.thresholds.split(","):
        t = t.strip()
        if t:
            thresholds.append(float(t))
    thresholds.sort(reverse=True)

    out_root = Path(args.out).resolve() if args.out else (
        OUTPUTS_DIR / "sweep" / weights.stem)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("阈值扫描评测")
    print("=" * 72)
    print("权重     : %s" % weights)
    print("anchor   : %s" % anchors.name)
    print("划分     : %s" % args.split)
    print("阈值     : %s" % ", ".join("%g" % t for t in thresholds))
    print("输出目录 : %s" % out_root)
    print("")

    if not weights.is_file():
        print("[错误] 找不到权重文件：%s" % weights)
        return 1
    if not anchors.is_file():
        print("[错误] 找不到 anchor 文件：%s" % anchors)
        return 1

    rows = []
    t_all = time.time()

    for i, thr in enumerate(thresholds, 1):
        tag = ("%g" % thr).replace(".", "p")
        det_dir = out_root / ("detect_%s" % tag)
        met_dir = out_root / ("metrics_%s" % tag)

        print("-" * 72)
        print("[%d/%d] 阈值 %g" % (i, len(thresholds), thr))
        print("-" * 72)

        rc = run([tf_py, CODE_DIR / "detect_images.py",
                  "--weights", weights,
                  "--split", args.split,
                  "--score", "%g" % thr,
                  "--anchors", anchors,
                  "--output", det_dir], cwd=CODE_DIR)
        if rc != 0:
            print("  [警告] 检测失败（退出码 %d），跳过该阈值" % rc)
            continue

        rc = run([sys.executable, CODE_DIR / "evaluate_detection.py",
                  "--pred", det_dir / "detections.csv",
                  "--iou", "%g" % args.iou,
                  "--out", met_dir], cwd=CODE_DIR)
        if rc != 0:
            print("  [警告] 评测失败（退出码 %d），跳过该阈值" % rc)
            continue

        mfile = met_dir / "detection_metrics.json"
        if not mfile.is_file():
            continue
        m = json.loads(mfile.read_text(encoding="utf-8-sig"))
        cat = list(m.get("classes", {}).values())
        if not cat:
            continue
        c = cat[0]
        rows.append({
            "threshold": thr,
            "n_pred": c["n_pred"], "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "precision": c["precision"], "recall": c["recall"],
            "f1": c["f1"], "ap": c["ap"],
            "metrics_dir": str(met_dir), "detect_dir": str(det_dir),
        })
        print("  -> P=%.4f R=%.4f F1=%.4f AP=%.4f（预测 %d，TP %d）"
              % (c["precision"], c["recall"], c["f1"], c["ap"],
                 c["n_pred"], c["tp"]))

    if not rows:
        print("")
        print("[错误] 所有阈值都没有产出有效指标。")
        return 1

    # ---------------- 汇总 ----------------
    print("")
    print("=" * 72)
    print("汇总（IoU = %.2f）" % args.iou)
    print("=" * 72)
    print("%-10s %8s %6s %6s %6s %10s %8s %8s %8s"
          % ("阈值", "预测数", "TP", "FP", "FN", "Precision", "Recall", "F1", "AP"))
    print("-" * 72)
    for r in rows:
        print("%-10g %8d %6d %6d %6d %10.4f %8.4f %8.4f %8.4f"
              % (r["threshold"], r["n_pred"], r["tp"], r["fp"], r["fn"],
                 r["precision"], r["recall"], r["f1"], r["ap"]))

    best_ap = max(rows, key=lambda r: r["ap"])
    # F1 最优的阈值通常更适合"实际可用"的汇报口径
    best_f1 = max(rows, key=lambda r: r["f1"])
    print("")
    print("按 AP 最佳    : 阈值 %g  (AP=%.4f, P=%.4f, R=%.4f)"
          % (best_ap["threshold"], best_ap["ap"], best_ap["precision"], best_ap["recall"]))
    print("按 F1 最佳    : 阈值 %g  (F1=%.4f, P=%.4f, R=%.4f)"
          % (best_f1["threshold"], best_f1["f1"], best_f1["precision"], best_f1["recall"]))
    print("")
    print("注：理论上 AP 只看预测的**排序**，与阈值无关；但本流程里阈值是在")
    print("    检测阶段就过滤掉的（detect_images.py 的 --score），所以高阈值会")
    print("    先丢掉一部分预测、让排序不完整，算出来的 AP 偏低。")
    print("    因此**最低阈值那一档的 AP 最接近真实值**，报告里建议以它为准，")
    print("    同时另给一档高阈值的 P/R/F1 说明实际使用时的取舍。")

    # ---------------- 落盘 ----------------
    csv_path = out_root / "sweep.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = ["# 阈值扫描结果", "",
          "权重：`%s`　anchor：`%s`　IoU 阈值：%.2f" % (weights.name, anchors.name, args.iou),
          "",
          "| 置信度阈值 | 预测数 | TP | FP | FN | Precision | Recall | F1 | AP |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append("| %g | %d | %d | %d | %d | %.4f | %.4f | %.4f | %.4f |"
                  % (r["threshold"], r["n_pred"], r["tp"], r["fp"], r["fn"],
                     r["precision"], r["recall"], r["f1"], r["ap"]))
    md += ["", "按 AP 最佳：阈值 %g（AP=%.4f）" % (best_ap["threshold"], best_ap["ap"]),
           "按 F1 最佳：阈值 %g（F1=%.4f）" % (best_f1["threshold"], best_f1["f1"])]
    md_path = out_root / "sweep.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("")
    print("已写入：%s" % csv_path)
    print("已写入：%s" % md_path)
    print("总用时：%.1f 分钟" % ((time.time() - t_all) / 60))
    print("")
    print("下一步（用最佳阈值的检测结果出报告图）：")
    print("  python make_report_figures.py --logs <训练日志目录> \\")
    print("      --metrics \"%s\" --detect \"%s\""
          % (best_ap["metrics_dir"], best_ap["detect_dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
