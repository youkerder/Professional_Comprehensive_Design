# -*- coding: utf-8 -*-
"""实验三 · 跟踪评价（对应“执行步骤 5”）。

一次读完某次运行的轨迹 CSV 与真值，同时算出三套指标的常用子集：

OTB 口径（单目标，逐目标统计）
  * success AUC      —— success plot 曲线下面积（阈值 0~1 扫 IoU）
  * OP@0.5 / OP@0.75 —— IoU ≥ 阈值 的帧占比（overlap precision）
  * precision@20     —— 中心误差 ≤ 20 px 的帧占比
  * 平均中心误差（px）、失败帧数（IoU<0.1）、失败次数（连续失败段数）

MOTChallenge 口径（多目标，帧级匈牙利匹配）
  * MOTA = 1 - (FN + FP + IDSW) / GT      （越大越好）
  * MOTP = 匹配对的平均 IoU / 平均中心距    （越大越好 / 越小越好）
  * IDSW 换号次数、Frag 轨迹断裂次数、MT/ML 多数跟踪/多数丢失

VOT 口径（轻量版）
  * accuracy   = 平均 IoU（仅统计真值可见帧）
  * robustness = 失败（IoU<0.1）次数，等价于需要重初始化的次数
  说明：VOT 官方协议是“失败后按真值重初始化”，本题不重初始化，
  因此 robustness 取“若不重初始化则持续失败”的失败事件数，含义一致、数值可复现。

用法
----
  python evaluate.py                     # 评价 outputs/metrics/run_*.json 里所有运行
  python evaluate.py --seq cross
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from common import (ANN_DIR, METRIC_DIR, ensure_dir, hungarian, iou, load_mot_gt,
                    load_otb_gt, load_trace, load_seq_gt, center_distance,
                    setup_console)

IOU_FAIL = 0.1        # VOT 风格的失败阈值
IOU_MATCH = 0.5       # MOT 帧级匹配阈值
SUCCESS_THRS = np.arange(0.0, 1.0001, 0.01)
PREC_THRS = np.arange(0.0, 51.0, 1.0)


# ---------------------------------------------------------------------------
# 单目标：OTB
# ---------------------------------------------------------------------------
def otb_metrics(pred, gt, vis=None):
    """pred / gt: {frame: (x,y,w,h)}，vis: {frame: 可见比例}"""
    frames = sorted(set(gt) & set(pred))
    if not frames:
        return {"n": 0}
    ious = np.array([iou(pred[f], gt[f]) for f in frames])
    errs = np.array([center_distance(pred[f], gt[f]) for f in frames])

    succ = np.array([(ious >= t).mean() for t in SUCCESS_THRS])
    prec = np.array([(errs <= t).mean() for t in PREC_THRS])

    fail = ious < IOU_FAIL
    # 失败事件数：从“跟住”切到“失败”的次数
    events = int(np.sum(fail[1:] & ~fail[:-1])) + int(fail[0])
    # 最长连续失败
    longest = cur = 0
    for f in fail:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)

    out = {
        "n": len(frames),
        "auc": float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(succ, SUCCESS_THRS)),
        "mean_iou": float(ious.mean()),
        "op50": float((ious >= 0.5).mean()),
        "op75": float((ious >= 0.75).mean()),
        "prec20": float((errs <= 20).mean()),
        "mean_err": float(errs.mean()),
        "median_err": float(np.median(errs)),
        "fail_frames": int(fail.sum()),
        "fail_events": events,
        "longest_fail": int(longest),
        "success_curve": succ.tolist(),
        "precision_curve": prec.tolist(),
    }
    if vis:
        # 按可见比例分三档：清晰(≥0.9) / 部分遮挡(0.05~0.9) / 完全遮挡(<0.05)
        v = np.array([vis.get(f, 1.0) for f in frames])
        buckets = {"clear": v >= 0.9, "partial": (v >= 0.05) & (v < 0.9), "full": v < 0.05}
        for k, m in buckets.items():
            out[f"n_{k}"] = int(m.sum())
            if m.any():
                out[f"mean_iou_{k}"] = float(ious[m].mean())
                out[f"op50_{k}"] = float((ious[m] >= 0.5).mean())
    return out


# ---------------------------------------------------------------------------
# 多目标：MOTChallenge
# ---------------------------------------------------------------------------
def mot_metrics(pred_by_frame, gt_by_frame, target_ids):
    """pred_by_frame: {frame: {name: box}}；gt_by_frame: {frame: {id: box}}

    target_ids: {name: gt_id}，用于把“槽位”对应到真值 ID。
    """
    gt_total = 0
    tp = fp = fn = idsw = 0
    match_ious, match_dists = [], []
    prev_match = {}       # gt_id -> 槽位名
    tracked_hist = {tid: [] for tid in target_ids.values()}
    frag = 0
    prev_tracked = {}

    for f in sorted(set(gt_by_frame) & set(pred_by_frame)):
        gts = gt_by_frame[f]                       # {id: box}
        preds = pred_by_frame[f]                   # {name: box}
        names = [n for n in target_ids if n in preds and target_ids[n] in gts]
        if not names:
            continue
        gt_ids = [target_ids[n] for n in names]
        cost = np.zeros((len(names), len(gt_ids)))
        for i, n in enumerate(names):
            for j, tid in enumerate(gt_ids):
                cost[i, j] = 1.0 - iou(preds[n], gts[tid])
        pairs = hungarian(cost)
        gt_total += len(gt_ids)

        matched_gt = set()
        for i, j in pairs:
            if cost[i, j] <= 1.0 - IOU_MATCH:
                n, tid = names[i], gt_ids[j]
                matched_gt.add(tid)
                tp += 1
                match_ious.append(1.0 - cost[i, j])
                match_dists.append(center_distance(preds[n], gts[tid]))
                if tid in prev_match and prev_match[tid] != n:
                    idsw += 1
                prev_match[tid] = n
                if prev_tracked.get(tid) is False:
                    frag += 1
                prev_tracked[tid] = True
            # 未达阈值的配对按“没配上”处理
        fn += len(gt_ids) - len(matched_gt)
        fp += len(names) - len(matched_gt)
        for tid in gt_ids:
            if tid not in matched_gt:
                prev_tracked[tid] = False
        for tid in gt_ids:
            tracked_hist[tid].append(1 if tid in matched_gt else 0)

    ratios = {tid: (float(np.mean(h)) if h else 0.0) for tid, h in tracked_hist.items()}
    mt = sum(1 for r in ratios.values() if r >= 0.8)
    ml = sum(1 for r in ratios.values() if r <= 0.2)
    mota = 1.0 - (fn + fp + idsw) / gt_total if gt_total else float("nan")
    return {
        "gt_total": gt_total, "tp": tp, "fp": fp, "fn": fn, "idsw": idsw,
        "mota": float(mota),
        "motp_iou": float(np.mean(match_ious)) if match_ious else 0.0,
        "motp_px": float(np.mean(match_dists)) if match_dists else float("nan"),
        "mt": mt, "ml": ml, "frag": frag,
        "recall": tp / gt_total if gt_total else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
    }


# ---------------------------------------------------------------------------
# 评价一次运行
# ---------------------------------------------------------------------------
def evaluate_run(run_json):
    run = json.loads(Path(run_json).read_text(encoding="utf-8"))
    seq, tag = run["seq"], run["tag"]
    rows = load_trace(run["trace"])

    per_target_pred, per_frame_pred = {}, {}
    for r in rows:
        per_target_pred.setdefault(r["target"], {})[r["frame"]] = r["box"]
        per_frame_pred.setdefault(r["frame"], {})[r["target"]] = r["box"]

    otb_gt, mot_gt, _ = load_seq_gt(seq)
    # 可见比例
    vis = {}
    gt_path = ANN_DIR / seq / "gt.txt"
    if gt_path.exists():
        name_by_id = {}
        for line in gt_path.read_text(encoding="utf-8").splitlines():
            p = line.split(",")
            if len(p) >= 9:
                fid = int(float(p[1]))
                name = f"target{fid}"
                vis.setdefault(name, {})[int(float(p[0])) - 1] = float(p[8])

    targets = sorted(per_target_pred)
    target_ids = {n: int(n.replace("target", "")) for n in targets}

    otb = {}
    for n in targets:
        otb[n] = otb_metrics(per_target_pred[n], otb_gt.get(n, {}), vis.get(n))
    mot = mot_metrics(per_frame_pred, mot_gt, target_ids)

    # 汇总（目标间平均）
    keys = ["auc", "mean_iou", "op50", "op75", "prec20", "mean_err", "fail_frames"]
    agg = {k: float(np.mean([otb[n][k] for n in targets if k in otb[n]])) for k in keys}
    agg.update({
        "fail_events": int(sum(otb[n]["fail_events"] for n in targets)),
        "longest_fail": int(max(otb[n]["longest_fail"] for n in targets)),
        "accuracy": agg["mean_iou"],
        "robustness": int(sum(otb[n]["fail_events"] for n in targets)),
    })
    # 可见性分档（按帧数加权合并各目标），用于遮挡鲁棒性表
    for b in ("clear", "partial", "full"):
        ns = sum(int(otb[n].get(f"n_{b}", 0)) for n in targets)
        if ns:
            agg[f"n_{b}"] = ns
            agg[f"mean_iou_{b}"] = float(sum(
                otb[n].get(f"mean_iou_{b}", 0.0) * otb[n].get(f"n_{b}", 0) for n in targets) / ns)
            agg[f"op50_{b}"] = float(sum(
                otb[n].get(f"op50_{b}", 0.0) * otb[n].get(f"n_{b}", 0) for n in targets) / ns)
    out = {"seq": seq, "tag": tag, "backend": run.get("backend"),
           "algos": run.get("algos"), "fps": run.get("fps"), "mean_ms": run.get("mean_ms"),
           "init_ms": run.get("init_ms"), "p95_ms": run.get("p95_ms"),
           "frames": run.get("frames"), "targets": targets,
           "otb_per_target": otb, "otb": agg, "mot": mot,
           "video": run.get("video"), "trace": run.get("trace")}
    (ensure_dir(METRIC_DIR) / f"eval_{seq}_{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


SUMMARY_FIELDS = [
    "seq", "tag", "algos", "backend", "fps", "mean_ms", "p95_ms", "init_ms",
    "auc", "mean_iou", "op50", "op75", "prec20", "mean_err", "fail_frames",
    "fail_events", "longest_fail", "accuracy", "robustness",
    "n_clear", "mean_iou_clear", "op50_clear",
    "n_partial", "mean_iou_partial", "op50_partial",
    "n_full", "mean_iou_full", "op50_full",
    "mota", "motp_iou", "motp_px", "idsw", "frag", "fp", "fn", "mt", "ml",
]


def write_summary(results, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in results:
            row = {"seq": r["seq"], "tag": r["tag"], "algos": "/".join(r["algos"] or []),
                   "backend": r["backend"]}
            for k in ("fps", "mean_ms", "p95_ms", "init_ms"):
                row[k] = round(float(r[k]), 3) if r.get(k) is not None else ""
            for k in ("auc", "mean_iou", "op50", "op75", "prec20", "mean_err",
                      "fail_frames", "fail_events", "longest_fail",
                      "accuracy", "robustness",
                      "mean_iou_clear", "op50_clear",
                      "mean_iou_partial", "op50_partial",
                      "mean_iou_full", "op50_full"):
                row[k] = round(float(r["otb"][k]), 4) if k in r["otb"] else ""
            for k in ("n_clear", "n_partial", "n_full"):
                row[k] = int(r["otb"][k]) if k in r["otb"] else ""
            for k in ("mota", "motp_iou", "motp_px", "idsw", "frag", "fp", "fn", "mt", "ml"):
                row[k] = round(float(r["mot"][k]), 4) if k in r["mot"] else ""
            w.writerow(row)


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="OTB / MOT / VOT 指标评价")
    ap.add_argument("--seq", default=None, help="只评价某序列")
    ap.add_argument("--tag", default=None, help="只评价某个算法组合")
    ap.add_argument("--metrics-dir", default=str(METRIC_DIR))
    args = ap.parse_args()

    mdir = Path(args.metrics_dir)
    runs = sorted(mdir.glob("run_mt_*.json"))
    if not runs:
        raise SystemExit(f"{mdir} 里没有 run_mt_*.json —— 先跑 run_multitracker.py")
    results = []
    for rp in runs:
        run = json.loads(rp.read_text(encoding="utf-8"))
        if args.seq and run["seq"] != args.seq:
            continue
        if args.tag and run["tag"] != args.tag:
            continue
        res = evaluate_run(rp)
        results.append(res)
        a, m = res["otb"], res["mot"]
        print(f"[{res['seq']:8s} {res['tag']:20s}] {res['fps']:7.1f} FPS  "
              f"AUC {a['auc']:.3f}  meanIoU {a['mean_iou']:.3f}  OP@0.5 {a['op50'] * 100:5.1f}%  "
              f"P@20 {a['prec20'] * 100:5.1f}%  err {a['mean_err']:5.1f}px  "
              f"MOTA {m['mota']:+.3f}  MOTP {m['motp_iou']:.3f}  IDSW {m['idsw']}  Frag {m['frag']}")

    if results:
        summary = mdir / "summary.csv"
        write_summary(results, summary)
        print(f"\n汇总表：{summary}  （{len(results)} 次运行）")


if __name__ == "__main__":
    main()
