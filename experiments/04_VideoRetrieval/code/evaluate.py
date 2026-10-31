# -*- coding: utf-8 -*-
"""实验 04 · 检索精度评价（两种评价指标：Precision@K 与 mAP）。

  以图搜图（i2i）：查询图 vs 120 张图像库，相关 = 同场景；
  以图搜视频（i2v）：查询图 vs 12 段视频，相关 = 来源视频，三种视频聚合方式：
      mean-pool   查询与"帧特征平均"算相似度
      max-pool    查询与"帧特征逐维最大"算相似度
      frame-max   查询与每一帧算相似度，视频得分 = 各帧最大值

基线：HSV 颜色直方图（128 维，传统方法），检验深度特征的必要性。

输出：
  outputs/metrics/accuracy.csv            汇总（长表）
  outputs/metrics/accuracy_per_query.csv  每个查询图的明细
  outputs/metrics/accuracy_per_scene.csv  每个场景的 AP
  outputs/metrics/summary.md              人读汇总

用法：python code/evaluate.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATS_DIR, METRICS_DIR, P_AT_K_LIST
from utils import cosine_scores, evaluate_ranking, load_meta


def rank_ids(q, g):
    return np.argsort(-cosine_scores(q, g))


def run_eval(entries, per_query_rows, per_scene_rows, task, feature, mode,
             queries, q_meta, gallery_feats):
    """entries: {场景名: 库内相关条目索引集合}（相关 = 同场景的全部库图 / 源视频）。

    ranked 是库内条目索引，AP/P/R 都以「相关条目数 = len(relevant)」为分母。"""
    ap_sum = {m: [] for m in ["mAP"] + [f"P@{k}" for k in P_AT_K_LIST]
              + [f"R@{k}" for k in P_AT_K_LIST]}
    for qi, q in enumerate(queries):
        ranked = rank_ids(q, gallery_feats)
        relevant = entries[q_meta["scenes"][qi]]
        sc = evaluate_ranking(list(map(int, ranked)), relevant, P_AT_K_LIST)
        row = {"query": os.path.basename(q_meta["paths"][qi]),
               "scene": q_meta["scenes"][qi], "task": task,
               "feature": feature, "mode": mode, **{k: round(v, 4) for k, v in sc.items()}}
        per_query_rows.append(row)
        for m, v in sc.items():
            ap_sum[m].append(v)
        per_scene_rows.append({"scene": q_meta["scenes"][qi], "task": task,
                               "feature": feature, "mode": mode,
                               "AP": round(sc["mAP"], 4)})
    summary = {m: float(np.mean(v)) for m, v in ap_sum.items()}
    summary.update({"task": task, "feature": feature, "mode": mode,
                    "n_queries": len(queries)})
    return summary


def main():
    imgs = np.load(os.path.join(FEATS_DIR, "images_resnet50.npy"))
    imgs_h = np.load(os.path.join(FEATS_DIR, "images_hist.npy"))
    img_meta = load_meta(os.path.join(FEATS_DIR, "images_meta.json"))

    qs = np.load(os.path.join(FEATS_DIR, "queries_resnet50.npy"))
    qs_h = np.load(os.path.join(FEATS_DIR, "queries_hist.npy"))
    q_meta = load_meta(os.path.join(FEATS_DIR, "queries_meta.json"))

    v_mean = np.load(os.path.join(FEATS_DIR, "videos_resnet50_mean.npy"))
    v_max = np.load(os.path.join(FEATS_DIR, "videos_resnet50_max.npy"))
    v_frames = np.load(os.path.join(FEATS_DIR, "video_frames_resnet50.npy"))
    v_meta = load_meta(os.path.join(FEATS_DIR, "videos_meta.json"))
    fv = np.array(v_meta["frame_video_idx"])
    v_hist_mean = np.load(os.path.join(FEATS_DIR, "videos_hist_mean.npy"))
    v_hist_frames = None
    if os.path.exists(os.path.join(FEATS_DIR, "videos_hist_frames.npy")):
        v_hist_frames = np.load(os.path.join(FEATS_DIR, "videos_hist_frames.npy"))

    video_names = v_meta["names"]
    scenes = sorted(set(img_meta["scenes"]) | set(q_meta["scenes"]))

    per_query_rows, per_scene_rows, summary_rows = [], [], []

    # ---- 以图搜图 ----
    gal_scenes = img_meta["scenes"]
    # 相关条目集合：场景 s 的查询图 → 库中所有同场景图片的下标
    i2i_entries = {s: {i for i, sc in enumerate(gal_scenes) if sc == s}
                   for s in scenes}
    for feature, qf, gf in (("deep_resnet50", qs, imgs), ("hist_hsv", qs_h, imgs_h)):
        s = run_eval(i2i_entries, per_query_rows, per_scene_rows, "i2i", feature,
                     "global", qf, q_meta, gf)
        summary_rows.append(s)
        print(f"[i2i] {feature:<14s} mAP={s['mAP']:.4f}  "
              f"P@5={s['P@5']:.4f}  P@10={s['P@10']:.4f}")

    # ---- 以图搜视频（三种聚合方式）----
    # 相关条目集合：场景 s 的查询图 → 源视频的下标（单相关项）
    video_entries = {name: {vi} for vi, name in enumerate(video_names)}

    i2v_setups = []
    for feature in ("deep_resnet50", "hist_hsv"):
        if feature == "deep_resnet50":
            i2v_setups += [
                ("deep_resnet50", "mean-pool", v_mean, None),
                ("deep_resnet50", "max-pool", v_max, None),
                ("deep_resnet50", "frame-max", None, (v_frames, fv)),
            ]
        else:
            i2v_setups += [("hist_hsv", "mean-pool", v_hist_mean, None)]
            if v_hist_frames is not None:
                i2v_setups.append(("hist_hsv", "frame-max", None,
                                   (v_hist_frames, fv)))
    for feature, mode, pooled, frameset in i2v_setups:
        qf = qs if feature == "deep_resnet50" else qs_h
        if pooled is not None:
            s = run_eval(video_entries, per_query_rows, per_scene_rows, "i2v",
                         feature, mode, qf, q_meta, pooled)
        else:
            frames, fv_idx = frameset
            # 帧级相似度 → 视频得分 = 该视频各帧最大值
            sim = cosine_scores(qf, frames)             # (Q, F)
            vsim = np.stack([sim[:, fv_idx == vi].max(1) for vi in range(len(video_names))],
                            axis=1)                     # (Q, V)
            order = np.argsort(-vsim, axis=1)
            agg = []
            for qi in range(len(qf)):
                relevant = video_entries[q_meta["scenes"][qi]]
                sc = evaluate_ranking(list(map(int, order[qi])), relevant, P_AT_K_LIST)
                row = {"query": os.path.basename(q_meta["paths"][qi]),
                       "scene": q_meta["scenes"][qi], "task": "i2v",
                       "feature": feature, "mode": mode,
                       **{k: round(v, 4) for k, v in sc.items()}}
                per_query_rows.append(row)
                per_scene_rows.append({"scene": q_meta["scenes"][qi], "task": "i2v",
                                       "feature": feature, "mode": mode,
                                       "AP": round(sc["mAP"], 4)})
                agg.append(sc)
            s = {"task": "i2v", "feature": feature, "mode": mode, "n_queries": len(qf)}
            for m in agg[0]:
                s[m] = float(np.mean([a[m] for a in agg]))
        summary_rows.append(s)
        print(f"[i2v] {feature:<14s} {mode:<10s} mAP={s['mAP']:.4f}  "
              f"P@1={s['P@1']:.4f}  P@3={s['P@3']:.4f}")

    # ---- 落盘 ----
    with open(os.path.join(METRICS_DIR, "accuracy.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "feature", "mode", "n_queries",
                                          "mAP"] + [f"P@{k}" for k in P_AT_K_LIST]
                                          + [f"R@{k}" for k in P_AT_K_LIST])
        w.writeheader()
        w.writerows(summary_rows)
    with open(os.path.join(METRICS_DIR, "accuracy_per_query.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)
    with open(os.path.join(METRICS_DIR, "accuracy_per_scene.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "task", "feature", "mode", "AP"])
        w.writeheader()
        w.writerows(per_scene_rows)

    lines = ["# 检索精度汇总（Precision@K 与 mAP）", "",
             "| 任务 | 特征 | 聚合方式 | mAP | P@1 | P@3 | P@5 | P@10 | R@5 | R@10 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary_rows:
        lines.append("| {task} | {feature} | {mode} | {mAP:.4f} | {P1:.4f} | "
                     "{P3:.4f} | {P5:.4f} | {P10:.4f} | {R5:.4f} | {R10:.4f} |".format(
                         task=s["task"], feature=s["feature"], mode=s["mode"],
                         mAP=s["mAP"], P1=s["P@1"], P3=s["P@3"], P5=s["P@5"],
                         P10=s["P@10"], R5=s["R@5"], R10=s["R@10"]))
    with open(os.path.join(METRICS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
