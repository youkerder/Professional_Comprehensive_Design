# -*- coding: utf-8 -*-
"""实验 04 · 检索效率对比。

对比四种检索方式（brute / faiss_flat / faiss_ivf / hnsw）在三个规模下的：
  建库耗时、单次查询延迟、QPS、内存占用、Top-10 召回率（与暴力精确解之比）

规模一：真实图像库 N=120（本实验的以图搜图）
规模二：视频帧库 N=192（本实验的以图搜视频帧级检索）
规模三：放大库 N=20000（把库特征重采样加噪声模拟"数据量稍大"的场景，
        用来观察索引方式随规模的渐近差异）

输出：outputs/metrics/efficiency.csv

用法：python code/benchmark.py
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (BENCH_BIG_N, BENCH_BIG_QUERIES, BENCH_QUERIES, BENCH_SEED,
                    FEATS_DIR, METRICS_DIR)
from utils import build_index, l2norm


def make_big_pool(feats, n, seed):
    """用真实库特征重采样 + 少量高斯噪声，构造规模更大的仿真库（保持各向异性）。"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(feats), n)
    X = feats[idx] + rng.normal(0, 0.02, (n, feats.shape[1])).astype(np.float32)
    return l2norm(X)


def bench_index(name, feats, queries, k=10, repeats=1):
    idx, t_build = build_index(name, feats)
    # 预热一次
    idx.search(queries[0], k)
    t0 = time.perf_counter()
    for r in range(repeats):
        for q in queries:
            idx.search(q, k)
    dt = time.perf_counter() - t0
    n_q = len(queries) * repeats
    lat_ms = dt / n_q * 1000.0
    qps = n_q / dt
    # Top-10 召回率（相对暴力精确解）
    exact = l2norm(feats) @ l2norm(queries).T      # (N, Q)
    gt = [set(np.argsort(-exact[:, i])[:k].tolist()) for i in range(len(queries))]
    recs = []
    for i, q in enumerate(queries):
        got = np.asarray(idx.search(q, k), dtype=int)
        got = got[got >= 0][:k]
        recs.append(len(set(got.tolist()) & gt[i]) / k)
    rec = float(np.mean(recs))
    return {"index": name, "n": len(feats), "build_ms": t_build * 1000,
            "latency_ms": lat_ms, "qps": qps,
            "memory_mb": idx.memory_bytes() / 1048576, "recall_at_k": rec}


def main():
    imgs = np.load(os.path.join(FEATS_DIR, "images_resnet50.npy"))
    frames = np.load(os.path.join(FEATS_DIR, "video_frames_resnet50.npy"))
    queries = np.load(os.path.join(FEATS_DIR, "queries_resnet50.npy"))
    rng = np.random.default_rng(BENCH_SEED)

    rows = []
    # ---- 规模一：真实图像库 ----
    print(f"[N={len(imgs)}] 真实图像库")
    for name in ("brute", "faiss_flat", "faiss_ivf", "hnsw"):
        r = bench_index(name, imgs, queries, k=10, repeats=BENCH_QUERIES // len(queries))
        rows.append({**r, "pool": "images_lib"})
        print(f"  {name:<10s} 建库 {r['build_ms']:8.2f} ms  延迟 {r['latency_ms']:8.3f} ms"
              f"  QPS {r['qps']:9.1f}  内存 {r['memory_mb']:7.3f} MB  R@10 {r['recall_at_k']:.3f}")

    # ---- 规模二：视频帧库 ----
    print(f"[N={len(frames)}] 视频帧库")
    for name in ("brute", "faiss_flat", "faiss_ivf", "hnsw"):
        r = bench_index(name, frames, queries, k=10,
                        repeats=BENCH_QUERIES // len(queries))
        rows.append({**r, "pool": "video_frames"})
        print(f"  {name:<10s} 建库 {r['build_ms']:8.2f} ms  延迟 {r['latency_ms']:8.3f} ms"
              f"  QPS {r['qps']:9.1f}  内存 {r['memory_mb']:7.3f} MB  R@10 {r['recall_at_k']:.3f}")

    # ---- 规模三：放大库 ----
    big = make_big_pool(imgs, BENCH_BIG_N, BENCH_SEED)
    bq = queries[rng.integers(0, len(queries), BENCH_BIG_QUERIES)]
    print(f"[N={len(big)}] 放大库（重采样+噪声）")
    for name in ("brute", "faiss_flat", "faiss_ivf", "hnsw"):
        r = bench_index(name, big, bq, k=10, repeats=3)
        rows.append({**r, "pool": f"big_{BENCH_BIG_N}"})
        print(f"  {name:<10s} 建库 {r['build_ms']:8.1f} ms  延迟 {r['latency_ms']:8.3f} ms"
              f"  QPS {r['qps']:9.1f}  内存 {r['memory_mb']:7.1f} MB  R@10 {r['recall_at_k']:.3f}")

    with open(os.path.join(METRICS_DIR, "efficiency.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pool", "n", "index", "build_ms",
                                          "latency_ms", "qps", "memory_mb",
                                          "recall_at_k"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print("\n已保存 outputs/metrics/efficiency.csv")


if __name__ == "__main__":
    main()
