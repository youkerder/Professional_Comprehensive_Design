# -*- coding: utf-8 -*-
"""实验 04 · 建立检索索引。

对图像库特征（120×2048）建立四种索引并保存：
  brute        暴力检索（numpy 矩阵乘）
  faiss_flat   FAISS IndexFlatIP（C++ 暴力检索）
  faiss_ivf    FAISS IndexIVFFlat（倒排 + 类内精确）
  hnsw         hnswlib HNSW 图索引

用法：python code/build_index.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATS_DIR, INDEX_DIR
from utils import build_index, l2norm


def main():
    feats = np.load(os.path.join(FEATS_DIR, "images_resnet50.npy"))
    print(f"图像库特征：{feats.shape}")
    report = {}
    for name in ("brute", "faiss_flat", "faiss_ivf", "hnsw"):
        idx, t_build = build_index(name, l2norm(feats))
        size_kb = idx.memory_bytes() / 1024
        report[name] = {"build_s": round(t_build, 4), "memory_kb": round(size_kb, 1)}
        print(f"  {name:<10s} 建库 {t_build * 1000:7.2f} ms   内存 {size_kb:8.1f} KB")
        # 简单自检：用库里第一张图检索，应命中的是自己
        hit = idx.search(feats[0], 1)[0]
        assert int(hit) == 0, f"{name} 自检失败：{hit}"
    np.save(os.path.join(INDEX_DIR, "index_report.npy"), report, allow_pickle=True)
    print("索引自检通过，报告已保存。")


if __name__ == "__main__":
    main()
