# -*- coding: utf-8 -*-
"""实验 04 · 公共工具：文件遍历、相似度、评价与索引实现。"""
import json
import os
import time

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- 文件与标签
def list_images(root):
    """递归列出目录下所有图片（排序保证可复现）。"""
    out = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def scene_of(path):
    """标签约定：图片路径里倒数第二级目录名即场景 id（videos/xxx.mp4 → xxx）。"""
    return os.path.basename(os.path.dirname(os.path.abspath(path)))


def query_scene(path):
    """查询图的场景 id：从文件名解析（sunset_sea_q0.jpg → sunset_sea）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_q", 1)[0] if "_q" in stem else stem


def save_meta(path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False, indent=1)


def load_meta(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 相似度
def l2norm(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def cosine_scores(query, gallery):
    """query (D,) 或 (Q,D)，gallery (N,D)，均已/将做 L2 归一化 → (N,) 或 (Q,N) 内积。"""
    q = l2norm(query)
    g = l2norm(gallery)
    return q @ g.T if q.ndim == 2 else g @ q


# ---------------------------------------------------------------- 评价
def precision_at_k(ranked_ids, relevant_set, k):
    if k <= 0:
        return 0.0
    topk = ranked_ids[:k]
    return float(sum(1 for r in topk if r in relevant_set)) / k


def recall_at_k(ranked_ids, relevant_set, k):
    if not relevant_set:
        return 0.0
    topk = ranked_ids[:k]
    return float(sum(1 for r in topk if r in relevant_set)) / len(relevant_set)


def average_precision(ranked_ids, relevant_set):
    """AP：相关项出现位置的 Precision 平均（排序越靠前贡献越大）。"""
    if not relevant_set:
        return 0.0
    hits, ap = 0, 0.0
    for i, r in enumerate(ranked_ids, start=1):
        if r in relevant_set:
            hits += 1
            ap += hits / i
    return ap / len(relevant_set)


def evaluate_ranking(ranked_ids, relevant_set, p_at_k=(1, 3, 5, 10)):
    out = {"mAP": average_precision(ranked_ids, relevant_set)}
    for k in p_at_k:
        out[f"P@{k}"] = precision_at_k(ranked_ids, relevant_set, k)
        out[f"R@{k}"] = recall_at_k(ranked_ids, relevant_set, k)
    return out


# ---------------------------------------------------------------- 索引实现
class BruteForceIndex:
    """暴力检索：与全库逐一算内积（L2 归一化后即余弦相似度）。"""

    name = "brute"

    def __init__(self, feats):
        self.feats = l2norm(np.ascontiguousarray(feats, dtype=np.float32))

    def build(self):
        pass  # 建库 = 一次 L2 归一化，在构造函数中已完成

    def search(self, q, k):
        s = l2norm(q[None, :])[0] @ self.feats.T
        idx = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return idx[np.argsort(-s[idx])]

    def memory_bytes(self):
        return self.feats.nbytes


class FaissFlatIndex:
    """FAISS IndexFlatIP：精确内积检索，C++ 实现的暴力检索。"""

    name = "faiss_flat"

    def __init__(self, feats):
        import faiss

        self.d = feats.shape[1]
        self.index = faiss.IndexFlatIP(self.d)
        self.index.add(l2norm(np.ascontiguousarray(feats, dtype=np.float32)))

    def build(self):
        pass

    def search(self, q, k):
        _, idx = self.index.search(l2norm(q[None, :]).astype(np.float32), k)
        return idx[0]

    def memory_bytes(self):
        return self.d * self.index.ntotal * 4


class FaissIVFIndex:
    """FAISS IndexIVFFlat：倒排文件索引（k-means 聚类 + 类内精确检索）。

    nlist 按 faiss 经验法则自适应：nlist ≈ 4·sqrt(N)，且不超过 N/39（每簇至少
    39 个训练点）；nprobe 取 nlist/16（至少 1），兼顾召回与剪枝。"""

    name = "faiss_ivf"

    def __init__(self, feats, nlist=None, nprobe=None):
        import faiss

        x = l2norm(np.ascontiguousarray(feats, dtype=np.float32))
        self.d = x.shape[1]
        n = x.shape[0]
        if nlist is None:
            nlist = max(1, min(int(4 * np.sqrt(n)), n // 39))
        if nprobe is None:
            nprobe = max(1, nlist // 16)
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)
        quantizer = faiss.IndexFlatIP(self.d)
        self.index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist,
                                        faiss.METRIC_INNER_PRODUCT)
        self.index.train(x)
        self.index.add(x)
        self.index.nprobe = self.nprobe

    def build(self):
        pass

    def search(self, q, k):
        _, idx = self.index.search(l2norm(q[None, :]).astype(np.float32), k)
        return idx[0][idx[0] >= 0]

    def memory_bytes(self):
        return self.d * self.index.ntotal * 4


class HNSWIndex:
    """hnswlib HNSW 图索引：分层可导航小世界图，近似检索。"""

    name = "hnsw"

    def __init__(self, feats, M=16, ef_construction=200, ef_search=64):
        import hnswlib

        x = np.ascontiguousarray(feats, dtype=np.float32)
        self.n, self.d = x.shape[0], x.shape[1]
        self.ef_search = ef_search
        self.index = hnswlib.Index(space="ip", dim=self.d)
        self.index.init_index(max_elements=self.n, ef_construction=ef_construction, M=M)
        self.index.add_items(l2norm(x), list(range(self.n)))

    def build(self):
        pass

    def search(self, q, k):
        self.index.set_ef(self.ef_search)
        idx, _ = self.index.knn_query(l2norm(q[None, :]).astype(np.float32), k=k)
        return idx[0]

    def memory_bytes(self):
        return int(self.index.get_current_count() * (self.d * 4 + 8 * 2) * 1.6)


def build_index(name, feats):
    makers = {
        "brute": BruteForceIndex,
        "faiss_flat": FaissFlatIndex,
        "faiss_ivf": FaissIVFIndex,
        "hnsw": HNSWIndex,
    }
    if name not in makers:
        raise ValueError(f"未知索引类型：{name}")
    t0 = time.perf_counter()
    obj = makers[name](feats)
    obj.build()
    return obj, time.perf_counter() - t0


# ---------------------------------------------------------------- 排序工具
def rank_all(query, gallery_feats):
    """全库余弦排序，返回按相似度降序的 id 数组。"""
    s = cosine_scores(query, gallery_feats)
    return np.argsort(-s)
