# -*- coding: utf-8 -*-
"""诊断脚本：检查 320 帧特征与标签是否存在帧级对齐关系。

背景：用上游 train.py 训练 FCSN 后，训练集/测试集 AUC 都停在 0.50（等价随机），
但同一份数据上 ridge 回归的测试 AUC=0.652。需要判断到底是
「特征与标签在帧级根本不对齐（数据问题）」还是「FCSN 优化问题」。
"""
import h5py
import numpy as np
import os

# 仓库根目录（由本文件位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根）
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

F1 = os.path.join(BASE, "datasets", "tvsum50", "fcsn_tvsum.h5")
F2 = os.path.join(BASE, "datasets", "tvsum50", "eccv16_dataset_tvsum_google_pool5_local.h5")
IDS = os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN", "logs", "test_ids.txt")


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    return (rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    test_ids = [l.strip().replace("video_", "") for l in open(IDS, encoding="utf-8") if l.strip()]
    f = h5py.File(F1, "r")
    f2 = h5py.File(F2, "r")
    items = {}
    for k in sorted(f.keys()):
        picks = f[k]["picks"][()]
        gs = f2[k]["gtscore"][()]
        items[k[6:]] = dict(fe=f[k]["feature"][()].astype(np.float64),
                            la=f[k]["label"][()].astype(np.float64),
                            gs=gs[picks])
    tr = [k for k in items if k not in test_ids]
    te = [k for k in items if k in test_ids]

    Xtr = np.concatenate([items[k]["fe"] for k in tr])
    ytr = np.concatenate([items[k]["la"] for k in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    X = (Xtr - mu) / sd
    w = np.linalg.solve(X.T @ X + 1e-2 * np.eye(1024), X.T @ (ytr - ytr.mean()))

    for tag, keys in (("测试集", te), ("训练集(前6)", tr[:6])):
        print(tag)
        for k in keys:
            it = items[k]
            s = ((it["fe"] - mu) / sd) @ w
            print("  %-12s r_label=%+.3f  r_gtscore=%+.3f" % (k, np.corrcoef(s, it["la"])[0, 1],
                                                              np.corrcoef(s, it["gs"])[0, 1]))

    print("\n单视频样本内 ridge（上限检验，测试集）:")
    for k in te:
        it = items[k]
        x = (it["fe"] - mu) / sd
        ww = np.linalg.solve(x.T @ x + 1e-1 * np.eye(1024), x.T @ (it["la"] - it["la"].mean()))
        s = x @ ww
        print("  %-12s AUC=%.3f r=%+.3f" % (k, auc(s, it["la"]), np.corrcoef(s, it["la"])[0, 1]))

    print("\n标签与 gtscore 的逐视频相关（检查标签生成是否与标注一致）:")
    for k in te:
        it = items[k]
        print("  %-12s r=%.3f  正样本占比=%.3f  gtscore均值=%.3f"
              % (k, np.corrcoef(it["la"], it["gs"])[0, 1], it["la"].mean(), it["gs"].mean()))


if __name__ == "__main__":
    main()
