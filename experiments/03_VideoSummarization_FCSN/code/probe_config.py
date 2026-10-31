# -*- coding: utf-8 -*-
"""训练配方探针 v2：定位 FCSN 在自建 TVSum 特征上「完全学不动」的原因。

已确认的事实（见 logs/diag_*.log）：
- 数据本身有信号：同一份 h5 上，ridge / numpy 逻辑回归在测试视频上 AUC=0.652；
- 用上游原版 train.py 跑 100 epoch（800 次更新）后，全部 50 段视频整体 AUC=0.491，
  等价于随机；
- 把标签和特征固定成单个视频让 FCSN 过拟合 300 次更新，AUC 仍只有 0.47~0.52，
  但同样数据上 numpy 逻辑回归（lr=0.5, 3000 次迭代）能拟合到 AUC=1.0
  → 不是数据/对齐问题，而是「更新次数 + 输入尺度 + 类别权重」这一侧的问题。

本脚本按 epoch 打印 train/test AUC，用来判断是「欠训练」还是「配方错误」。

用法：
  python probe_config.py --std --weight none --lr 1e-3 --epochs 150
"""
import argparse

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from fcsn import FCSN
import os

# 仓库根目录（由本文件位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根）
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATA = os.path.join(BASE, "datasets", "tvsum50", "fcsn_tvsum.h5")
IDS = os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN", "logs", "test_ids.txt")


def auc(score, label):
    score = np.asarray(score)
    label = np.asarray(label)
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    return (rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--std", action="store_true", help="按训练集统计量做逐通道标准化")
    ap.add_argument("--weight", default="none", choices=["none", "upstream", "inverse"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()

    test_ids = [l.strip().replace("video_", "") for l in open(IDS, encoding="utf-8") if l.strip()]
    with h5py.File(DATA, "r") as f:
        items = {}
        for k in sorted(f.keys()):
            g = f[k]
            items[k[6:]] = dict(feature=g["feature"][()].astype(np.float32),
                                label=g["label"][()].astype(np.int64))
    tr_ids = [k for k in items if k not in test_ids]
    te_ids = [k for k in items if k in test_ids]

    mu = np.stack([items[k]["feature"] for k in tr_ids]).mean(axis=(0, 1))
    sd = np.stack([items[k]["feature"] for k in tr_ids]).std(axis=(0, 1)) + 1e-6
    mu_t = torch.tensor(mu).cuda()
    sd_t = torch.tensor(sd).cuda()

    def prep(k):
        x = torch.tensor(items[k]["feature"]).cuda()
        if args.std:
            x = (x - mu_t) / sd_t
        return x.t().unsqueeze(0)

    feats = {k: prep(k) for k in items}
    labels = {k: torch.tensor(items[k]["label"]).cuda() for k in items}

    torch.manual_seed(0)
    model = FCSN(2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(0)

    def evaluate(ids):
        model.eval()
        scores, gts = [], []
        with torch.no_grad():
            for k in ids:
                p = torch.softmax(model(feats[k]), dim=1)[0, 1].cpu().numpy()
                scores.append(p)
                gts.append(items[k]["label"])
        model.train()
        return auc(np.concatenate(scores), np.concatenate(gts))

    print("配方: std=%s weight=%s lr=%g epochs=%d clip=%s"
          % (args.std, args.weight, args.lr, args.epochs, args.clip))
    print("epoch  trainAUC  testAUC  loss")
    for ep in range(args.epochs + 1):
        if ep % args.eval_every == 0:
            print("  %3d    %.3f     %.3f" % (ep, evaluate(tr_ids), evaluate(te_ids)), end="")
            print("   -" if ep == 0 else "")
        if ep == args.epochs:
            break
        order = rng.permutation(tr_ids)
        total = 0.0
        for start in range(0, len(order), 5):
            batch = order[start:start + 5]
            opt.zero_grad()
            losses = []
            for k in batch:
                out = model(feats[k])
                y = labels[k].unsqueeze(0)
                n_pos = float(y.float().mean())
                if args.weight == "upstream":
                    w = torch.tensor([n_pos * y.shape[1], y.shape[1] - n_pos * y.shape[1]],
                                     dtype=torch.float, device="cuda")
                elif args.weight == "inverse":
                    w = torch.tensor([1.0 / max(1 - n_pos, 1e-3), 1.0 / max(n_pos, 1e-3)],
                                     dtype=torch.float, device="cuda")
                    w = w / w.sum() * 2
                else:
                    w = None
                losses.append(F.nll_loss(torch.log_softmax(out, dim=1).reshape(-1, 2),
                                         y.reshape(-1), weight=w))
            loss = torch.stack(losses).mean()
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += float(loss) * len(batch)
        if ep % args.eval_every == 0:
            print("   loss=%.4f" % (total / len(order)))

    print("最终 trainAUC=%.3f testAUC=%.3f" % (evaluate(tr_ids), evaluate(te_ids)))


if __name__ == "__main__":
    main()
