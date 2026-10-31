# -*- coding: utf-8 -*-
"""诊断：训练过程中 BatchNorm 的 running stats 是否让 eval 模式失去预测能力。

做法：训练 N 个 epoch，然后分别在 train 模式（BN 用 batch 统计）和 eval 模式
（BN 用 running 统计）下算训练集/测试集 AUC。若 train 模式高、eval 模式 0.5，
说明是 BN 统计量问题，而不是特征/标签问题。
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

F1 = os.path.join(BASE, "datasets", "tvsum50", "fcsn_tvsum.h5")
IDS = os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN", "logs", "test_ids.txt")


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    return (rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batchnorm", default="momentum", choices=["momentum", "instance", "none"])
    args = ap.parse_args()

    test_ids = [l.strip().replace("video_", "") for l in open(IDS, encoding="utf-8") if l.strip()]
    with h5py.File(F1, "r") as f:
        items = {k[6:]: dict(fe=f[k]["feature"][()].astype(np.float32),
                             la=f[k]["label"][()].astype(np.int64)) for k in sorted(f.keys())}
    tr = [k for k in items if k not in test_ids]
    te = [k for k in items if k in test_ids]
    Xall = np.stack([items[k]["fe"] for k in tr])
    mu = Xall.mean(axis=(0, 1))
    sd = Xall.std(axis=(0, 1)) + 1e-6

    def prep(k):
        x = torch.tensor(items[k]["fe"]).cuda()
        return ((x - torch.tensor(mu).cuda()) / torch.tensor(sd).cuda()).t().unsqueeze(0)

    feats = {k: prep(k) for k in items}
    labels = {k: torch.tensor(items[k]["la"]).cuda() for k in items}

    model = FCSN(2).cuda()
    if args.batchnorm == "instance":
        def rep(m):
            for name, child in m.named_children():
                if isinstance(child, torch.nn.BatchNorm1d):
                    setattr(m, name, torch.nn.InstanceNorm1d(child.num_features, affine=True))
                else:
                    rep(child)
        rep(model)
    elif args.batchnorm == "none":
        def rep(m):
            for name, child in m.named_children():
                if isinstance(child, torch.nn.BatchNorm1d):
                    setattr(m, name, torch.nn.Identity())
                else:
                    rep(child)
        rep(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.RandomState(0)

    def evaluate(ids, mode):
        if mode == "train":
            model.train()
        else:
            model.eval()
        scores, gts = [], []
        with torch.no_grad():
            for k in ids:
                p = torch.softmax(model(feats[k]), dim=1)[0, 1].cpu().numpy()
                scores.append(p)
                gts.append(items[k]["la"])
        return auc(np.concatenate(scores), np.concatenate(gts))

    print("batchnorm=%s lr=%g epochs=%d" % (args.batchnorm, args.lr, args.epochs))
    print("epoch  tr@batch  tr@eval  te@batch  te@eval  loss")
    for ep in range(args.epochs + 1):
        if ep % 10 == 0:
            print("  %3d    %.3f     %.3f    %.3f     %.3f" %
                  (ep, evaluate(tr, "train"), evaluate(tr, "eval"),
                   evaluate(te, "train"), evaluate(te, "eval")), end="")
            print("    -" if ep == 0 else "    %.4f" % last_loss)
        if ep == args.epochs:
            break
        order = rng.permutation(tr)
        tot = 0.0
        for s in range(0, len(order), 5):
            batch = order[s:s + 5]
            opt.zero_grad()
            ls = []
            for k in batch:
                out = model(feats[k])
                ls.append(F.nll_loss(torch.log_softmax(out, dim=1).reshape(-1, 2),
                                     labels[k].unsqueeze(0).reshape(-1)))
            loss = torch.stack(ls).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(batch)
        last_loss = tot / len(order)


if __name__ == "__main__":
    main()
