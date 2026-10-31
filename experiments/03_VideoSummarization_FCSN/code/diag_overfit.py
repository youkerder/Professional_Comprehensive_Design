# -*- coding: utf-8 -*-
"""诊断：FCSN 能否在「单个视频」上过拟合（容量/梯度下限测试）。

如果连一段 320 帧的视频都拟合不到 AUC≈1，说明问题在网络里（BN/初始化/梯度），
与训练集大小无关。分别测试：
  norm=none       把 BatchNorm1d 换成 Identity
  norm=instance   换成 InstanceNorm1d(affine=True)
  norm=batch      保持 BatchNorm1d
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


def replace_norm(model, kind):
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.BatchNorm1d):
            if kind == "none":
                new = torch.nn.Identity()
            elif kind == "instance":
                new = torch.nn.InstanceNorm1d(child.num_features, affine=True)
            else:
                continue
            setattr(model, name, new)
        else:
            replace_norm(child, kind)
    return model


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    return (rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="none", choices=["none", "instance", "batch"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--vid", default=None)
    args = ap.parse_args()

    with h5py.File(F1, "r") as f:
        ks = sorted(f.keys())
        vid = args.vid or ks[0]
        feat = f[vid]["feature"][()].astype(np.float32)
        lab = f[vid]["label"][()].astype(np.int64)

    X = torch.tensor(feat).cuda()
    X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).t().unsqueeze(0)
    Y = torch.tensor(lab).cuda()
    model = replace_norm(FCSN(2).cuda(), args.norm)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def probs():
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(X), dim=1)[0, 1].cpu().numpy()

    print("vid=%s norm=%s lr=%g steps=%d 正样本占比=%.3f" % (vid, args.norm, args.lr, args.steps, lab.mean()))
    for step in range(args.steps + 1):
        if step % 250 == 0:
            p = probs()
            loss = F.nll_loss(torch.log_softmax(model(X), dim=1).reshape(-1, 2), Y).item()
            print("  step %4d loss=%.4f AUC=%.3f p.std=%.3f" % (step, loss, auc(p, lab), p.std()))
        model.train()
        out = model(X)
        loss = F.nll_loss(torch.log_softmax(out, dim=1).reshape(-1, 2), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
