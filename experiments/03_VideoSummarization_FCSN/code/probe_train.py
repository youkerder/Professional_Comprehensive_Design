# -*- coding: utf-8 -*-
"""训练配方探针：用若干组超参快速训练 FCSN，报训练/测试 AUC 与 F-score。

目的：train.py 默认配方（上游原版：Adam lr=1e-3 + NLLLoss 类别权重 [n_pos, n_neg]）
在自建 TVSum 特征上 F-score 约 0.21、AUC 约 0.49（等价随机），需要定位是
「特征/标签」问题还是「优化」问题。ridge 线性基线在同一份数据上测试 AUC=0.652，
说明数据本身有信号，因此这里只改训练侧配方。

用法：
  python probe_train.py --epochs 40
"""
import argparse
import itertools
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import eval as eval_utils
from fcsn import FCSN
import os

# 仓库根目录（由本文件位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根）
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATA = os.path.join(BASE, "datasets", "tvsum50", "fcsn_tvsum.h5")
IDS = os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN", "logs", "test_ids.txt")


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    return (rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def load():
    test_ids = [l.strip().replace("video_", "") for l in open(IDS, encoding="utf-8") if l.strip()]
    with h5py.File(DATA, "r") as f:
        items = {}
        for k in sorted(f.keys()):
            g = f[k]
            items[k[6:]] = dict(feature=g["feature"][()].astype(np.float32),
                                label=g["label"][()].astype(np.int64))
    tr = [v for k, v in items.items() if k not in test_ids]
    te = [v for k, v in items.items() if k in test_ids]
    return tr, te, items, test_ids


@torch.no_grad()
def predict(model, feat, mu=None, sd=None):
    x = torch.tensor(feat, dtype=torch.float32)
    if mu is not None:
        x = (x - mu) / sd
    model.eval()
    p = torch.softmax(model(x.t().unsqueeze(0).cuda()).squeeze(0), dim=0)[1].cpu().numpy()
    model.train()
    return p


def fscore(model, items, ids, mu=None, sd=None):
    scores, labels = [], []
    per = []
    for vid in ids:
        it = items[vid]
        p = predict(model, it["feature"], mu, sd)
        scores.append(p)
        labels.append(it["label"])
        # 用 eval.select_keyshots 的 15% 预算做选择，需要 cps/length，这里用整段均匀近似
        per.append(p)
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    return auc(s, y), s, y


def run(cfg, tr, te, items, test_ids, mu, sd):
    torch.manual_seed(0)
    model = FCSN(2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for ep in range(cfg["epochs"]):
        for it in tr:
            x = torch.tensor(it["feature"], dtype=torch.float32)
            if cfg["std"]:
                x = (x - mu) / sd
            x = x.t().unsqueeze(0).cuda()
            y = torch.tensor(it["label"], dtype=torch.long).unsqueeze(0).cuda()
            logits = model(x)
            logp = torch.log_softmax(logits, dim=1)
            if cfg["weight"] == "upstream":
                n_pos = float(y.sum()) / y.shape[0]
                n_neg = y.shape[1] - n_pos
                w = torch.tensor([n_pos, n_neg], dtype=torch.float, device="cuda")
            elif cfg["weight"] == "inverse":
                n_pos = float(y.float().mean())
                w = torch.tensor([1.0 / max(1 - n_pos, 1e-3), 1.0 / max(n_pos, 1e-3)],
                                 dtype=torch.float, device="cuda")
                w = w / w.sum() * 2
            else:
                w = None
            loss = F.nll_loss(logp.reshape(-1, 2), y.reshape(-1), weight=w)
            opt.zero_grad()
            loss.backward()
            if cfg["clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()

    tr_auc, _, _ = fscore(model, items, [k for k in items if k not in test_ids], mu, sd)
    te_auc, _, _ = fscore(model, items, test_ids, mu, sd)
    tag = "lr=%g std=%d w=%s clip=%s ep=%d" % (cfg["lr"], cfg["std"], cfg["weight"],
                                              cfg["clip"], cfg["epochs"])
    print("%-52s trainAUC=%.3f testAUC=%.3f" % (tag, tr_auc, te_auc))
    return dict(cfg=tag, train_auc=tr_auc, test_auc=te_auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--grid", default="quick")
    args = ap.parse_args()

    tr, te, items, test_ids = load()
    allx = np.stack([it["feature"] for it in tr])
    mu = allx.mean(axis=(0, 1))
    sd = allx.std(axis=(0, 1)) + 1e-6
    print("训练视频 %d 测试视频 %d" % (len(tr), len(te)))

    grid = {
        "quick": [
            dict(lr=1e-3, std=False, weight="upstream", clip=0, epochs=args.epochs),
            dict(lr=1e-3, std=True, weight="upstream", clip=0, epochs=args.epochs),
            dict(lr=1e-4, std=True, weight="none", clip=5.0, epochs=args.epochs),
            dict(lr=1e-4, std=True, weight="inverse", clip=5.0, epochs=args.epochs),
        ],
    }[args.grid]

    res = []
    for cfg in grid:
        res.append(run(cfg, tr, te, items, test_ids, mu, sd))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
