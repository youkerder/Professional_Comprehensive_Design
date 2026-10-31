# -*- coding: utf-8 -*-
"""最小化诊断：确认 FCSN 的优化器是否真的在更新参数。

只做四件事：
1. 取一段视频的 320 帧特征；
2. 跑 200 步 Adam + NLLLoss（无类别权重）；
3. 每 20 步打印 loss、预测 std、AUC，以及 conv8.bias 的变化量；
4. 打印前向/反向是否出现 NaN。
"""
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from fcsn import FCSN
import os

# 仓库根目录（由本文件位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根）
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

F1 = os.path.join(BASE, "datasets", "tvsum50", "fcsn_tvsum.h5")


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    return (rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)


def main():
    with h5py.File(F1, "r") as f:
        vid = sorted(f.keys())[0]
        fe = f[vid]["feature"][()].astype(np.float32)
        la = f[vid]["label"][()].astype(np.int64)
    X = torch.tensor(fe).cuda()
    X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).t().unsqueeze(0)
    Y = torch.tensor(la).cuda()

    model = FCSN(2).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    b0 = model.conv8[0].bias.detach().clone() if isinstance(model.conv8[0], torch.nn.Conv1d) else None
    w0 = model.conv8[0].weight.detach().clone() if b0 is not None else None
    if b0 is None:                      # conv8 可能是 Sequential，取里面第一个 Conv1d
        for m in model.modules():
            if isinstance(m, torch.nn.Conv1d) and m.out_channels == 2:
                b0, w0 = m.bias.detach().clone(), m.weight.detach().clone()
                break

    print("特征: %s shape %s | 正样本占比 %.3f" % (vid, tuple(fe.shape), la.mean()))
    for step in range(201):
        model.train()
        out = model(X)
        loss = F.nll_loss(torch.log_softmax(out, dim=1).reshape(-1, 2), Y)
        opt.zero_grad()
        loss.backward()
        gn = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
        opt.step()
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                p = torch.softmax(model(X), dim=1)[0, 1].cpu().numpy()
            cur_b = None
            for m in model.modules():
                if isinstance(m, torch.nn.Conv1d) and m.out_channels == 2:
                    cur_b = m.bias.detach()
                    cur_w = m.weight.detach()
                    break
            print("step %3d loss=%8.5f grad_norm=%9.3f p.std=%.4f AUC=%.3f |Δbias|=%.3e |Δw|=%.3e"
                  % (step, loss.item(), gn, p.std(), auc(p, la),
                     float((cur_b - b0).abs().max()), float((cur_w - w0).abs().max())))


if __name__ == "__main__":
    main()
