# -*- coding: utf-8 -*-
"""开发期自检：KTS 向量化实现与仓库原始实现等价性 + GoogLeNet 特征维度。"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_eccv16_tvsum import cpd_nonlin, calc_scatters


def cpd_nonlin_ref(K, ncp, lmin=1, lmax=100000, backtrack=True):
    """builddataset.py 原始实现（逐元素循环），用于对拍。"""
    m = int(ncp)
    n, _ = K.shape
    J = calc_scatters(K)
    I = 1e101 * np.ones((m + 1, n + 1))
    I[0, lmin:lmax] = J[0, lmin - 1:lmax - 1]
    p = np.zeros((m + 1, n + 1), dtype=int)
    for k in range(1, m + 1):
        for l in range((k + 1) * lmin, n + 1):
            tmin = max(k * lmin, l - lmax)
            tmax = l - lmin + 1
            c = J[tmin:tmax, l - 1].reshape(-1) + I[k - 1, tmin:tmax].reshape(-1)
            I[k, l] = np.min(c)
            p[k, l] = np.argmin(c) + tmin
    cps = np.zeros(m, dtype=int)
    cur = n
    for k in range(m, 0, -1):
        cps[k - 1] = p[k, cur]
        cur = cps[k - 1]
    return cps, I[:, n].copy()


def main():
    rng = np.random.RandomState(0)
    ok_all = True
    for (n, m) in [(40, 5), (60, 8), (120, 10)]:
        X = rng.randn(n, 16)
        K = X @ X.T
        a, b = cpd_nonlin(K, m, backtrack=True)
        ra, rb = cpd_nonlin_ref(K, m, backtrack=True)
        same = np.array_equal(a, ra) and np.allclose(b, rb)
        print("n=%d m=%d  向量化=%s  原始=%s  一致=%s" % (n, m, a, ra, same))
        ok_all = ok_all and same

    import torch
    from build_eccv16_tvsum import build_feature_net
    net = build_feature_net(gpu=torch.cuda.is_available())
    x = torch.randn(2, 3, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        y = net(x)
    print("GoogLeNet 特征输出:", tuple(y.shape))
    print("总判定:", "PASS" if ok_all and tuple(y.shape) == (2, 1024, 1, 1) else "FAIL")


if __name__ == "__main__":
    main()
