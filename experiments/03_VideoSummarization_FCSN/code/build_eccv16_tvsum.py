# -*- coding: utf-8 -*-
"""构建与 eccv16_dataset_tvsum_google_pool5.h5 等价的本地 h5（离线自建）。

背景：官方 `eccv16_dataset_tvsum_google_pool5.h5` 发布在 Google Drive，本机无法访问，
因此按同样的字段定义在本地离线重建：
  - change_points / n_frame_per_seg：用 KTS（Kernel Temporal Segmentation）在
    GoogLeNet pool5 特征上自动分割得到（与 builddataset.py 的算法一致）；
  - user_summary / gtscore：由 ydata-tvsum50-anno.tsv 的 20 位标注者打分重算；
  - length / fps / picks 等元信息：由原视频直接读取。

字段与 KaiyangZhou/pytorch-vsumm-reinforce 的 eccv16 h5 对齐：
  video_<youtube_id>/
      feature          (320, 1024) float32  pool5 特征（本脚本也一并存下）
      length           int    视频总帧数
      fps              float
      picks            (320,) int  采样帧号
      change_points    (S, 2) int  KTS 分割区间 [start, end)
      n_frame_per_seg  (S,)   int  每段帧数
      user_summary     (20, N) float32  逐帧二值重要性（1 = 该用户认为重要）
      gtscore          (N,)   float32  20 人平均重要性（0~1）
      video_name       str

用法：
  python build_eccv16_tvsum.py --video_dir ... --anno_path ... --info_path ... --h5_path ...
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np


# ----------------------------------------------------------------------------
# KTS: Kernel Temporal Segmentation（与 FCSN 仓库 builddataset.py 同算法）
# ----------------------------------------------------------------------------
def calc_scatters(K):
    """计算所有候选区间的 scatter（段内方差和），用于 KTS 动态规划。"""
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n + 1, n + 1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1)
    diagK2 = np.diag(K2)
    i = np.arange(n).reshape((-1, 1))
    j = np.arange(n).reshape((1, -1))
    scatters = (K1[1:].reshape((1, -1)) - K1[:-1].reshape((-1, 1))
                - (diagK2[1:].reshape((1, -1)) + diagK2[:-1].reshape((-1, 1))
                   - K2[1:, :-1].T - K2[:-1, 1:]) / ((j - i + 1).astype(float) + (j == i - 1).astype(float)))
    scatters[j < i] = 0
    return scatters


def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=False):
    """KTS 核心：把区间 [0, n) 切成 m+1 段的最优动态规划（向量化版本）。

    与原始双重循环实现结果完全一致，但内层对 l 做了向量化，速度约快 5~10 倍。
    """
    n = K.shape[0]
    m = int(ncp)
    J = calc_scatters(K)
    INF = 1e101
    I = INF * np.ones((m + 1, n + 1))
    I[0, lmin:lmax] = J[0, lmin - 1:lmax - 1]
    p = np.zeros((m + 1, n + 1), dtype=int) if backtrack else None

    for k in range(1, m + 1):
        I[k, :] = INF
        for t in range(k * lmin, n):          # 上一段结束位置
            base = I[k - 1, t]
            if base >= 1e99:
                continue
            l_start = max(t + lmin, (k + 1) * lmin)
            l_end = min(n + 1, t + lmax)
            if l_start >= l_end:
                continue
            cand = J[t, l_start - 1:l_end - 1] + base
            cur = I[k, l_start:l_end]
            better = cand < cur
            if better.any():
                cur[better] = cand[better]
                I[k, l_start:l_end] = cur
                if backtrack:
                    pv = p[k, l_start:l_end]
                    pv[better] = t
                    p[k, l_start:l_end] = pv

    cps = np.zeros(m, dtype=int)
    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k - 1] = p[k, cur]
            cur = cps[k - 1]

    scores = I[:, n].copy()
    scores[scores > 1e99] = np.inf
    return cps, scores


def cpd_auto(K, ncp, vmax, desc_rate=1):
    """自动确定分段数：在 1..ncp 个候选里按 scatter + penalty 取最优。"""
    m = ncp
    _, scores = cpd_nonlin(K, m, backtrack=False)
    N = K.shape[0]
    penalties = np.zeros(m + 1)
    ncp_arr = np.arange(1, m + 1)
    penalties[1:] = (ncp_arr / (2.0 * N)) * (np.log(float(N) / ncp_arr) + 1)
    costs = scores / float(N) + penalties
    m_best = int(np.argmin(costs))
    cps, scores2 = cpd_nonlin(K, max(m_best, 1))
    return cps, costs


# ----------------------------------------------------------------------------
# 特征提取
# ----------------------------------------------------------------------------
def build_feature_net(gpu=True):
    import torch
    import torch.nn as nn
    from torchvision import models
    net = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).float()
    net.eval()
    fea_net = nn.Sequential(*list(net.children())[:-2])
    if gpu and torch.cuda.is_available():
        fea_net = fea_net.cuda()
    for p in fea_net.parameters():
        p.requires_grad_(False)
    return fea_net


def mean_normalize(x):
    """与 Kaiyang eccv16 预处理一致：先减均值再按行 L2 归一化。"""
    x = np.array(x)
    x = x - np.mean(x, axis=1, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def read_annotations(anno_path):
    """读取 ydata-tvsum50-anno.tsv：视频id -> (20, 100) 打分矩阵（1~5）。"""
    from collections import OrderedDict
    anno = OrderedDict()
    with open(anno_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            vid, _cat, raw = parts[0], parts[1], parts[2]
            scores = np.array([int(v) for v in raw.split(",")], dtype=np.float32)
            anno.setdefault(vid, []).append(scores)
    return {k: np.stack(v) for k, v in anno.items()}


def upsample_onehot(scores_100, n_frames):
    """把 100 段的打分线性插值到逐帧（对齐 eccv16 的 100 -> N 映射）。"""
    x_old = np.linspace(0, n_frames - 1, len(scores_100))
    x_new = np.arange(n_frames)
    return np.interp(x_new, x_old, scores_100).astype(np.float32)


def process_video(path, fea_net, fps_sample=2.0, target_len=320, gpu=True, batch=32):
    """解码一个视频，返回 320 维采样特征、KTS 分割、帧数、fps。"""
    import cv2
    import torch
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("无法打开视频: %s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ratio = max(n_frames // target_len, 1)
    step20 = max(int(round(fps / fps_sample)), 1)   # 2Hz 采样步长

    feats_slow = []      # 2Hz，用于 KTS
    idx_slow = []
    feats_320 = []       # 320 帧，用于特征存储
    idx_320 = []
    buf, buf_meta = [], []

    def flush():
        if not buf:
            return
        x = torch.stack([transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in buf])
        if gpu and torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            out = fea_net(x).squeeze(-1).squeeze(-1).cpu().numpy()
        for meta, feat in zip(buf_meta, out):
            if meta == "slow":
                feats_slow.append(feat)
            else:
                feats_320.append(feat)
        buf.clear()
        buf_meta.clear()

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        take_slow = (i % step20 == 0)
        take_320 = ((i + 1) % ratio == 0 and len(feats_320) + sum(1 for m in buf_meta if m == "320") < target_len)
        if take_slow or take_320:
            small = cv2.resize(frame, (224, 224))
            if take_slow:
                buf.append(small)
                buf_meta.append("slow")
                idx_slow.append(i)
            if take_320:
                buf.append(small)
                buf_meta.append("320")
                idx_320.append(i)
            if len(buf) >= batch:
                flush()
        i += 1
    flush()
    cap.release()

    feats_slow = np.asarray(feats_slow, dtype=np.float32)
    feats_320 = np.asarray(feats_320, dtype=np.float32)
    return dict(fps=float(fps), n_frames=int(n_frames), ratio=ratio,
                feats_slow=feats_slow, idx_slow=np.asarray(idx_slow, dtype=np.int32),
                feats_320=feats_320, idx_320=np.asarray(idx_320, dtype=np.int32))


def kts_change_points(feats_slow, n_frames, fps):
    """在 2Hz 特征上做 KTS，返回 change_points (S,2) 与 n_frame_per_seg (S,)。"""
    feat = mean_normalize(feats_slow)
    K = np.dot(feat, feat.T)
    n_slow = K.shape[0]
    duration = n_frames / fps
    m = int(np.ceil(duration / 2.0))
    m = max(1, min(m, n_slow - 1))
    cps_slow, _ = cpd_auto(K, m, 1)

    # 慢采样帧号 -> 原视频帧号 -> 段边界
    step = max(int(round(fps / 2.0)), 1)
    cps_frame = np.clip(np.asarray(cps_slow, dtype=int) * step, 0, n_frames - 1)
    change_points = np.concatenate(([0], cps_frame, [n_frames - 1]))
    change_points = np.unique(change_points)

    segs = []
    for idx in range(len(change_points) - 1):
        if idx == len(change_points) - 2:
            segs.append([change_points[idx], change_points[idx + 1]])
        else:
            segs.append([change_points[idx], change_points[idx + 1] - 1])
    segs = np.asarray(segs, dtype=np.int32)
    n_frame_per_seg = (segs[:, 1] - segs[:, 0]).astype(np.int32)
    return segs, n_frame_per_seg


def main():
    ap = argparse.ArgumentParser(description="离线重建 eccv16 格式 TVSum h5")
    ap.add_argument("--video_dir", required=True, help="mp4 所在目录")
    ap.add_argument("--anno_path", required=True, help="ydata-tvsum50-anno.tsv")
    ap.add_argument("--h5_path", required=True, help="输出 h5 路径")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个视频（调试用）")
    ap.add_argument("--gpu", action="store_true", help="用 GPU 提特征")
    ap.add_argument("--fps_sample", type=float, default=2.0, help="KTS 采样帧率")
    args = ap.parse_args()

    anno = read_annotations(args.anno_path)
    videos = sorted(f for f in os.listdir(args.video_dir) if f.lower().endswith(".mp4"))
    if args.limit:
        videos = videos[:args.limit]
    print("待处理视频: %d 个" % len(videos))

    fea_net = build_feature_net(args.gpu)

    with h5py.File(args.h5_path, "w") as f:
        for n, fn in enumerate(videos, 1):
            vid = os.path.splitext(fn)[0]
            if vid not in anno:
                print("  [跳过] %s 没有标注" % vid)
                continue
            t0 = time.time()
            info = process_video(os.path.join(args.video_dir, fn), fea_net,
                                 fps_sample=args.fps_sample, gpu=args.gpu)
            n_frames, fps = info["n_frames"], info["fps"]

            # 320 帧特征：不足则复制最后一帧补齐
            feat = info["feats_320"]
            picks = info["idx_320"]
            if len(feat) == 0:
                print("  [跳过] %s 解不出帧" % vid)
                continue
            if len(feat) < 320:
                pad = 320 - len(feat)
                feat = np.vstack([feat, np.repeat(feat[-1:], pad, axis=0)])
                picks = np.concatenate([picks, picks[-1] + np.arange(1, pad + 1)])
                picks = np.clip(picks, 0, n_frames - 1)

            segs, n_fps = kts_change_points(info["feats_slow"], n_frames, fps)

            scores20 = anno[vid][:20]                       # (20, 100) 1~5
            per_user = np.stack([upsample_onehot(s, n_frames) for s in scores20])
            user_summary = (per_user >= 3.0).astype(np.float32)   # 1~5 中 >=3 视为重要
            gtscore = per_user.mean(axis=0).astype(np.float32) / 5.0

            g = f.create_group("video_" + vid)
            g.create_dataset("feature", data=feat.astype(np.float32))
            g.create_dataset("length", data=np.int32(n_frames))
            g.create_dataset("fps", data=np.float32(fps))
            g.create_dataset("picks", data=picks.astype(np.int32))
            g.create_dataset("change_points", data=segs)
            g.create_dataset("n_frame_per_seg", data=n_fps)
            g.create_dataset("user_summary", data=user_summary)
            g.create_dataset("gtscore", data=gtscore)
            g.attrs["video_name"] = vid
            print("  [%2d/%2d] %s  frames=%d fps=%.2f segs=%d  用时 %.1fs"
                  % (n, len(videos), vid, n_frames, fps, len(segs), time.time() - t0))

    print("已写入:", args.h5_path)


if __name__ == "__main__":
    sys.exit(main())
