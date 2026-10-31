# -*- coding: utf-8 -*-
"""实验 04 · 特征提取。

深度特征：torchvision 在 ImageNet 上预训练的 ResNet50，去掉分类层，
取 avgpool 的 2048 维向量（图像检索的通用做法）。

传统基线（用于对比"深度特征是否值得"）：HSV 颜色直方图（8x4x4=128 维）。

产物（outputs/features/）：
  images_resnet50.npy / images_hist.npy        图像库特征 (N,2048) / (N,128)
  videos_resnet50_mean.npy / _max.npy          视频级特征（帧特征平均/最大池化）
  video_frames_resnet50.npy                    全部抽帧特征 (12*16,2048)
  queries_resnet50.npy / queries_hist.npy      查询图特征
  *_meta.json                                  路径与场景标签
  extract_log.json                             各阶段耗时（供效率分析）

用法：python code/extract_features.py
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (BACKBONE, DATA_IMAGES, DATA_QUERIES, DATA_VIDEOS, FEATS_DIR,
                    FEATURE_DIM, FRAMES_PER_VIDEO, HIST_BINS)
from utils import list_images, l2norm, load_meta, query_scene, save_meta, scene_of

LOG = {}


def timed(name):
    def deco(fn):
        def wrap(*a, **kw):
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            LOG[name] = LOG.get(name, 0.0) + time.perf_counter() - t0
            return out
        return wrap
    return deco


# ------------------------------------------------------------------ 深度特征
class DeepExtractor:
    def __init__(self, device=None):
        import torch
        import torchvision
        self.torch = torch
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.fc = torch.nn.Identity()          # 去掉分类层 → avgpool 2048 维
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        from torchvision.transforms import v2 as T
        self.tf = T.Compose([
            T.ToImage(),
            T.Resize(232, antialias=True),
            T.CenterCrop(224),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @timed("deep_forward")
    def batch(self, bgr_list):
        """BGR uint8 图像列表 → (B,2048) float32（RGB、ImageNet 归一化）。"""
        torch = self.torch
        with torch.no_grad():
            x = torch.stack([self.tf(img[:, :, ::-1].copy()) for img in bgr_list])
            f = self.model(x.to(self.device))
            if self.device != "cpu":
                torch.cuda.synchronize()
        return f.cpu().numpy().astype(np.float32)


def hist_feature(bgr):
    """HSV 8x4x4 颜色直方图（对比度归一化），128 维传统基线特征。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256])
    h = h.flatten().astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


@timed("hist_forward")
def hist_batch(bgr_list):
    return np.stack([hist_feature(b) for b in bgr_list]).astype(np.float32)


# ------------------------------------------------------------------ 数据读取
def iter_image_batches(paths, bs=32):
    for i in range(0, len(paths), bs):
        batch = [cv2.imread(p) for p in paths[i:i + bs]]
        batch = [b for b in batch if b is not None]
        if batch:
            yield i, batch


def sample_video_frames(vpath, n_frames):
    """均匀抽 n_frames 帧，返回 (帧列表, 帧号列表)。"""
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    frames, got = [], []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, img = cap.read()
        if ok:
            frames.append(img)
            got.append(int(fi))
    cap.release()
    return frames, got


def extract_gallery(ext, hist_mode=False):
    """图像库特征：返回 (feats, paths, scenes)。"""
    paths = list_images(DATA_IMAGES)
    paths = [p for p in paths if os.path.basename(p) != "images_manifest.csv"]
    feats, keep = [], []
    for i, batch in iter_image_batches(paths):
        f = hist_batch(batch) if hist_mode else ext.batch(batch)
        feats.append(f)
        keep.extend(paths[i:i + len(batch)])
    feats = np.vstack(feats)
    scenes = [scene_of(p) for p in keep]
    return feats, keep, scenes


def extract_queries(ext, hist_mode=False):
    paths = [p for p in list_images(DATA_QUERIES)
             if os.path.basename(p) != "relevance.csv"]
    feats, keep = [], []
    for i, batch in iter_image_batches(paths):
        f = hist_batch(batch) if hist_mode else ext.batch(batch)
        feats.append(f)
        keep.extend(paths[i:i + len(batch)])
    return np.vstack(feats), keep


def extract_videos(ext, hist_mode=False):
    """视频特征：每段均匀抽 FRAMES_PER_VIDEO 帧。

    返回 (video_mean, video_max, frame_feats, video_names, frame_video_idx)。
    """
    vnames = sorted(os.path.splitext(f)[0] for f in os.listdir(DATA_VIDEOS)
                    if f.lower().endswith((".mp4", ".avi", ".mov")))
    all_feats, vid_of_frame = [], []
    for vi, name in enumerate(vnames):
        frames, _ = sample_video_frames(os.path.join(DATA_VIDEOS, f"{name}.mp4"),
                                        FRAMES_PER_VIDEO)
        f = hist_batch(frames) if hist_mode else ext.batch(frames)
        all_feats.append(f)
        vid_of_frame.extend([vi] * len(frames))
        print(f"  [video] {name}: {len(frames)} 帧 → 特征 {f.shape}")
    frames_feats = np.vstack(all_feats)
    vid_of_frame = np.array(vid_of_frame)
    vmean = np.stack([frames_feats[vid_of_frame == vi].mean(0) for vi in range(len(vnames))])
    vmax = np.stack([frames_feats[vid_of_frame == vi].max(0) for vi in range(len(vnames))])
    return vmean, vmax, frames_feats, vnames, vid_of_frame


def main():
    t_all = time.perf_counter()
    meta_common = {"backbone": BACKBONE, "feature_dim": FEATURE_DIM,
                   "hist_bins": list(HIST_BINS)}

    # ---- 深度特征 ----
    print("加载 ResNet50（ImageNet 预训练）……")
    ext = DeepExtractor()
    print(f"  设备：{ext.device}")

    img_feats, img_paths, img_scenes = extract_gallery(ext)
    print(f"图像库：{img_feats.shape[0]} 张 → {img_feats.shape[1]} 维")
    q_feats, q_paths = extract_queries(ext)
    print(f"查询图：{q_feats.shape[0]} 张")
    vmean, vmax, vframes, vnames, vf_idx = extract_videos(ext)
    print(f"视频：{len(vnames)} 段 × {FRAMES_PER_VIDEO} 帧 → 帧 特 征 {vframes.shape}")

    np.save(os.path.join(FEATS_DIR, "images_resnet50.npy"), l2norm(img_feats))
    np.save(os.path.join(FEATS_DIR, "queries_resnet50.npy"), l2norm(q_feats))
    np.save(os.path.join(FEATS_DIR, "videos_resnet50_mean.npy"), l2norm(vmean))
    np.save(os.path.join(FEATS_DIR, "videos_resnet50_max.npy"), l2norm(vmax))
    np.save(os.path.join(FEATS_DIR, "video_frames_resnet50.npy"), l2norm(vframes))
    save_meta(os.path.join(FEATS_DIR, "images_meta.json"), paths=img_paths,
              scenes=img_scenes, **meta_common)
    save_meta(os.path.join(FEATS_DIR, "queries_meta.json"), paths=q_paths,
              scenes=[query_scene(p) for p in q_paths], **meta_common)
    save_meta(os.path.join(FEATS_DIR, "videos_meta.json"), names=vnames,
              frame_video_idx=vf_idx.tolist(), n_frames_per_video=FRAMES_PER_VIDEO,
              **meta_common)

    # ---- 颜色直方图基线 ----
    img_h, img_paths2, img_scenes2 = extract_gallery(None, hist_mode=True)
    q_h, q_paths2 = extract_queries(None, hist_mode=True)
    vh_mean, vh_max, vh_frames, _, _ = extract_videos(None, hist_mode=True)
    np.save(os.path.join(FEATS_DIR, "images_hist.npy"), l2norm(img_h))
    np.save(os.path.join(FEATS_DIR, "queries_hist.npy"), l2norm(q_h))
    np.save(os.path.join(FEATS_DIR, "videos_hist_mean.npy"), l2norm(vh_mean))
    save_meta(os.path.join(FEATS_DIR, "images_hist_meta.json"), paths=img_paths2,
              scenes=img_scenes2, **meta_common)
    save_meta(os.path.join(FEATS_DIR, "queries_hist_meta.json"), paths=q_paths2,
              scenes=[query_scene(p) for p in q_paths2], **meta_common)

    LOG["total"] = time.perf_counter() - t_all
    LOG["device"] = ext.device
    with open(os.path.join(FEATS_DIR, "extract_log.json"), "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=1)
    print("耗时统计：", json.dumps({k: round(v, 2) if isinstance(v, (int, float)) else v
                                    for k, v in LOG.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
