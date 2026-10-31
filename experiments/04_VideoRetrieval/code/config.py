# -*- coding: utf-8 -*-
"""实验 04 · 视频检索：路径与参数配置。

数据目录沿用仓库约定：
  datasets/retrieval/images/   以图搜图底库（按场景类别分子目录）
  datasets/retrieval/videos/   以图搜视频底库（12 段合成短视频）
  datasets/retrieval/queries/  查询图片（从视频场景裁剪/增强得到）
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 仓库根目录由脚本位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))

DATA_IMAGES = os.path.join(REPO, "datasets", "retrieval", "images")
DATA_VIDEOS = os.path.join(REPO, "datasets", "retrieval", "videos")
DATA_QUERIES = os.path.join(REPO, "datasets", "retrieval", "queries")

FEATS_DIR = os.path.join(EXP, "outputs", "features")
INDEX_DIR = os.path.join(EXP, "outputs", "index")
RESULTS_DIR = os.path.join(EXP, "outputs", "results")
METRICS_DIR = os.path.join(EXP, "outputs", "metrics")
LOGS_DIR = os.path.join(EXP, "logs")
FIGS_DIR = os.path.join(REPO, "reports", "figures")

for _d in (FEATS_DIR, INDEX_DIR, RESULTS_DIR, METRICS_DIR, LOGS_DIR, FIGS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- 数据集参数 -----------------------------------------------------------
VIDEO_W, VIDEO_H = 640, 360
VIDEO_FPS = 24
VIDEO_SECONDS = 8                 # 每段 8 s → 192 帧
FRAMES_PER_VIDEO_IN_LIB = 8       # 每段视频向图像库贡献的关键帧数
QUERIES_PER_SCENE = 2             # 每个场景 2 张查询图
N_DISTRACTOR_SCENES = 6           # 只进图像库、没有对应视频的干扰场景
IMGS_PER_DISTRACTOR = 4

# ---- 特征提取参数 ---------------------------------------------------------
BACKBONE = "resnet50"             # torchvision ImageNet 预训练
FEATURE_DIM = 2048                # ResNet50 avgpool 输出
FRAMES_PER_VIDEO = 16             # 以图搜视频：每段视频均匀抽帧数
HIST_BINS = (8, 4, 4)             # HSV 颜色直方图基线（H/S/V）

# ---- 检索与评价参数 -------------------------------------------------------
TOPK_I2I = 10                     # 以图搜图返回 Top-10
TOPK_I2V = 3                      # 以图搜视频返回 Top-3（库只有 12 段视频）
P_AT_K_LIST = (1, 3, 5, 10)

# ---- 效率对比参数 ---------------------------------------------------------
BENCH_QUERIES = 100               # 小库上的查询重复次数
BENCH_BIG_N = 20000               # 规模实验：把库放大到 2 万向量
BENCH_BIG_QUERIES = 20
BENCH_SEED = 2026
