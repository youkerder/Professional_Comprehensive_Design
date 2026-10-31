#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一（TensorFlow 编程实验）· AlexNet 图像分类 —— 路径与超参配置。

题目要求（摘自 `docs/extracted_text/timu.txt`）：
    基于 tensorflow 的框架，搭建一个基础的 AlexNet 的网络模型，对图像进行分类。
    具体要求为，编写程序实现输入一张图像，输出图像的类别，使用两种不同的激活函数
    进行比较，ReLU 以及任意另一种，使用两种不同的学习率进行比较。

所有脚本统一从这里取路径与超参，避免硬编码绝对路径。
本模块**只依赖标准库**，不装 TensorFlow 也能导入（便于跑数据检查类脚本）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 目录定位
# --------------------------------------------------------------------------
# 本文件位于 <repo>/experiments/01_TensorFlow_AlexNet/code/config.py
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent                    # experiments/01_TensorFlow_AlexNet
EXPERIMENTS_DIR = EXP_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent

DATASETS_DIR = REPO_ROOT / "datasets"
DATA_ROOT = DATASETS_DIR / "classification"  # 分类数据集根目录（本实验新建）
CIFAR_DIR = DATA_ROOT / "cifar10"            # CIFAR-10 下载与缓存位置

MODELS_DIR = EXP_DIR / "models"              # 训练得到的权重
LOGS_DIR = EXP_DIR / "logs"                  # TensorBoard events 与 checkpoint
OUTPUTS_DIR = EXP_DIR / "outputs"            # 推理 / 评测 / 对比结果
PREDICT_DIR = OUTPUTS_DIR / "predict"
EVAL_DIR = OUTPUTS_DIR / "eval"
COMPARE_DIR = OUTPUTS_DIR / "compare"

REPORTS_DIR = REPO_ROOT / "reports"
REPORT_FIG_DIR = REPORTS_DIR / "figures"     # 与实验一共享图目录，文件名用 01b_ 前缀区分
REPORT_TAB_DIR = REPORTS_DIR / "tables"

# 实验一（目标检测）的图片，用于把两部分串起来的演示
CAT_FACE_JPEG_DIR = DATASETS_DIR / "cat_face" / "JPEGImages"

# --------------------------------------------------------------------------
# 数据集：CIFAR-10
# --------------------------------------------------------------------------
# 官方 python 版压缩包（163 MB 左右）。file_hash 与 Keras 自带的 cifar10.py 一致，
# 用于校验下载完整性——这一步很重要，断网重连后拿到半个包会导致解析出奇怪的形状。
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_DIRNAME = "cifar-10-batches-py"
CIFAR_HASH = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"

NUM_CLASSES = 10
IMAGE_SHAPE = (32, 32, 3)

# CIFAR-10 的 10 个类别。中文名用于 predict.py 的输出，顺序必须与数据集一致。
CLASS_NAMES = [
    "airplane",     # 0
    "automobile",   # 1
    "bird",         # 2
    "cat",          # 3
    "deer",         # 4
    "dog",          # 5
    "frog",         # 6
    "horse",        # 7
    "ship",         # 8
    "truck",        # 9
]
CLASS_NAMES_ZH = {
    "airplane": "飞机",
    "automobile": "汽车",
    "bird": "鸟",
    "cat": "猫",
    "deer": "鹿",
    "dog": "狗",
    "frog": "青蛙",
    "horse": "马",
    "ship": "船",
    "truck": "卡车",
}

# 从 50000 张训练集里划出多少作验证集。
# 测试集 10000 张**始终**不参与训练与选型，只在最终评测时用一次。
VAL_SIZE = 5000

# --------------------------------------------------------------------------
# 模型：基础 AlexNet
# --------------------------------------------------------------------------
# 通道数按 CIFAR-10 的 32x32 输入缩放。
# 实测（i7-13700H，TF 2.10，CPU，batch 128）：
#   标准通道 96/256/384/384/256 -> 7.6 ms/图，一个 epoch（5 万张）约 6.3 分钟
#   轻量通道 64/192/384/256/256 -> 4.2 ms/图，一个 epoch 约 3.5 分钟   <- 采用
CONV_CHANNELS = (64, 192, 384, 256, 256)
FC_UNITS = (512, 512)
DROPOUT = 0.5
LEAKY_ALPHA = 0.1          # LeakyReLU 的负区间斜率（Keras 默认值）

# 归一化方式。原始 AlexNet 用 LRN，这里默认用 BN：
# BN 能让模型在 epoch 数有限时更快收敛，对本次"15 epoch 内看出差异"的目标更有利。
# 可用 --norm lrn 回到更接近原始论文的做法。
DEFAULT_NORM = "bn"

# --------------------------------------------------------------------------
# 激活函数与学习率：本次要对比的两个自变量
# --------------------------------------------------------------------------
# 题目要求"ReLU 以及任意另一种"，这里选 LeakyReLU（只在负区间斜率不同，对比最干净）。
ACTIVATIONS = ("relu", "leaky_relu")
ACT_LABELS = {"relu": "ReLU", "leaky_relu": "LeakyReLU"}

# 题目要求两种学习率。用 SGD + momentum + weight decay，与原始 AlexNet 一致。
LEARNING_RATES = (0.01, 0.001)

# --------------------------------------------------------------------------
# 训练默认超参
# --------------------------------------------------------------------------
DEFAULT_EPOCHS = 15         # 4 组共约 3.5 小时（CPU 实测）
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 42
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4         # Keras 里用 SGD(decay=...) 实现 L2 正则

# 数据增强：CIFAR-10 的标准做法（随机水平翻转 + 4 像素零填充随机裁剪）
PAD_SIZE = 4

# 每 N 个 epoch 无条件保存一次权重。
# 理由同实验一：save_best_only 依赖验证指标，可能长期不落盘，一旦中断就白练。
KEEP_EVERY = 5

# 四组对比的 run_name 规则
RUN_NAME_FMT = "{act}-lr{lr:g}"


def run_name(act: str, lr: float) -> str:
    """由激活函数与学习率生成 run_name，例如 `relu-lr0.01`。"""
    return RUN_NAME_FMT.format(act=act.replace("_", ""), lr=lr)


def all_runs():
    """返回四组对比配置：[(run_name, act, lr), ...]。"""
    return [(run_name(a, r), a, r) for a in ACTIVATIONS for r in LEARNING_RATES]


# --------------------------------------------------------------------------
# 画图公共设置
# --------------------------------------------------------------------------
# matplotlib 默认字体不含中文字形，直接画中文会变成一个个"方框"，
# 而且只在 stderr 打一句 `Glyph xxxxx missing from current font` 警告——
# 不仔细看根本注意不到，很容易把带方框的图交出去（本实验就踩过一次：
# 混淆矩阵图的标题、坐标轴、色标全是方框）。
# 所以统一放在这里，所有画图脚本都调 setup_matplotlib()。
CJK_FONTS = ["Microsoft YaHei", "SimHei", "SimSun", "DengXian",
             "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]


def setup_matplotlib():
    """配置 matplotlib（无界面后端 + 中文字体），返回 pyplot 模块。

    画图的脚本都应调用它，而不是各自 `import matplotlib.pyplot as plt`，
    否则很容易漏掉中文字体设置。
    """
    import matplotlib
    matplotlib.use("Agg")                # 后台运行不需要 GUI
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = CJK_FONTS
    plt.rcParams["axes.unicode_minus"] = False   # 负号用 ASCII，避免变成方框
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


def describe():
    """打印当前配置，便于排查路径与超参问题。"""
    print("仓库根目录     : %s" % REPO_ROOT)
    print("数据集目录     : %s  %s" % (DATA_ROOT, "存在" if DATA_ROOT.is_dir() else "!! 不存在"))
    print("  CIFAR-10     : %s  %s" % (CIFAR_DIR, "存在" if CIFAR_DIR.is_dir() else "尚未下载"))
    print("实验目录       : %s" % EXP_DIR)
    print("类别数         : %d  %s" % (NUM_CLASSES, ", ".join(CLASS_NAMES)))
    print("卷积通道       : %s" % (CONV_CHANNELS,))
    print("全连接         : %s" % (FC_UNITS,))
    print("对比激活函数   : %s" % ", ".join(ACT_LABELS[a] for a in ACTIVATIONS))
    print("对比学习率     : %s" % ", ".join("%g" % r for r in LEARNING_RATES))
    print("默认 epochs    : %d   batch %d   seed %d"
          % (DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_SEED))
    print("")
    print("四组对比：")
    for name, act, lr in all_runs():
        print("  %-22s %-11s lr=%g" % (name, ACT_LABELS[act], lr))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    describe()
