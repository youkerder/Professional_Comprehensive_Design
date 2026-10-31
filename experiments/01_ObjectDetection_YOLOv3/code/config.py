#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 01 路径与类别配置。

所有脚本统一从这里取路径，避免在代码里硬编码绝对路径
（指导书里的示例代码就有 `/Users/youxinlin/Desktop/...` 这类硬编码）。
"""

import os
import sys

from pathlib import Path

# --------------------------------------------------------------------------
# 兼容层：让 keras-yolo3 能在较新的 Keras 上导入。
# 放在最前面，保证任何脚本 import config 时补丁已生效。
# 用 try/except 包住，这样不装 Keras 的纯标准库脚本（make_imagesets、
# voc_annotation、evaluate_detection 等）依然能正常使用本模块。
# --------------------------------------------------------------------------
try:
    import keras_compat  # noqa: F401  （导入即生效）
except Exception:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import keras_compat  # noqa: F401
    except Exception:
        pass

# --------------------------------------------------------------------------
# 目录定位
# --------------------------------------------------------------------------
# 本文件位于 <repo>/experiments/01_ObjectDetection_YOLOv3/code/config.py
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent          # experiments/01_ObjectDetection_YOLOv3
EXPERIMENTS_DIR = EXP_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent

# 数据集（仓库内的共享数据集目录）
DATASETS_DIR = REPO_ROOT / "datasets"
VOC_ROOT = DATASETS_DIR / "cat_face"      # VOC2007 风格的数据集根目录
JPEG_DIR = VOC_ROOT / "JPEGImages"        # 原图
ANNO_DIR = VOC_ROOT / "Annotations"       # LabelImg 生成的 .xml
IMAGESETS_DIR = VOC_ROOT / "ImageSets" / "Main"   # trainval/train/val/test 索引
TEST_IMAGES_DIR = VOC_ROOT / "test"       # 检测演示用的 8 张图（指导书 detect_img 的输入）

# 实验 01 自己的目录
MODEL_DATA_DIR = CODE_DIR / "model_data"  # yolo_anchors.txt / voc_classes.txt / yolov3.cfg
MODELS_DIR = EXP_DIR / "models"           # 训练得到的权重
OUTPUTS_DIR = EXP_DIR / "outputs"         # 检测结果图 / 视频
LOGS_DIR = EXP_DIR / "logs"               # TensorBoard events 与 checkpoint

# --------------------------------------------------------------------------
# 类别
# --------------------------------------------------------------------------
# 本实验只做猫脸检测，只有一个类别。标签名必须与 LabelImg 里填的完全一致。
CLASSES = ["cat"]

# YOLOv3 相关推导量
NUM_CLASSES = len(CLASSES)
# 每个 yolo 层的 filters = 3 * (5 + len(classes))，指导书 1.4.1 节
FILTERS = 3 * (5 + NUM_CLASSES)

# 训练超参（与指导书一致）
INPUT_SHAPE = (416, 416)
ANCHORS_PATH = MODEL_DATA_DIR / "yolo_anchors.txt"
CLASSES_PATH = MODEL_DATA_DIR / "voc_classes.txt"

# ImageSets 划分比例（修正了指导书 test.py 里写反的两个值，详见 make_imagesets.py）
TRAINVAL_PERCENT = 0.9   # trainval 占全部有标注图片的比例，其余为 test
TRAIN_PERCENT = 0.9      # train 占 trainval 的比例，其余为 val
SPLIT_SEED = 42          # 固定种子，保证可复现

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def ensure_dirs():
    """建好本实验需要的输出目录。"""
    for d in (MODEL_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe():
    """打印当前路径配置，便于排查“文件找不到”的问题。"""
    print("仓库根目录     : %s" % REPO_ROOT)
    print("数据集根目录   : %s  %s" % (VOC_ROOT, "存在" if VOC_ROOT.is_dir() else "!! 不存在"))
    print("  原图         : %s" % JPEG_DIR)
    print("  标注         : %s" % ANNO_DIR)
    print("  索引         : %s" % IMAGESETS_DIR)
    print("  检测演示图   : %s" % TEST_IMAGES_DIR)
    print("实验目录       : %s" % EXP_DIR)
    print("类别           : %s  (num_classes=%d, filters=%d)"
          % (CLASSES, NUM_CLASSES, FILTERS))


if __name__ == "__main__":
    # Windows 控制台默认 GBK，统一改成 UTF-8 免得中文乱码
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    describe()
    _ = os
