#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在本数据集上重新聚类 anchors，并对比默认 COCO anchors。

======================================================================
为什么需要
======================================================================
`model_data/yolo_anchors.txt` 里的 9 组 anchors 是在 **COCO** 上聚类得到的，
最小一组只有 10×13。而本数据集的猫脸框很大（宽高中位数都在 250 px 左右），
尺度分布与 COCO 完全不同。

后果（实测）：用默认 anchors 时，**所有目标框的最佳 anchor 都落在最大的
三组上**，也就是全部被分配到最粗的 13×13 尺度；26×26 与 52×52 两个尺度
一个正样本都没有，只贡献背景损失：

    level 0 (13×13)  正样本 14
    level 1 (26×26)  正样本  0
    level 2 (52×52)  正样本  0

这既让损失被大量背景项主导，也让两个细尺度完全学不到东西。

本脚本用 keras-yolo3 自带的 `kmeans.py` 在本数据集上重新聚类，
并**用真实的 `preprocess_true_boxes`** 对比两套 anchors 的正样本分配情况，
给出可量化的结论。

输出：
    code/model_data/yolo_anchors_catface.txt    新 anchors（不动原来的文件）
    终端打印对比表

用法：
    python compute_anchors.py
    python compute_anchors.py --k 9 --samples 40
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from config import ANCHORS_PATH, CODE_DIR, INPUT_SHAPE, MODEL_DATA_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ANCHORS_PATH, CODE_DIR, INPUT_SHAPE, MODEL_DATA_DIR

OUT_ANCHORS = MODEL_DATA_DIR / "yolo_anchors_catface.txt"


def load_lines(path: Path):
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def sample_boxes(lines, input_shape, samples):
    """用 get_random_data 取一批真实的 416 空间框（与训练时完全一致的处理）。

    注意：标注文件里写的是**相对 code/ 目录**的路径，而 get_random_data 是按
    当前工作目录解析的，所以要先重建成绝对路径（复用 train.py 里的实现）。
    """
    import numpy as np
    from yolo3.utils import get_random_data
    try:
        from train import rebase_annotation_lines
        lines, _ = rebase_annotation_lines(lines)
    except Exception:
        pass

    box_data = []
    for ln in lines[:samples]:
        _img, box = get_random_data(ln, input_shape, random=False)
        box_data.append(box)
    return np.array(box_data, dtype="float32")


def level_assignment(box_data, anchors, input_shape, num_classes=1):
    """用真实的 preprocess_true_boxes 统计每个尺度的正样本数。"""
    import numpy as np
    from yolo3.model import preprocess_true_boxes

    y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
    counts = []
    for yt in y_true:
        counts.append(int(yt[..., 4].sum()))
    return counts, y_true


def avg_iou(boxes_wh, anchors):
    """框与 anchors 的平均最佳 IoU（与 kmeans.py 的 avg_iou 口径一致）。"""
    import numpy as np
    # 以原点为中心计算 IoU
    inter = np.minimum(boxes_wh[:, None, :], anchors[None, :, :]).prod(axis=2)
    area_b = boxes_wh.prod(axis=1)[:, None]
    area_a = anchors.prod(axis=1)[None, :]
    union = area_a + area_b - inter
    return float((inter / union).max(axis=1).mean())


def main():
    ap = argparse.ArgumentParser(description="在本数据集上重新聚类 anchors")
    ap.add_argument("--k", type=int, default=9, help="anchor 组数（YOLOv3 用 9）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（上游 kmeans 每次结果不同，固定后报告可复现）")
    ap.add_argument("--samples", type=int, default=40, help="参与统计的图片数")
    ap.add_argument("--train", default=str(CODE_DIR / "trainval.txt"),
                    help="标注文件（默认 trainval.txt；没有就用 train.txt）")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    import numpy as np
    from kmeans import YOLO_Kmeans

    input_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1])

    print("=" * 72)
    print("在本数据集上重新聚类 anchors")
    print("=" * 72)

    # ---------- 准备标注文件 ----------
    train_path = Path(args.train)
    if not train_path.is_file():
        train_path = CODE_DIR / "train.txt"
    if not train_path.is_file():
        print("[错误] 找不到 trainval.txt / train.txt，请先运行 voc_annotation.py")
        return 1
    lines = load_lines(train_path)
    print("标注文件 : %s（%d 行）" % (train_path.name, len(lines)))

    # kmeans.py 的 txt2boxes 需要「路径 框 框 ...」格式（空格式），我们的文件正好是。
    #
    # 注意上游有个 bug：YOLO_Kmeans.__init__ 里把 self.filename **硬编码**成了
    # "2012_train.txt"，完全忽略传进去的 filename 参数，所以必须构造后手动覆盖。
    km = YOLO_Kmeans(args.k, str(train_path))
    km.filename = str(train_path)
    boxes_wh = km.txt2boxes()
    print("目标框数 : %d" % len(boxes_wh))
    print("框宽高   : 中位数 %s，范围 %s ~ %s"
          % (np.median(boxes_wh, axis=0).astype(int).tolist(),
             boxes_wh.min(axis=0).tolist(), boxes_wh.max(axis=0).tolist()))
    print("")

    # ---------- 聚类 ----------
    #
    # 上游 kmeans.py 的 kmeans() 里有一句 `np.random.seed()`（**无参**调用），
    # 它会用系统熵重新播种，导致每次运行聚类结果都不一样，报告里没法复现。
    # 这里临时把 np.random.seed 换成空操作，让我们在调用前设的种子真正生效。
    print("正在聚类 k=%d（seed=%d）..." % (args.k, args.seed))
    import numpy as _np
    _np.random.seed(args.seed)
    _orig_seed = _np.random.seed
    _np.random.seed = lambda *a, **k: None
    try:
        kresult = km.kmeans(boxes_wh, k=args.k)
    finally:
        _np.random.seed = _orig_seed
    kresult = kresult[np.lexsort(kresult.T[0, None])]
    new_iou = avg_iou(boxes_wh, kresult)

    coco = np.array([float(x) for x in
                     Path(ANCHORS_PATH).read_text(encoding="utf-8").split(",") if x.strip()])
    coco = coco.reshape(-1, 2)
    old_iou = avg_iou(boxes_wh, coco)

    print("")
    print("-" * 72)
    print("%-28s %s" % ("", "anchors（按宽度升序）"))
    print("-" * 72)
    print("%-28s %s" % ("COCO 默认（当前在用）",
                        "  ".join("%d,%d" % (a[0], a[1]) for a in coco)))
    print("%-28s %s" % ("本数据集聚类（新）",
                        "  ".join("%d,%d" % (a[0], a[1]) for a in kresult)))
    print("")
    print("%-28s %.2f%%" % ("平均最佳 IoU - COCO 默认", old_iou * 100))
    print("%-28s %.2f%%   （提升 %.2f 个百分点）"
          % ("平均最佳 IoU - 新 anchors", new_iou * 100,
             (new_iou - old_iou) * 100))

    # ---------- 真实的正样本分配 ----------
    print("")
    print("-" * 72)
    print("各尺度正样本分配（用真实的 preprocess_true_boxes，%d 张图）" % args.samples)
    print("-" * 72)
    box_data = sample_boxes(lines, input_shape, args.samples)
    old_counts, _ = level_assignment(box_data, coco, input_shape)
    new_counts, _ = level_assignment(box_data, kresult, input_shape)

    names = ["level 0 (13x13, 最大 anchor)", "level 1 (26x26, 中等 anchor)",
             "level 2 (52x52, 最小 anchor)"]
    print("%-30s %12s %12s" % ("尺度", "COCO 默认", "新 anchors"))
    for n, a, b in zip(names, old_counts, new_counts):
        print("%-30s %12d %12d" % (n, a, b))
    print("%-30s %12d %12d" % ("合计", sum(old_counts), sum(new_counts)))

    if new_counts[1] + new_counts[2] > old_counts[1] + old_counts[2]:
        print("")
        print("结论：新 anchors 让更多目标落到细尺度上，原先「细尺度零正样本」的问题得到缓解。")
    else:
        print("")
        print("结论：本数据集目标普遍偏大，即使重新聚类，多数框仍会落在最粗尺度上。")

    # ---------- 写文件 ----------
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    txt = ",".join("%d,%d" % (a[0], a[1]) for a in kresult)
    # keras-yolo3 的 yolo_anchors.txt 是一行、逗号分隔，这里保持一致
    OUT_ANCHORS.write_text(txt + "\n", encoding="utf-8")
    print("")
    print("新 anchors 已写入：%s" % OUT_ANCHORS)
    print("（原来的 %s 未改动）" % ANCHORS_PATH.name)
    print("")
    print("要用新 anchors 训练：")
    print("  python train.py --anchors \"%s\"" % OUT_ANCHORS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
