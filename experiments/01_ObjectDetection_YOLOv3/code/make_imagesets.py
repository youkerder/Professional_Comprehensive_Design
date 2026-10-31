#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 VOC 数据集 ImageSets/Main 下的 trainval / train / val / test 索引文件。

本脚本替代《目标检测实验指导书》1.3 节的 `test.py`。

======================================================================
与指导书 test.py 的区别（重要）
======================================================================
指导书原文：

    trainval_percent = 0.1
    train_percent = 0.9
    ...
    tv = int(num * trainval_percent)      # 10%
    tr = int(tv * train_percent)          # 9%
    trainval = random.sample(list, tv)    # 只抽 10% 作为 trainval
    train = random.sample(trainval, tr)
    ...
    for i in list:
        if i in trainval:
            ftrainval.write(name)
            if i in train: ftest.write(name)   # 注意：写进了 test.txt
            else:          fval.write(name)
        else:              ftrain.write(name)

这样得到的划分是：train 90%、test 9%、val 1%、trainval 10%。
问题有二：
  1. **trainval_percent 写反了**。VOC 惯例是 trainval 占 90%、test 占 10%，
     而这里 trainval 只占 10%，test 反而成了大头。
  2. 命名的语义混乱：`train` 的补集被写进了 `test.txt`，
     而 `trainval` 内部又被切成 `test` 和 `val`。

本脚本按 VOC 惯例实现，语义清晰：

    test     = 1 - trainval_percent           默认 10%
    train    = train_percent * trainval       默认 81%
    val      = (1 - train_percent) * trainval 默认  9%

**注意**：`train.txt` 是网络训练用的划分，`test.txt` 只是留出的测试划分。
指导书 2.2 节用于检测演示的 8 张图在 `JPEGImages` 之外单独存放
（`<voc_root>/test/`），与这里的 `test.txt` 不是一回事。

其它改进：
  * 只收录**有标注**的图片（索引里的图片没标注会导致训练时读不到框）；
  * 检查图片与标注的配对情况，缺标注 / 多余标注都会告警；
  * 固定随机种子，结果可复现；
  * 图片数量太少时给出明确告警，而不是静默生成空文件。

用法：
    python make_imagesets.py
    python make_imagesets.py --voc-root ../../datasets/cat_face
    python make_imagesets.py --trainval-percent 0.8 --train-percent 0.75 --seed 0
    python make_imagesets.py --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

try:
    from config import (
        IMAGE_EXTS, IMAGESETS_DIR, SPLIT_SEED,
        TRAIN_PERCENT, TRAINVAL_PERCENT, VOC_ROOT,
    )
except ImportError:  # 允许作为独立脚本从任意目录运行
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        IMAGE_EXTS, IMAGESETS_DIR, SPLIT_SEED,
        TRAIN_PERCENT, TRAINVAL_PERCENT, VOC_ROOT,
    )


def collect(voc_root: Path):
    """返回 (图片 stem->路径, 标注 stem->路径)。"""
    jpeg_dir = voc_root / "JPEGImages"
    anno_dir = voc_root / "Annotations"

    if not jpeg_dir.is_dir():
        raise SystemExit("找不到图片目录：%s" % jpeg_dir)
    if not anno_dir.is_dir():
        raise SystemExit("找不到标注目录：%s" % anno_dir)

    images = {}
    for p in sorted(jpeg_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images[p.stem] = p

    annos = {}
    for p in sorted(anno_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".xml":
            annos[p.stem] = p

    return images, annos


def main():
    ap = argparse.ArgumentParser(
        description="生成 VOC ImageSets/Main 下的划分索引文件（替代指导书的 test.py）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--voc-root", default=str(VOC_ROOT),
                    help="VOC 数据集根目录（默认：datasets/cat_face）")
    ap.add_argument("--out-dir", default=str(IMAGESETS_DIR),
                    help="索引输出目录（默认：<voc-root>/ImageSets/Main）")
    ap.add_argument("--trainval-percent", type=float, default=TRAINVAL_PERCENT,
                    help="trainval 占全部有标注图片的比例，其余为 test（默认 %.2f）"
                         % TRAINVAL_PERCENT)
    ap.add_argument("--train-percent", type=float, default=TRAIN_PERCENT,
                    help="train 占 trainval 的比例，其余为 val（默认 %.2f）" % TRAIN_PERCENT)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED,
                    help="随机种子，固定后结果可复现（默认 %d）" % SPLIT_SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印划分结果，不写文件")
    args = ap.parse_args()

    voc_root = Path(args.voc_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not 0.0 < args.trainval_percent <= 1.0:
        raise SystemExit("--trainval-percent 必须在 (0, 1] 之间")
    if not 0.0 < args.train_percent <= 1.0:
        raise SystemExit("--train-percent 必须在 (0, 1] 之间")

    print("=" * 70)
    print("生成 ImageSets/Main 划分索引")
    print("=" * 70)
    print("数据集根目录 : %s" % voc_root)
    print("输出目录     : %s%s" % (out_dir, "  (dry-run，不写文件)" if args.dry_run else ""))
    print("")

    images, annos = collect(voc_root)

    img_stems = set(images)
    anno_stems = set(annos)
    paired = sorted(img_stems & anno_stems)
    no_anno = sorted(img_stems - anno_stems)
    no_image = sorted(anno_stems - img_stems)

    print("图片总数           : %d" % len(img_stems))
    print("标注文件总数       : %d" % len(anno_stems))
    print("图片与标注配对成功 : %d" % len(paired))
    print("")

    if no_anno:
        print("[警告] 有 %d 张图片**没有标注**，不会进入任何索引文件。" % len(no_anno))
        print("       例如：%s" % ", ".join(no_anno[:8]))
        print("       请用 LabelImg 补标（标签统一填 cat），再重新运行本脚本。")
    if no_image:
        print("[警告] 有 %d 个标注**找不到对应图片**（可能是图片被删或改名）：" % len(no_image))
        print("       例如：%s" % ", ".join(no_image[:8]))
    if not no_anno and not no_image and paired:
        print("[ OK ] 图片与标注完全配对。")

    coverage = (len(paired) / len(img_stems) * 100.0) if img_stems else 0.0
    print("")
    print("标注覆盖率         : %d / %d = %.1f%%"
          % (len(paired), len(img_stems), coverage))
    if paired and coverage < 60.0:
        print("       !! 覆盖率偏低。YOLOv3 从头训练通常需要每张图都有标注，")
        print("          未标注的图片会被当成背景，显著拉低检测效果。")
    print("")

    if not paired:
        print("[错误] 没有任何「图片 + 标注」配对，无法生成有效的索引文件。")
        print("       请先用 LabelImg 标注图片（Open Dir 选 JPEGImages，")
        print("       Change Save Dir 选 Annotations，标签名填 cat）。")
        return 1

    # ---------------- 划分 ----------------
    rng = random.Random(args.seed)
    stems = list(paired)
    rng.shuffle(stems)

    n_total = len(stems)
    n_trainval = int(round(n_total * args.trainval_percent))

    # 小数据集兜底：至少要有一条进入 train，否则训练会直接失败
    if n_total >= 2:
        n_trainval = max(1, min(n_total - 1, n_trainval))
    else:
        n_trainval = n_total

    trainval = stems[:n_trainval]
    test = stems[n_trainval:]

    n_train = int(round(len(trainval) * args.train_percent))
    if len(trainval) >= 2:
        n_train = max(1, min(len(trainval) - 1, n_train))
    else:
        n_train = len(trainval)

    train = trainval[:n_train]
    val = trainval[n_train:]

    print("-" * 70)
    print("划分结果（seed=%d, trainval_percent=%.2f, train_percent=%.2f）"
          % (args.seed, args.trainval_percent, args.train_percent))
    print("-" * 70)
    print("  trainval : %4d 条  (%.1f%%)" % (len(trainval), len(trainval) / n_total * 100))
    print("  train    : %4d 条  (%.1f%%)" % (len(train), len(train) / n_total * 100))
    print("  val      : %4d 条  (%.1f%%)" % (len(val), len(val) / n_total * 100))
    print("  test     : %4d 条  (%.1f%%)" % (len(test), len(test) / n_total * 100))
    print("")
    if len(train) == 0:
        print("[错误] train 为空，无法训练。请补充标注后重试。")
        return 1
    if len(val) == 0:
        print("[警告] val 为空。训练脚本里 val_split 会拿不到验证集，")
        print("       建议标注更多图片，或调大 --train-percent 让 trainval 更小。")
    if len(train) < 20:
        print("[警告] train 只有 %d 条。指导书的实验环境用 230 张图训练，" % len(train))
        print("       当前规模远小于此，检测效果会很差。")

    if args.dry_run:
        print("")
        print("(dry-run) 未写入任何文件。")
        return 0

    # ---------------- 写文件 ----------------
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "trainval.txt": sorted(trainval),
        "train.txt": sorted(train),
        "val.txt": sorted(val),
        "test.txt": sorted(test),
    }
    print("")
    print("-" * 70)
    for name, items in files.items():
        p = out_dir / name
        # 用 \n 而不是 Windows 默认的 \r\n：keras-yolo3 里 strip() 后按 ',' split，
        # 但索引文件被其它脚本读取时按行比较，统一 LF 最稳。
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for stem in items:
                fh.write(stem + "\n")
        print("  已写入 %-14s %4d 行  ->  %s" % (name, len(items), p))

    print("")
    print("完成。下一步：")
    print("  python voc_annotation.py     # 生成 train.txt / val.txt / test.txt（含框坐标）")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
