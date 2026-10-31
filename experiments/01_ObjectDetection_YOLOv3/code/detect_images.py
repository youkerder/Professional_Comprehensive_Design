#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 01 批量检测：用训练好的权重对一批图片做猫脸检测并保存结果。

对应《目标检测实验指导书》2.2 节。指导书要求把 `yolo.py` 里加载的 `.h5` 权重路径
改掉，然后用改写过 `detect_img` 的 `yolo_video.py` 跑 `--image`，把结果存到 `outdir`。
指导书那段 `detect_img` 是：

    def detect_img(yolo):
        path = "D:\\VOCdevkit\\VOC2007\\test\\*.jpg"          # 硬编码
        outdir = "D:\\VOCdevkit\\VOC2007\\SegmentationClass"  # 硬编码
        for jpgfile in glob.glob(path):
            img = Image.open(jpgfile)
            img = yolo.detect_image(img)
            img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        yolo.close_session()

本脚本把同样的流程做成命令行工具：
  * 权重 / 输入 / 输出都是参数，不硬编码；
  * 额外导出检测框**坐标** CSV（上游 detect_image 只返回图，拿不到坐标），
    供 evaluate_detection.py 算指标、供报告画图；
  * 统计耗时与检出数；
  * 默认压掉上游 detect_image 里逐张打印的调试输出。

用法：
    # 对指导书里的 8 张演示图检测（这 8 张**没有标注**，只能看效果）
    python detect_images.py --weights ../models/xxx_trained_weights.h5

    # 跑有标注的测试划分 —— 结果可以直接接 evaluate_detection.py 算 mAP
    python detect_images.py --weights <权重> --split test

    # 指定输入输出
    python detect_images.py --weights <权重> --input ../../datasets/cat_face/test --output ../outputs/detect/run1

    # 调低阈值多检出一些
    python detect_images.py --weights <权重> --score 0.2

    # 用 tiny 模型（anchor 文件也要换成 6 组的那个）
    python detect_images.py --weights <tiny权重> --anchors model_data/tiny_yolo_anchors.txt
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from config import (
        ANCHORS_PATH, CLASSES_PATH, IMAGE_EXTS, IMAGESETS_DIR, JPEG_DIR,
        OUTPUTS_DIR, TEST_IMAGES_DIR, ensure_dirs,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        ANCHORS_PATH, CLASSES_PATH, IMAGE_EXTS, IMAGESETS_DIR, JPEG_DIR,
        OUTPUTS_DIR, TEST_IMAGES_DIR, ensure_dirs,
    )

QUIET = io.StringIO()


def images_from_split(split: str, jpeg_dir: Path = JPEG_DIR,
                      sets_dir: Path = IMAGESETS_DIR):
    """按 ImageSets/Main/<split>.txt 里的文件名列表取图。

    为什么需要这个：指导书用于演示的 8 张图（datasets/cat_face/test/）
    **没有标注**，对它们跑检测只能看效果、算不了 mAP。
    而 ImageSets 里的 test 划分是有标注的，用它跑检测就能直接接上
    evaluate_detection.py 算指标。
    """
    list_file = sets_dir / ("%s.txt" % split)
    if not list_file.is_file():
        raise SystemExit(
            "找不到 %s。\n"
            "       先运行 make_imagesets.py 生成划分，或用 --input 指定图片目录。"
            % list_file)

    stems = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    out, missing = [], []
    for stem in stems:
        for ext in IMAGE_EXTS:
            p = jpeg_dir / (stem + ext)
            if p.is_file():
                out.append(p)
                break
        else:
            missing.append(stem)
    if missing:
        print("[警告] %d 个文件名在 %s 下找不到对应图片，例如：%s"
              % (len(missing), jpeg_dir, ", ".join(missing[:5])))
    return out


def collect_inputs(src: Path):
    """输入可以是单张图，或一个目录（递归）。"""
    if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
        return [src]
    if src.is_dir():
        return sorted(p for p in src.rglob("*")
                      if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return []


def boxes_to_str(boxes, scores, classes):
    """把 yolo_eval 的 (top,left,bottom,right) 转成 x0,y0,x1,y1 形式的字符串。"""
    items = []
    for box, score, cls in zip(boxes, scores, classes):
        top, left, bottom, right = [int(round(float(v))) for v in box]
        items.append("%d,%d,%d,%d,%.4f,%d" % (left, top, right, bottom, float(score), int(cls)))
    return ";".join(items)


def main():
    ap = argparse.ArgumentParser(
        description="实验 01：批量猫脸检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--weights", required=True, help="训练好的 .h5 权重")
    ap.add_argument("--input", default=str(TEST_IMAGES_DIR),
                    help="输入图片或目录（默认：指导书的 8 张演示图）")
    ap.add_argument("--split", default=None,
                    help="改为按 ImageSets/Main/<split>.txt 选图（如 test / val）。"
                         "用这个跑出来的结果**有标注**，可以直接接 evaluate_detection.py")
    ap.add_argument("--output", default=None,
                    help="结果保存目录（默认：outputs/detect/<权重名>）")
    ap.add_argument("--anchors", default=str(ANCHORS_PATH), help="anchor 文件")
    ap.add_argument("--classes", default=str(CLASSES_PATH), help="类别文件")
    ap.add_argument("--score", type=float, default=0.3,
                    help="置信度阈值（上游 yolo.py 默认 0.3）")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值")
    ap.add_argument("--model-image-size", type=int, default=416,
                    help="推理输入边长（要与训练时一致，32 的倍数）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全部）")
    ap.add_argument("--verbose", action="store_true",
                    help="保留上游 detect_image 的逐张调试输出")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ensure_dirs()
    weights = Path(args.weights).resolve()
    src = Path(args.input).resolve()

    print("=" * 70)
    print("实验 01 · 批量猫脸检测")
    print("=" * 70)
    print("权重         : %s" % weights)
    print("输入         : %s" % src)
    print("anchor       : %s" % args.anchors)
    print("类别         : %s" % args.classes)
    print("score / iou  : %.2f / %.2f" % (args.score, args.iou))
    print("")

    if not weights.is_file():
        print("[错误] 找不到权重文件：%s" % weights)
        print("       先训练：python train.py --epochs 200")
        return 1
    if not Path(args.anchors).is_file():
        print("[错误] 找不到 anchor 文件：%s" % args.anchors)
        return 1
    if not Path(args.classes).is_file():
        print("[错误] 找不到类别文件：%s" % args.classes)
        print("       可用 voc_annotation.py 生成。")
        return 1

    if args.split:
        print("划分         : %s（来自 ImageSets/Main/%s.txt）" % (args.split, args.split))
        images = images_from_split(args.split)
    else:
        images = collect_inputs(src)
        if not images:
            print("[错误] 在 %s 下没有找到图片。" % src)
            print("       可用 --input 指定单张图或目录。")
            print("       也可以 --split test 直接跑有标注的测试划分。")
            return 1
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        print("[错误] 没有取到任何图片。")
        return 1
    print("待检测图片   : %d 张" % len(images))
    print("")

    # ---------------- 加载模型 ----------------
    print("正在加载模型（首次运行较慢）...")
    t_load = time.time()
    try:
        from detector import BoxYOLO
    except Exception as e:
        print("[错误] 导入 detector/yolo 失败：%s: %s" % (type(e).__name__, e))
        print("       yolo.py 及其依赖（yolo3/、font/）是否已就位？")
        print("       可运行 fetch_base_code.py 获取。")
        return 2

    try:
        yolo = BoxYOLO(model_path=str(weights),
                       anchors_path=str(Path(args.anchors).resolve()),
                       classes_path=str(Path(args.classes).resolve()),
                       score=args.score,
                       iou=args.iou,
                       model_image_size=(args.model_image_size, args.model_image_size),
                       gpu_num=1)
    except Exception as e:
        print("[错误] 模型加载失败：%s: %s" % (type(e).__name__, e))
        print("")
        print("       常见原因：")
        print("         * 权重与类别数/anchor 数不匹配 —— 检查 --anchors 是否与训练时一致")
        print("           （YOLOv3 用 9 组 anchors，Tiny-YOLOv3 用 6 组，")
        print("             即 model_data/yolo_anchors.txt 或 tiny_yolo_anchors.txt）")
        print("         * 权重文件损坏或不是本实验训练的")
        return 2
    print("模型加载完成，用时 %.1f 秒" % (time.time() - t_load))
    print("")

    # ---------------- 输出目录 ----------------
    out_dir = Path(args.output).resolve() if args.output else (OUTPUTS_DIR / "detect" / weights.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("结果目录     : %s" % out_dir)
    print("")

    # ---------------- 逐张检测 ----------------
    from PIL import Image

    rows = []
    n_ok = n_err = n_empty = 0
    total_boxes = 0
    times = []
    t_all = time.time()

    print("开始检测 ...")
    for i, img_path in enumerate(images, 1):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            n_err += 1
            print("  [%d/%d] %s  打开失败：%s" % (i, len(images), img_path.name, e))
            rows.append({"filename": img_path.name, "num_boxes": 0, "boxes": "",
                         "elapsed_s": "", "status": "open_error"})
            continue

        t0 = time.time()
        try:
            if args.verbose:
                result, boxes, scores, classes = yolo.detect(image)
            else:
                # 上游 detect_image 里有 print(image_data.shape) 等调试输出，
                # 逐张刷屏没有意义，这里临时吞掉（异常仍走 stderr）
                with contextlib.redirect_stdout(QUIET):
                    result, boxes, scores, classes = yolo.detect(image)
        except Exception as e:
            n_err += 1
            print("  [%d/%d] %s  检测失败：%s: %s"
                  % (i, len(images), img_path.name, type(e).__name__, e))
            rows.append({"filename": img_path.name, "num_boxes": 0, "boxes": "",
                         "elapsed_s": "", "status": "detect_error"})
            continue
        dt = time.time() - t0
        times.append(dt)

        result.save(out_dir / img_path.name)

        nb = len(boxes)
        total_boxes += nb
        if nb == 0:
            n_empty += 1
        n_ok += 1
        rows.append({"filename": img_path.name, "num_boxes": nb,
                     "boxes": boxes_to_str(boxes, scores, classes),
                     "elapsed_s": "%.3f" % dt,
                     "status": "ok" if nb else "no_detection"})
        if i % 20 == 0 or i == len(images):
            print("  已处理 %d/%d ..." % (i, len(images)))

    elapsed = time.time() - t_all
    print("")
    print("-" * 70)
    print("处理完成：成功 %d 张，失败 %d 张，用时 %.1f 秒" % (n_ok, n_err, elapsed))
    print("检出框总数 : %d" % total_boxes)
    print("未检出     : %d 张" % n_empty)
    if times:
        print("单张耗时   : 平均 %.3f s，最快 %.3f s，最慢 %.3f s"
              % (sum(times) / len(times), min(times), max(times)))
        print("等效吞吐   : %.1f 张/秒" % (len(times) / sum(times)))
    print("结果图片   : %s" % out_dir)

    # ---------------- 清单 ----------------
    manifest = out_dir / "detections.csv"
    with open(manifest, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "num_boxes", "boxes",
                                           "elapsed_s", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("检测明细   : %s" % manifest)
    print("             （boxes 列格式：x0,y0,x1,y1,score,class_id，多个框用 ; 分隔）")

    yolo.close_session()

    print("")
    print("要算准确率 / mAP：")
    print("  python evaluate_detection.py --pred \"%s\"" % manifest)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
