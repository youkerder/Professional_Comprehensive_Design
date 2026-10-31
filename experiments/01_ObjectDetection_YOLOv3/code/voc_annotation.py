#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 VOC 格式的 XML 标注转换成 YOLOv3 训练用的标注文件。

输出格式（一行一张图，keras-yolo3 的 get_random_data 直接读）：

    <图片路径> <xmin>,<ymin>,<xmax>,<ymax>,<类别id> <xmin>,<ymin>,<xmax>,<ymax>,<类别id> ...

注意**分隔符**：图片路径与各框之间用**空格**，逗号只用在框内部。
原因见 yolo3/utils.py 的 get_random_data：

    line = annotation_line.split()      # 按空白切分
    image = Image.open(line[0])         # line[0] 必须是**纯路径**

生成的文件：
    code/train.txt         训练集
    code/val.txt           验证集
    code/test.txt          测试集
    code/model_data/voc_classes.txt    类别列表（每行一个类别）

======================================================================
与指导书 1.4 节的差别
======================================================================
指导书的 `voc_annotation.py` 由 keras-yolo3 提供，生成的是
`2007_train.txt` / `2007_test.txt`，然后**要求手动删掉文件名前缀的 `2007_`**。
原因在于它内部写死了 `VOC2007/JPEGImages/...` 这样的路径前缀。

本脚本：
  * 直接生成 `train.txt` / `val.txt` / `test.txt`，**不需要手动改名**；
  * 路径可配置，不写死 `VOC2007/`；
  * 路径默认写成**相对于 code/ 目录**的形式，所以必须在 code/ 目录下运行 train.py；
  * 额外做校验：坐标越界、坐标反转、类别名不在 CLASSES 里都会报告；
  * 输出数据集统计（图片数、框数、每类框数、每图框数分布）。

用法：
    cd experiments/01_ObjectDetection_YOLOv3/code
    python voc_annotation.py
    python voc_annotation.py --voc-root ../../datasets/cat_face
    python voc_annotation.py --absolute-paths
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    from config import ANNO_DIR, CLASSES, CODE_DIR, IMAGESETS_DIR, MODEL_DATA_DIR, VOC_ROOT
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ANNO_DIR, CLASSES, CODE_DIR, IMAGESETS_DIR, MODEL_DATA_DIR, VOC_ROOT


def parse_xml(xml_path: Path, class_to_id: dict, stats: Counter):
    """解析一个 VOC XML，返回 [(class_id, xmin, ymin, xmax, ymax), ...]。"""
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError as e:
        stats["xml_parse_error"] += 1
        print("  [错误] XML 解析失败 %s：%s" % (xml_path.name, e))
        return []

    size = root.find("size")
    img_w = img_h = None
    if size is not None:
        try:
            img_w = int(float(size.findtext("width", "0")))
            img_h = int(float(size.findtext("height", "0")))
        except (TypeError, ValueError):
            img_w = img_h = None

    boxes = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not name:
            stats["no_name"] += 1
            continue

        # 与 keras-yolo3 一致：跳过 difficult=1 的目标
        diff_txt = obj.findtext("difficult")
        try:
            difficult = int(float(diff_txt)) if diff_txt not in (None, "") else 0
        except (TypeError, ValueError):
            difficult = 0
        if difficult == 1:
            stats["skipped_difficult"] += 1
            continue

        if name not in class_to_id:
            stats["unknown_class"] += 1
            if stats["unknown_class"] <= 5:
                print("  [警告] %s 里有未定义的类别 %r（CLASSES 里没有），已跳过"
                      % (xml_path.name, name))
            continue

        bnd = obj.find("bndbox")
        if bnd is None:
            stats["no_bndbox"] += 1
            continue

        try:
            xmin = int(float(bnd.findtext("xmin")))
            ymin = int(float(bnd.findtext("ymin")))
            xmax = int(float(bnd.findtext("xmax")))
            ymax = int(float(bnd.findtext("ymax")))
        except (TypeError, ValueError):
            stats["bad_coord"] += 1
            print("  [警告] %s 有坐标为空或非数字的框，已跳过" % xml_path.name)
            continue

        # LabelImg 拖反了方向会产生 xmax<xmin，交换修正
        if xmax < xmin:
            xmin, xmax = xmax, xmin
            stats["swapped"] += 1
        if ymax < ymin:
            ymin, ymax = ymax, ymin
            stats["swapped"] += 1

        # 越界裁剪到图片范围内
        if img_w and img_h:
            cx0, cy0 = max(0, xmin), max(0, ymin)
            cx1, cy1 = min(img_w - 1, xmax), min(img_h - 1, ymax)
            if (cx0, cy0, cx1, cy1) != (xmin, ymin, xmax, ymax):
                stats["clipped"] += 1
            xmin, ymin, xmax, ymax = cx0, cy0, cx1, cy1

        if xmax <= xmin or ymax <= ymin:
            stats["degenerate"] += 1
            print("  [警告] %s 有退化框（宽或高为 0），已跳过" % xml_path.name)
            continue

        boxes.append((class_to_id[name], xmin, ymin, xmax, ymax))
        stats["boxes"] += 1
        stats["per_class"][name] += 1

    return boxes


def resolve_image(stem: str, jpeg_dir: Path):
    """按 stem 找到实际图片文件（扩展名可能不是 .jpg）。"""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"):
        p = jpeg_dir / (stem + ext)
        if p.is_file():
            return p
    return None


def build_split(split_name: str, stems, jpeg_dir: Path, anno_dir: Path,
                class_to_id: dict, use_absolute: bool):
    """为某个划分生成 (行文本列表, 统计)。"""
    lines = []
    stats = Counter()
    stats["per_class"] = Counter()
    missing_img = []
    missing_xml = []
    empty = []

    for stem in stems:
        stats["images"] += 1

        img = resolve_image(stem, jpeg_dir)
        if img is None:
            missing_img.append(stem)
            stats["images_dropped"] += 1
            continue

        xml = anno_dir / (stem + ".xml")
        if not xml.is_file():
            missing_xml.append(stem)
            stats["images_dropped"] += 1
            continue

        boxes = parse_xml(xml, class_to_id, stats)
        if not boxes:
            empty.append(stem)
            stats["images_dropped"] += 1
            continue

        if use_absolute:
            img_field = str(img.resolve())
        else:
            # 相对于 code/ 目录，正斜杠（Windows 下 PIL 也能正常打开）
            img_field = _rel(img, CODE_DIR)

        # 注意格式：keras-yolo3 的 yolo3/utils.py 里 get_random_data 是
        #     line = annotation_line.split()      # 按**空白**切分
        #     image = Image.open(line[0])
        # 所以一行必须是「图片路径 + 空格 + 每个框」，逗号只用在**框内部**：
        #     /path/to/1.jpg 189,54,352,218,0 100,70,180,140,0
        # 若整行都用逗号连接，line[0] 会变成整行，Image.open 直接 FileNotFoundError。
        boxes_str = " ".join("%d,%d,%d,%d,%d" % (x0, y0, x1, y1, cid)
                             for cid, x0, y0, x1, y1 in boxes)
        lines.append("%s %s" % (img_field, boxes_str))
        stats["images_used"] += 1
        stats["boxes_used"] += len(boxes)

    stats["missing_img"] = missing_img
    stats["missing_xml"] = missing_xml
    stats["empty"] = empty
    return lines, stats


def _rel(path: Path, base: Path) -> str:
    """相对路径，统一用正斜杠。"""
    import os
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def load_stems(imagesets_dir: Path, name: str):
    p = imagesets_dir / (name + ".txt")
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="把 VOC XML 标注转成 YOLOv3 训练用的标注文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--voc-root", default=str(VOC_ROOT),
                    help="VOC 数据集根目录（默认：datasets/cat_face）")
    ap.add_argument("--imagesets-dir", default=str(IMAGESETS_DIR),
                    help="ImageSets/Main 目录（默认：<voc-root>/ImageSets/Main）")
    ap.add_argument("--out-dir", default=str(CODE_DIR),
                    help="train.txt / val.txt / test.txt 的输出目录（默认：code/）")
    ap.add_argument("--absolute-paths", action="store_true",
                    help="图片路径写成绝对路径（默认相对于 code/ 目录）")
    ap.add_argument("--no-test", action="store_true",
                    help="不生成 test.txt（只生成 train/val）")
    args = ap.parse_args()

    print("=" * 70)
    print("生成 YOLOv3 标注文件")
    print("=" * 70)

    voc_root = Path(args.voc_root).resolve()
    imagesets_dir = Path(args.imagesets_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    jpeg_dir = voc_root / "JPEGImages"
    anno_dir = voc_root / "Annotations"

    print("数据集根目录 : %s" % voc_root)
    print("图片目录     : %s" % jpeg_dir)
    print("标注目录     : %s" % anno_dir)
    print("索引目录     : %s" % imagesets_dir)
    print("输出目录     : %s" % out_dir)
    print("路径模式     : %s" % ("绝对路径" if args.absolute_paths else "相对 code/ 目录"))
    print("")

    if not jpeg_dir.is_dir():
        raise SystemExit("找不到图片目录：%s" % jpeg_dir)
    if not anno_dir.is_dir():
        raise SystemExit("找不到标注目录：%s" % anno_dir)

    class_to_id = {name: i for i, name in enumerate(CLASSES)}
    print("类别映射     : %s" % class_to_id)
    print("")

    splits = ["train", "val"] + ([] if args.no_test else ["test"])
    results = {}
    for name in splits:
        stems = load_stems(imagesets_dir, name)
        if stems is None:
            print("[警告] 找不到 %s/%s.txt，跳过该划分。" % (imagesets_dir, name))
            print("       请先运行：python make_imagesets.py")
            continue
        lines, stats = build_split(name, stems, jpeg_dir, anno_dir,
                                   class_to_id, args.absolute_paths)
        results[name] = (lines, stats)

    if not results:
        print("")
        print("[错误] 没有任何可生成的划分。请先运行 make_imagesets.py 生成索引文件。")
        return 1

    print("-" * 70)
    print("统计")
    print("-" * 70)
    all_classes = Counter()
    total_used = total_boxes = 0
    for name, (lines, stats) in results.items():
        print("【%s】" % name)
        print("  索引条目           : %d" % stats["images"])
        print("  实际可用（图片+框）: %d" % stats["images_used"])
        print("  丢弃               : %d" % stats["images_dropped"])
        print("  目标框总数         : %d" % stats["boxes_used"])
        if stats["images_used"]:
            print("  平均每图框数       : %.2f" % (stats["boxes_used"] / stats["images_used"]))
        if stats["per_class"]:
            print("  各类别框数         : %s"
                  % ", ".join("%s×%d" % (k, v) for k, v in stats["per_class"].most_common()))
        for key, label in (("missing_img", "找不到图片"),
                           ("missing_xml", "找不到标注"),
                           ("empty", "标注里没有有效目标框")):
            if stats[key]:
                print("  [警告] %s %d 个：%s"
                      % (label, len(stats[key]), ", ".join(stats[key][:8])))
        for key, label in (("unknown_class", "类别名未定义"),
                           ("skipped_difficult", "difficult=1 被跳过"),
                           ("bad_coord", "坐标非数字"),
                           ("no_bndbox", "缺 bndbox"),
                           ("degenerate", "退化框"),
                           ("swapped", "坐标反转已交换"),
                           ("clipped", "越界已裁剪"),
                           ("xml_parse_error", "XML 解析失败")):
            if stats[key]:
                print("  提示：%s %d 处" % (label, stats[key]))
        print("")
        all_classes.update(stats["per_class"])
        total_used += stats["images_used"]
        total_boxes += stats["boxes_used"]

    print("合计：%d 张图，%d 个目标框" % (total_used, total_boxes))
    if all_classes:
        print("类别分布：%s" % ", ".join("%s×%d" % (k, v) for k, v in all_classes.most_common()))

    # ---------------- 写文件 ----------------
    out_dir.mkdir(parents=True, exist_ok=True)
    print("")
    print("-" * 70)
    written = 0
    skipped = []
    for name, (lines, stats) in results.items():
        p = out_dir / (name + ".txt")
        if not lines:
            # 不写 0 字节文件：空文件会被误认为「已经准备好」，
            # 训练时才会报 FileNotFoundError 或 "Train on 0 samples"，不如现在就提醒
            skipped.append(name)
            continue
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for ln in lines:
                fh.write(ln + "\n")
        print("  已写入 %-12s %4d 行  ->  %s" % (name + ".txt", len(lines), p))
        written += 1

    for name in skipped:
        print("  跳过   %-12s 0 行（没有可用样本，未生成文件）" % (name + ".txt"))
    if skipped and "train" in skipped:
        print("")
        print("[错误] 训练集为空，train.txt 未生成。请先补标注再运行：")
        print("       1) LabelImg 标注 -> 2) make_imagesets.py -> 3) 本脚本")

    # ---------------- 类别文件 ----------------
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    classes_file = MODEL_DATA_DIR / "voc_classes.txt"
    with open(classes_file, "w", encoding="utf-8", newline="\n") as fh:
        for name in CLASSES:
            fh.write(name + "\n")
    print("  已写入 %-12s %4d 行  ->  %s" % ("voc_classes.txt", len(CLASSES), classes_file))

    print("")
    print("-" * 70)
    print("提醒：yolov3.cfg 里有 3 处 yolo 层需要同步修改")
    print("      classes = %d" % len(CLASSES))
    print("      filters = 3 * (5 + %d) = %d" % (len(CLASSES), 3 * (5 + len(CLASSES))))
    print("      可运行 patch_yolov3_cfg.py 自动完成")
    print("")
    print("下一步：")
    print("  python patch_yolov3_cfg.py        # 修改 cfg 的 classes/filters")
    print("  python train.py                   # 开始训练（需要 tensorflow + keras）")
    return 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    sys.exit(main())
