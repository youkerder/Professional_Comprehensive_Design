#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动预标注：用 COCO 预训练的 Faster R-CNN 检出图片里的猫，生成 LabelImg 可编辑的 VOC XML。

用途：实验 01 有 205 张猫咪图片但只有 1 个标注，先自动框一遍，
再用 LabelImg 打开微调（把整只猫的框收紧到**猫脸**），比从零画框快很多。

======================================================================
重要说明：这是"预标注"，不是"最终标注"
======================================================================
* COCO 数据集里只有 **cat（整只猫）** 这个类别，**没有 cat face（猫脸）**。
  所以模型给出的框是**整只猫**，而本实验要检测的是**猫脸**。
  这些框**必须**用 LabelImg 人工收紧，不能直接拿去训练。
* 生成的 XML 里 `<source><database>` 被写成 `auto-fasterrcnn`，
  方便区分哪些是机器标的、哪些是人改过的。
* 默认**不会覆盖**已经存在的 XML，人工改过的成果不会被冲掉。

集成流程：
    python auto_annotate.py            # 1. 自动预标注（本脚本）
    labelImg                           # 2. 人工微调（Open Dir / Change Save Dir）
    python auto_annotate.py --report   # 3. 看看还剩多少张没处理
    python make_imagesets.py           # 4. 生成索引
    python voc_annotation.py           # 5. 生成训练标注

用法：
    python auto_annotate.py                     # 标注所有缺标注的图
    python auto_annotate.py --limit 5 --preview # 先试 5 张，并出预览图
    python auto_annotate.py --score 0.6         # 提高置信度阈值
    python auto_annotate.py --head-box          # 顺带用启发式往头部收一下（见下）
    python auto_annotate.py --report            # 只统计当前标注情况，不跑模型
    python auto_annotate.py --overwrite         # 覆盖已存在的 XML（慎用）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from config import ANNO_DIR, IMAGE_EXTS, JPEG_DIR, MODELS_DIR, OUTPUTS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ANNO_DIR, IMAGE_EXTS, JPEG_DIR, MODELS_DIR, OUTPUTS_DIR

# COCO 里 cat 的下标（0 是 __background__）
COCO_CAT_INDEX = 17
AUTO_TAG = "auto-fasterrcnn"


def ensure_torch_home():
    """决定预训练权重缓存到哪儿。

    优先用 PyTorch 默认的 `~/.cache/torch`；如果那个位置**不可写**
    （受限账户、或在文件沙箱里运行时很常见），就退回到本实验的
    `models/torch_hub/`，让权重跟着项目走，便于复现和迁移。

    必须在 import torch **之前**调用，因为 TORCH_HOME 是在导入时读取的。
    """
    env = os.environ.get("TORCH_HOME")
    if env:
        return Path(env), "已由环境变量 TORCH_HOME 指定"

    default = Path.home() / ".cache" / "torch"
    try:
        (default / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
        return default, "PyTorch 默认缓存"
    except OSError:
        pass

    fallback = MODELS_DIR / "torch_hub"
    (fallback / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(fallback)
    return fallback, "默认缓存不可写，改用工作区内目录"


# --------------------------------------------------------------------------
# 统计 / 报告
# --------------------------------------------------------------------------

def read_xml_boxes(path: Path):
    """读一个 VOC XML 里的所有框，返回 [(xmin,ymin,xmax,ymax), ...]（已排序）。"""
    try:
        root = ET.parse(str(path)).getroot()
    except Exception:
        return None
    out = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        if b is None:
            continue
        try:
            out.append(tuple(int(float(b.findtext(k)))
                              for k in ("xmin", "ymin", "xmax", "ymax")))
        except (TypeError, ValueError):
            pass
    return sorted(out)


def load_manifest_boxes(out_dir: Path):
    """读自动预标注时留下的清单，返回 {stem: [(x0,y0,x1,y1), ...]}。

    这是判断「这张图人工改过没有」的**可靠依据**：
    不能看 XML 里的 `source/database`，因为 labelImg 保存时会原样保留该字段，
    即使框被改得面目全非，它仍然写着 `auto-fasterrcnn`。
    所以只能拿当前框和预标注时的框逐一比对。
    """
    import csv
    man = Path(out_dir) / "manifest.csv"
    if not man.is_file():
        return None
    out = {}
    with open(man, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            boxes = []
            for item in (row.get("boxes") or "").split(";"):
                parts = item.split(",")
                if len(parts) >= 4:
                    try:
                        boxes.append(tuple(int(float(v)) for v in parts[:4]))
                    except ValueError:
                        pass
            out[Path(fn).stem] = sorted(boxes)
    return out or None


def scan_status(jpeg_dir: Path, anno_dir: Path, auto_out_dir: Path = None):
    """返回 (images, annotated, unannotated, reviewed, untouched)。

    `reviewed` / `untouched` 通过与预标注清单比对得出：
      * 清单里有、当前框与清单**不同** -> reviewed（人工改过）
      * 清单里有、当前框与清单**相同** -> untouched（还没改）
      * 清单里没有（人工新建的 XML）  -> reviewed
    """
    images = {}
    for p in sorted(jpeg_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images[p.stem] = p

    annotated = set()
    for stem in images:
        if (anno_dir / (stem + ".xml")).is_file():
            annotated.add(stem)

    manifest = load_manifest_boxes(auto_out_dir) if auto_out_dir else None

    reviewed, untouched = set(), set()
    for stem in annotated:
        xml = anno_dir / (stem + ".xml")
        cur = read_xml_boxes(xml)
        if manifest is None:
            # 没有清单时退回到 database 字段（不准，仅作兜底）
            try:
                db = (ET.parse(str(xml)).getroot().findtext("source/database") or "").strip()
            except Exception:
                db = ""
            (untouched if db.startswith("auto") else reviewed).add(stem)
            continue
        if stem not in manifest:
            reviewed.add(stem)               # 清单里没有 -> 人工新建的
        elif cur is not None and cur == manifest[stem]:
            untouched.add(stem)              # 与预标注完全一致 -> 还没改
        else:
            reviewed.add(stem)               # 改过了（也可能删光了框）

    unannotated = sorted(set(images) - annotated)
    return (images, sorted(annotated), unannotated,
            sorted(reviewed), sorted(untouched))


def do_report(jpeg_dir: Path, anno_dir: Path, auto_out_dir: Path = None):
    (images, annotated, unannotated,
     reviewed, untouched) = scan_status(jpeg_dir, anno_dir, auto_out_dir)

    print("=" * 70)
    print("当前标注情况")
    print("=" * 70)
    print("图片总数            : %d" % len(images))
    print("已有 XML            : %d" % len(annotated))
    print("  人工改过的        : %d" % len(reviewed))
    print("  与预标注一致(未改): %d" % len(untouched))
    print("没有 XML            : %d" % len(unannotated))
    print("")
    if len(images):
        print("标注覆盖率          : %d / %d = %.1f%%"
              % (len(annotated), len(images), len(annotated) / len(images) * 100))
    print("")
    if auto_out_dir and load_manifest_boxes(auto_out_dir) is None:
        print("[警告] 找不到预标注清单 %s，只能用 XML 里的 database 字段判断，"
              % (Path(auto_out_dir) / "manifest.csv"))
        print("       而 labelImg 保存时不会改该字段，所以下面的数字不可信。")

    if unannotated:
        print("")
        print("没有 XML 的图：%s" % ", ".join(unannotated))
    if untouched:
        print("")
        print("** 这些图还是「整只猫」的预标注框，需要收紧到猫脸：**")
        print("   %d 个：%s" % (len(untouched), ", ".join(untouched)))

    print("")
    todo = len(untouched) + len(unannotated)
    if todo == 0:
        print("[ OK ] 全部 %d 张都已完成人工标注。" % len(annotated))
    else:
        print("[待办] 还有 %d 张需要处理（未改 %d + 未标注 %d）"
              % (todo, len(untouched), len(unannotated)))
    return todo


# --------------------------------------------------------------------------
# 写 VOC XML
# --------------------------------------------------------------------------

def write_voc_xml(path: Path, filename: str, img_path: Path,
                  width: int, height: int, boxes, database: str = AUTO_TAG):
    """boxes: [(name, xmin, ymin, xmax, ymax, score)]"""
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = str(img_path)

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = database

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(root, "segmented").text = "0"

    for (name, x0, y0, x1, y1, score) in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        # score 不是 VOC 标准字段，但 LabelImg 会保留未知子节点，
        # 解析 VOC 的脚本（含本仓库的 voc_annotation.py）也不会读它
        if score is not None:
            ET.SubElement(obj, "score").text = "%.4f" % score
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(x0))
        ET.SubElement(bnd, "ymin").text = str(int(y0))
        ET.SubElement(bnd, "xmax").text = str(int(x1))
        ET.SubElement(bnd, "ymax").text = str(int(y1))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")
    tree.write(str(path), encoding="utf-8", xml_declaration=False)


def head_box(x0, y0, x1, y1, frac, top):
    """把整只猫的框启发式地往头部收。

    猫脸大致位于身体框的**上部中央**，所以：
      * 宽度取原框的 frac 倍，水平居中；
      * 高度取原框的 frac 倍；
      * 垂直方向从 y0 + h*top 开始。
    frac=1.0 且 top=0 时等价于原框。

    这只是个粗略猜测，不同姿态（趴着/坐着/侧躺）差异很大，
    所以默认**不启用**，需要显式加 --head-box。
    """
    w, h = x1 - x0, y1 - y0
    nw, nh = w * frac, h * frac
    cx = x0 + w / 2.0
    nx0 = cx - nw / 2.0
    ny0 = y0 + h * top
    return nx0, ny0, nx0 + nw, ny0 + nh


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="用 COCO 预训练 Faster R-CNN 自动预标注猫咪图片（生成 VOC XML）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--images-dir", default=str(JPEG_DIR), help="图片目录")
    ap.add_argument("--anno-dir", default=str(ANNO_DIR), help="XML 输出目录")
    ap.add_argument("--score", type=float, default=0.5,
                    help="猫的置信度阈值，低于此值不写入（默认 0.5）")
    ap.add_argument("--max-cats", type=int, default=0,
                    help="每张图最多保留几只猫（0 = 不限，默认 0）")
    ap.add_argument("--overwrite", action="store_true",
                    help="覆盖已存在的 XML（默认跳过，保护人工标注成果）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 张（0 = 全部），用于先试跑")
    ap.add_argument("--head-box", action="store_true",
                    help="启用启发式「往头部收」（默认关闭，框是整只猫）")
    ap.add_argument("--head-frac", type=float, default=0.6,
                    help="--head-box 时，宽高取原框的比例（默认 0.6）")
    ap.add_argument("--head-top", type=float, default=0.0,
                    help="--head-box 时，框顶从原框顶部往下偏移的比例（默认 0.0）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="推理设备（默认 auto）")
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="保存 N 张带框预览图到 outputs/auto_annotate/preview/")
    ap.add_argument("--report", action="store_true",
                    help="只统计当前标注情况，不跑模型")
    ap.add_argument("--out-dir", default=str(OUTPUTS_DIR / "auto_annotate"),
                    help="清单与预览图的输出目录")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    jpeg_dir = Path(args.images_dir).resolve()
    anno_dir = Path(args.anno_dir).resolve()

    print("=" * 70)
    print("实验 01 · 自动预标注（COCO Faster R-CNN -> VOC XML）")
    print("=" * 70)
    print("图片目录 : %s" % jpeg_dir)
    print("标注目录 : %s" % anno_dir)
    print("")

    if not jpeg_dir.is_dir():
        raise SystemExit("找不到图片目录：%s" % jpeg_dir)
    anno_dir.mkdir(parents=True, exist_ok=True)

    # 预标注清单所在目录（用于判断哪些图人工改过）
    auto_out = Path(args.out_dir).resolve()

    if args.report:
        do_report(jpeg_dir, anno_dir, auto_out)
        return 0

    # ---------------- 选图 ----------------
    (images, annotated, unannotated,
     reviewed, untouched) = scan_status(jpeg_dir, anno_dir, auto_out)

    if args.overwrite:
        targets = sorted(images)
    else:
        targets = unannotated

    if args.limit > 0:
        targets = targets[:args.limit]

    print("图片总数        : %d" % len(images))
    print("已有 XML        : %d（人工改过 %d，与预标注一致 %d）"
          % (len(annotated), len(reviewed), len(untouched)))
    print("本次待处理      : %d %s"
          % (len(targets), "(--overwrite 全覆盖)" if args.overwrite else "(仅缺标注的)"))
    print("")

    if not targets:
        print("[ OK ] 没有需要处理的图片。")
        print("       如需重新预标注全部图片，加 --overwrite（会覆盖人工改动，慎用）。")
        do_report(jpeg_dir, anno_dir, auto_out)
        return 0

    # ---------------- 加载模型 ----------------
    # 必须在 import torch 之前定好权重缓存位置
    torch_home, why = ensure_torch_home()

    try:
        import torch
        from PIL import Image
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn,
        )
        from torchvision.transforms.functional import to_tensor
    except ImportError as e:
        print("[错误] 缺少依赖：%s" % e)
        print("       自动预标注需要在 pytorch 环境里运行：")
        print("       python auto_annotate.py")
        print("")
        print("       若确实不想装 PyTorch，就跳过自动预标注，直接用 LabelImg 手标，")
        print("       然后运行 make_imagesets.py 与 voc_annotation.py。")
        return 2

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[警告] 指定了 cuda 但不可用，回退到 cpu。")
        device = "cpu"

    print("推理设备        : %s%s" % (device,
          ("  (%s)" % torch.cuda.get_device_name(0)) if device == "cuda" else ""))
    print("置信度阈值      : %.2f" % args.score)
    print("框的形式        : %s" % ("启发式往头部收 (frac=%.2f, top=%.2f)"
          % (args.head_frac, args.head_top) if args.head_box else "整只猫（需人工收紧到猫脸）"))
    print("权重缓存目录    : %s  (%s)" % (torch_home, why))
    print("")
    print("正在加载 COCO 预训练权重（首次运行需要联网下载约 160MB）...")
    t0 = time.time()
    try:
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(weights=weights)
    except Exception as e:
        print("")
        print("[错误] 加载预训练权重失败：%s: %s" % (type(e).__name__, e))
        print("       常见原因：")
        print("         1) 没有联网 —— 首次运行必须联网下载权重；")
        print("         2) 缓存目录不可写 —— 已尝试回退到 models/torch_hub/；")
        print("         3) 代理/防火墙拦住了 download.pytorch.org。")
        print("       也可以手动下载后放到：")
        print("         %s" % (torch_home / "hub" / "checkpoints"))
        print("         文件名：fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
        print("")
        print("       实在拿不到模型就跳过自动预标注，直接用 LabelImg 手工标注。")
        return 2
    model.eval().to(device)
    print("加载完成，用时 %.1f 秒" % (time.time() - t0))
    print("")

    # ---------------- 推理 ----------------
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "preview"
    if args.preview > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
        iterator = tqdm(targets, desc="预标注", unit="张", ncols=90)
    except ImportError:
        iterator = targets

    rows = []
    n_ok = n_empty = n_err = 0
    total_boxes = 0
    t_start = time.time()

    with torch.no_grad():
        for i, stem in enumerate(iterator):
            img_path = images[stem]
            try:
                pil = Image.open(img_path)
                pil = pil.convert("RGB")
                W, H = pil.size
                tensor = to_tensor(pil).to(device)
                pred = model([tensor])[0]
            except Exception as e:
                n_err += 1
                rows.append({"filename": img_path.name, "n_cats": 0, "best_score": "",
                             "boxes": "", "status": "error:%s" % type(e).__name__})
                print("  [错误] %s: %s" % (img_path.name, e))
                continue

            # 挑出 cat 且分数达标的框
            cand = []
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if int(label) != COCO_CAT_INDEX:
                    continue
                s = float(score)
                if s < args.score:
                    continue
                x0, y0, x1, y1 = [float(v) for v in box]
                cand.append((s, x0, y0, x1, y1))

            cand.sort(key=lambda t: -t[0])
            if args.max_cats > 0:
                cand = cand[:args.max_cats]

            if not cand:
                n_empty += 1
                rows.append({"filename": img_path.name, "n_cats": 0, "best_score": "",
                             "boxes": "", "status": "no_detection"})
                continue

            boxes = []
            for (s, x0, y0, x1, y1) in cand:
                if args.head_box:
                    x0, y0, x1, y1 = head_box(x0, y0, x1, y1,
                                              args.head_frac, args.head_top)
                # 裁剪到图片范围内
                x0 = max(0.0, min(x0, W - 1))
                y0 = max(0.0, min(y0, H - 1))
                x1 = max(0.0, min(x1, W - 1))
                y1 = max(0.0, min(y1, H - 1))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                boxes.append(("cat", x0, y0, x1, y1, s))

            if not boxes:
                n_empty += 1
                rows.append({"filename": img_path.name, "n_cats": 0, "best_score": "",
                             "boxes": "", "status": "all_boxes_degenerate"})
                continue

            write_voc_xml(anno_dir / (stem + ".xml"), img_path.name,
                          img_path, W, H, boxes)
            n_ok += 1
            total_boxes += len(boxes)
            rows.append({
                "filename": img_path.name,
                "n_cats": len(boxes),
                "best_score": "%.4f" % boxes[0][5],
                "boxes": ";".join("%d,%d,%d,%d" % (int(b[1]), int(b[2]), int(b[3]), int(b[4]))
                                  for b in boxes),
                "status": "auto",
            })

            # 预览图
            if args.preview > 0 and i < args.preview:
                _draw_preview(pil, boxes, preview_dir / (stem + "_preview.jpg"))

    elapsed = time.time() - t_start
    n_done = n_ok + n_empty + n_err
    print("")
    print("-" * 70)
    print("处理完成：%d 张，用时 %.1f 秒（平均 %.2f 秒/张）"
          % (n_done, elapsed, elapsed / n_done if n_done else 0))
    print("  生成 XML        : %d 张，共 %d 个框" % (n_ok, total_boxes))
    print("  未检出猫        : %d 张（需要手工画框）" % n_empty)
    print("  出错            : %d 张" % n_err)
    if n_ok:
        print("  平均每图猫数    : %.2f" % (total_boxes / n_ok))

    # ---------------- 清单 ----------------
    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["filename", "n_cats", "best_score", "boxes", "status"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("")
    print("清单已写入：%s" % manifest)
    if args.preview > 0:
        print("预览图目录：%s" % preview_dir)

    # ---------------- 后续提醒 ----------------
    print("")
    print("=" * 70)
    print("接下来必须做的事")
    print("=" * 70)
    print("生成的框是**整只猫**，而本实验要检测的是**猫脸**。")
    print("请用 LabelImg 打开逐张收紧框：")
    print("  1) labelImg")
    print("  2) Open Dir      -> %s" % jpeg_dir)
    print("  3) Change Save Dir -> %s" % anno_dir)
    print("  4) 逐张把框收紧到猫脸（标签保持 cat），Ctrl+S 保存")
    print("")
    print("查看还剩多少没处理：")
    print("  python auto_annotate.py --report")
    print("")
    print("标签名一定要是 cat，别改成 Cat / cat_face 之类，否则类别数会算错。")
    if n_empty:
        print("")
        print("注意有 %d 张没检出猫，这些必须手工画框（清单里 status=no_detection）。" % n_empty)
    return 0


def _draw_preview(pil, boxes, out_path: Path):
    """画带框的预览图，方便肉眼快速确认预标注质量。"""
    try:
        from PIL import ImageDraw
    except ImportError:
        return
    im = pil.copy()
    d = ImageDraw.Draw(im)
    for (name, x0, y0, x1, y1, score) in boxes:
        d.rectangle([x0, y0, x1, y1], outline=(255, 60, 60), width=3)
        label = "%s %.2f" % (name, score)
        ty = max(0, y0 - 14)
        d.rectangle([x0, ty, x0 + 8 * len(label), ty + 14], fill=(255, 60, 60))
        d.text((x0 + 3, ty + 1), label, fill=(255, 255, 255))
    im.save(out_path, quality=88)


if __name__ == "__main__":
    sys.exit(main())
