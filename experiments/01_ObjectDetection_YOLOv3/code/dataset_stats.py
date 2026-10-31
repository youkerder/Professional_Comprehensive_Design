#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 01 数据集统计：给实验报告的"数据集"一节提供真实数字。

统计内容：
  * 图片数量、分辨率分布、宽高比
  * 目标框数量、每图框数分布
  * 框的宽高与面积分布（对判断默认 anchor 是否合适有参考价值）
  * 标签名统计、`difficult=1` 数量
  * **标注来源**：多少是 `auto-fasterrcnn` 预标注（待人工收紧），多少已人工确认
  * 图片与标注的配对情况

只依赖标准库；加 --figures 时额外用 matplotlib 画分布图（需要 matplotlib 环境）。

用法：
    python dataset_stats.py
    python dataset_stats.py --figures
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    from config import ANNO_DIR, IMAGE_EXTS, JPEG_DIR, VOC_ROOT
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ANNO_DIR, IMAGE_EXTS, JPEG_DIR, VOC_ROOT

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "reports" / "tables"
FIG_DIR = REPO_ROOT / "reports" / "figures"


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def collect(jpeg_dir: Path, anno_dir: Path):
    images = {}
    for p in sorted(jpeg_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images[p.stem] = p

    annos = {p.stem: p for p in anno_dir.glob("*.xml")}

    box_w, box_h, box_area, box_ar = [], [], [], []
    boxes_per_image = []
    labels = Counter()
    databases = Counter()
    n_difficult = 0
    n_bad_coord = 0
    per_image_boxes = {}

    for stem, xml in annos.items():
        try:
            root = ET.parse(str(xml)).getroot()
        except ET.ParseError:
            continue
        databases[(root.findtext("source/database") or "(未标注来源)").strip()] += 1
        n = 0
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if not name:
                continue
            labels[name] += 1
            d = obj.findtext("difficult")
            try:
                if d is not None and int(float(d)) == 1:
                    n_difficult += 1
                    continue
            except (TypeError, ValueError):
                pass
            b = obj.find("bndbox")
            if b is None:
                continue
            try:
                x0 = float(b.findtext("xmin")); y0 = float(b.findtext("ymin"))
                x1 = float(b.findtext("xmax")); y1 = float(b.findtext("ymax"))
            except (TypeError, ValueError):
                n_bad_coord += 1
                continue
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w <= 0 or h <= 0:
                n_bad_coord += 1
                continue
            box_w.append(w)
            box_h.append(h)
            box_area.append(w * h)
            box_ar.append(w / h)
            n += 1
        boxes_per_image.append(n)
        per_image_boxes[stem] = n

    return {
        "images": images,
        "annos": annos,
        "box_w": box_w, "box_h": box_h, "box_area": box_area, "box_ar": box_ar,
        "boxes_per_image": boxes_per_image,
        "labels": labels, "databases": databases,
        "n_difficult": n_difficult, "n_bad_coord": n_bad_coord,
        "per_image_boxes": per_image_boxes,
    }


def image_sizes(images):
    sizes = Counter()
    ar = []
    try:
        from PIL import Image
    except ImportError:
        return sizes, ar
    for p in images.values():
        try:
            with Image.open(p) as im:
                w, h = im.size
            sizes[(w, h)] += 1
            if h:
                ar.append(w / h)
        except Exception:
            continue
    return sizes, ar


def fmt_stats(name, values, unit=""):
    if not values:
        return "| %s | - | - | - | - | - |" % name
    return ("| %s | %.1f | %.1f | %.1f | %.1f | %.1f |"
            % (name, min(values), percentile(values, 25),
               statistics.median(values), percentile(values, 75), max(values)))


def main():
    ap = argparse.ArgumentParser(description="实验 01 数据集统计")
    ap.add_argument("--jpeg-dir", default=str(JPEG_DIR))
    ap.add_argument("--anno-dir", default=str(ANNO_DIR))
    ap.add_argument("--out", default=str(TABLE_DIR / "01_dataset_stats.md"))
    ap.add_argument("--figures", action="store_true", help="额外画分布图")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    jpeg_dir = Path(args.jpeg_dir).resolve()
    anno_dir = Path(args.anno_dir).resolve()

    print("=" * 70)
    print("实验 01 · 数据集统计")
    print("=" * 70)
    print("图片目录 : %s" % jpeg_dir)
    print("标注目录 : %s" % anno_dir)
    print("")

    c = collect(jpeg_dir, anno_dir)
    n_img = len(c["images"])
    n_anno = len(c["annos"])
    paired = len(set(c["images"]) & set(c["annos"]))
    sizes, img_ar = image_sizes(c["images"])
    bp = c["boxes_per_image"]
    n_boxes = sum(bp)

    print("图片总数        : %d" % n_img)
    print("标注文件数      : %d" % n_anno)
    print("图片/标注配对   : %d" % paired)
    print("目标框总数      : %d" % n_boxes)
    print("标签名          : %s" % dict(c["labels"]))
    print("标注来源        : %s" % dict(c["databases"]))
    print("difficult=1     : %d" % c["n_difficult"])
    print("非法坐标        : %d" % c["n_bad_coord"])
    print("")

    # ---------------- Markdown ----------------
    L = []
    L.append("# 实验 01 数据集统计")
    L.append("")
    L.append("> 由 `code/dataset_stats.py` 自动生成，数字来自 `datasets/cat_face/` 的实际文件。")
    L.append("")
    L.append("## 一、总体情况")
    L.append("")
    L.append("| 项目 | 数值 |")
    L.append("|---|---:|")
    L.append("| 图片总数 | %d |" % n_img)
    L.append("| 标注文件数 | %d |" % n_anno)
    L.append("| 图片与标注配对成功 | %d |" % paired)
    L.append("| 目标框总数 | %d |" % n_boxes)
    L.append("| 标注覆盖率 | %.1f%% |" % (paired / n_img * 100 if n_img else 0))
    L.append("")

    L.append("## 二、标注来源")
    L.append("")
    L.append("| `source/database` | 文件数 | 说明 |")
    L.append("|---|---:|---|")
    desc = {
        "auto-fasterrcnn": "COCO 预训练 Faster R-CNN 自动预标注，框是**整只猫**，需人工收紧到猫脸",
        "Unknown": "LabelImg 手工标注（默认写的 `Unknown`），框是猫脸，可直接用于训练",
        "(未标注来源)": "没有 `<source><database>` 字段",
    }
    for k, v in c["databases"].most_common():
        L.append("| `%s` | %d | %s |" % (k, v, desc.get(k, "—")))
    L.append("")

    L.append("## 三、标签与目标")
    L.append("")
    L.append("| 标签名 | 数量 |")
    L.append("|---|---:|")
    for k, v in c["labels"].most_common():
        L.append("| `%s` | %d |" % (k, v))
    L.append("")
    if c["n_difficult"]:
        L.append("`difficult=1` 的目标：%d 个（训练时按 keras-yolo3 的约定跳过）" % c["n_difficult"])
        L.append("")
    if c["n_bad_coord"]:
        L.append("坐标非法或退化的目标：%d 个（已排除在统计之外）" % c["n_bad_coord"])
        L.append("")

    L.append("## 四、每图框数分布")
    L.append("")
    cnt = Counter(bp)
    L.append("| 每图框数 | 图片数 |")
    L.append("|---|---:|")
    for k in sorted(cnt):
        L.append("| %d | %d |" % (k, cnt[k]))
    L.append("")
    if bp:
        L.append("平均每图 %.2f 个框，中位数 %.1f 个。" % (n_boxes / len(bp), statistics.median(bp)))
        L.append("")

    L.append("## 五、目标框尺寸分布")
    L.append("")
    L.append("| 量 | 最小 | P25 | 中位数 | P75 | 最大 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    L.append(fmt_stats("宽 (px)", c["box_w"]))
    L.append(fmt_stats("高 (px)", c["box_h"]))
    L.append(fmt_stats("面积 (px²)", c["box_area"]))
    L.append(fmt_stats("宽高比 w/h", c["box_ar"]))
    L.append("")
    if c["box_area"]:
        L.append("框面积的中位数是 %.0f px²，开方后约 %.0f px 见方。" %
                 (statistics.median(c["box_area"]),
                  statistics.median(c["box_area"]) ** 0.5))
        L.append("")
        L.append("> 与默认 anchor 的对比：`model_data/yolo_anchors.txt` 里的 9 组 anchor 是")
        L.append("> COCO 上聚类得到的，最小一组为 10×13。可以用 `kmeans.py` 在本数据集上")
        L.append("> 重新聚类出更贴合猫脸的 anchor —— 这是一个值得做的改进点。")
        L.append("")

    L.append("## 六、图片分辨率分布")
    L.append("")
    if sizes:
        L.append("| 分辨率 (宽×高) | 图片数 |")
        L.append("|---|---:|")
        for (w, h), v in sizes.most_common(15):
            L.append("| %d×%d | %d |" % (w, h, v))
        L.append("")
        if len(sizes) > 15:
            L.append("（共 %d 种不同分辨率，上表只列前 15 种）" % len(sizes))
            L.append("")
        L.append("分辨率种类数：%d" % len(sizes))
        if img_ar:
            L.append("宽高比：中位数 %.2f，范围 %.2f ~ %.2f"
                     % (statistics.median(img_ar), min(img_ar), max(img_ar)))
        L.append("")
        L.append("> 分辨率不统一是正常的：训练时 `get_random_data` 会把图片 letterbox 到")
        L.append("> `input_shape`（默认 416×416），所以不需要预先统一尺寸。")
        L.append("")
    else:
        L.append("（读取图片尺寸失败，需要 Pillow）")
        L.append("")

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("已写入 Markdown : %s" % out)

    js = out.with_suffix(".json")
    js.write_text(json.dumps({
        "images": n_img, "annotations": n_anno, "paired": paired,
        "boxes": n_boxes, "labels": dict(c["labels"]),
        "databases": dict(c["databases"]),
        "difficult": c["n_difficult"], "bad_coord": c["n_bad_coord"],
        "boxes_per_image": dict(Counter(bp)),
        "box_w": {"min": min(c["box_w"]) if c["box_w"] else 0,
                  "median": statistics.median(c["box_w"]) if c["box_w"] else 0,
                  "max": max(c["box_w"]) if c["box_w"] else 0},
        "box_h": {"min": min(c["box_h"]) if c["box_h"] else 0,
                  "median": statistics.median(c["box_h"]) if c["box_h"] else 0,
                  "max": max(c["box_h"]) if c["box_h"] else 0},
        "resolutions": {"%dx%d" % k: v for k, v in sizes.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写入 JSON     : %s" % js)

    # ---------------- 可选分布图 ----------------
    if args.figures:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["figure.dpi"] = 130
            plt.rcParams["savefig.bbox"] = "tight"

            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
            axes[0].hist(c["box_w"], bins=30, color="#4C78A8")
            axes[0].set_title("框宽分布"); axes[0].set_xlabel("px"); axes[0].set_ylabel("数量")
            axes[1].hist(c["box_h"], bins=30, color="#F58518")
            axes[1].set_title("框高分布"); axes[1].set_xlabel("px")
            axes[2].hist(c["box_ar"], bins=30, color="#54A24B")
            axes[2].set_title("框宽高比分布"); axes[2].set_xlabel("w/h")
            for a in axes:
                a.grid(alpha=0.3)
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            fp = FIG_DIR / "01_box_distribution.png"
            fig.savefig(fp)
            plt.close(fig)
            print("已写入分布图   : %s" % fp)
        except ImportError:
            print("[跳过] 没装 matplotlib，未画分布图。可用 cv 环境：")
            print("  python dataset_stats.py --figures")

    print("")
    print("提示：把 %s 的内容贴进报告的「数据集」一节。" % out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
