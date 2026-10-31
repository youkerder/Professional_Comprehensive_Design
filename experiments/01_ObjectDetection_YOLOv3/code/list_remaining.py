#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""列出「还没人工改过」的图，并可把它们单独收集到一个目录，方便用 LabelImg 集中标注。

======================================================================
为什么需要
======================================================================
labelImg 打开的是整个 `JPEGImages`（205 张）。手动标到一半再想继续时，
要在 205 个文件里翻找剩下的几十张很麻烦。

本脚本通过与自动预标注时留下的清单
（`outputs/auto_annotate/manifest.csv`）逐一比对框坐标，
准确找出**还没被改动过**的图，然后把它们**硬链接**（不占额外空间）
到一个单独的目录，用 LabelImg 打开那个目录即可只看未完成的图。

注意：判断依据是**框坐标是否与预标注一致**，不能看 XML 里的
`source/database` 字段 —— labelImg 保存时会原样保留那个字段，
即使框被改得完全不同，它仍然写着 `auto-fasterrcnn`。

用法：
    python list_remaining.py                     # 只列出剩余
    python list_remaining.py --collect            # 同时收集到一个目录
    python list_remaining.py --collect --open     # 收集并直接启动 LabelImg
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from config import ANNO_DIR, IMAGE_EXTS, JPEG_DIR, OUTPUTS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ANNO_DIR, IMAGE_EXTS, JPEG_DIR, OUTPUTS_DIR

from auto_annotate import scan_status

AUTO_OUT = OUTPUTS_DIR / "auto_annotate"
COLLECT_DIR = OUTPUTS_DIR / "remaining_to_label"

# LabelImg 装在 pytorch 环境里（见 tools/start_labelimg.cmd 的说明）
# LabelImg 可执行文件：优先用环境变量 LABELIMG_EXE，默认取 PATH 上的 labelImg
LABELIMG = Path(os.environ.get("LABELIMG_EXE") or "labelImg")
PREDEFINED = Path(__file__).resolve().parents[3] / "datasets" / "cat_face" / "predefined_classes.txt"


def link_or_copy(src: Path, dst: Path):
    """优先建硬链接（不占空间），失败则退回复制。"""
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser(
        description="列出还没人工改过的图，并可集中到一个目录便于标注")
    ap.add_argument("--collect", action="store_true",
                    help="把剩余图片收集到 outputs/remaining_to_label/")
    ap.add_argument("--open", action="store_true", help="收集后直接启动 LabelImg")
    ap.add_argument("--out-dir", default=str(COLLECT_DIR), help="收集目录")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    (images, annotated, unannotated,
     reviewed, untouched) = scan_status(JPEG_DIR, ANNO_DIR, AUTO_OUT)

    todo = sorted(set(untouched) | set(unannotated))

    print("=" * 70)
    print("标注进度")
    print("=" * 70)
    print("  图片总数            : %d" % len(images))
    print("  已人工改过          : %d" % len(reviewed))
    print("  还是预标注(未改)    : %d" % len(untouched))
    print("  没有 XML            : %d  %s"
          % (len(unannotated), ("（%s）" % ", ".join(unannotated)) if unannotated else ""))
    print("")
    print("  剩余待处理          : %d" % len(todo))

    if todo:
        print("")
        print("  剩余清单：")
        # 每行 10 个，方便看
        for i in range(0, len(todo), 10):
            print("    %s" % "  ".join(todo[i:i + 10]))
    else:
        print("")
        print("  [ OK ] 全部完成，可以开始训练了。")

    if not args.collect:
        print("")
        print("  想把这些图单独收集到一个目录方便标注，加 --collect")
        return 0

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for stem in todo:
        src = None
        for ext in IMAGE_EXTS:
            p = JPEG_DIR / (stem + ext)
            if p.is_file():
                src = p
                break
        if src is None:
            print("  [警告] 找不到图片：%s" % stem)
            continue
        link_or_copy(src, out_dir / src.name)
        n += 1

    print("")
    print("  已收集 %d 张到：%s" % (n, out_dir))
    print("  （用硬链接，不额外占用磁盘空间）")

    if args.open:
        if not LABELIMG.is_file():
            print("  [错误] 找不到 LabelImg：%s" % LABELIMG)
            return 1
        print("")
        print("  正在启动 LabelImg ...")
        subprocess.Popen([str(LABELIMG), str(out_dir), str(PREDEFINED), str(ANNO_DIR)])
        print("  注：图片目录是收集目录，保存目录仍是正式的 Annotations。")
    else:
        print("")
        print("  用 LabelImg 打开这个目录即可（保存目录仍指向正式 Annotations）：")
        print("    %s \"%s\" \"%s\" \"%s\"" % (LABELIMG, out_dir, PREDEFINED, ANNO_DIR))
        print("  或者双击：tools\\start_labelimg_remaining.cmd")

    return 0


if __name__ == "__main__":
    sys.exit(main())
