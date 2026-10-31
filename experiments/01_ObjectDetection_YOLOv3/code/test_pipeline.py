#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据管线自检：在临时目录造一份合成 VOC 数据集，跑通
make_imagesets.py 与 voc_annotation.py，并断言各种边界情况的处理是否正确。

真实数据集里目前只有 1 张图有标注，覆盖不到多框、脏数据、缺标注等分支，
所以用这个自检来保证管线逻辑正确。

用法：
    python test_pipeline.py
    python test_pipeline.py --keep      # 保留临时目录，便于人工查看

退出码 0 = 全部通过。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# 临时工作目录放在仓库内（沙箱下系统临时目录可能不允许创建嵌套目录），
# 跑完自动删除；加 --keep 可保留。
REPO_ROOT = HERE.parents[2]
SCRATCH_ROOT = REPO_ROOT / ".selftest"

# --------------------------------------------------------------------------
# 断言小工具
# --------------------------------------------------------------------------
FAILED = []
PASSED = []


def check(cond, label, detail=""):
    if cond:
        PASSED.append(label)
        print("  [ OK ] %s" % label)
    else:
        FAILED.append(label)
        print("  [FAIL] %s  %s" % (label, detail))


def write_xml(path: Path, filename: str, width: int, height: int, objects):
    """objects: [(name, xmin, ymin, xmax, ymax, difficult)]"""
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    for (name, x0, y0, x1, y1, diff) in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = str(diff)
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(x0)
        ET.SubElement(bnd, "ymin").text = str(y0)
        ET.SubElement(bnd, "xmax").text = str(x1)
        ET.SubElement(bnd, "ymax").text = str(y1)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="\t")
    tree.write(str(path), encoding="utf-8", xml_declaration=False)


def read_lines(p: Path):
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def run(script: Path, *args, cwd: Path, expect_ok=True):
    cmd = [PY, str(script), *args]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True,
                          encoding="utf-8", errors="replace")
    if expect_ok and proc.returncode != 0:
        print("      ---- 子进程输出 ----")
        print(proc.stdout)
        print(proc.stderr)
    return proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    # 注意：不要用 tempfile.mkdtemp()。它在 Windows 上给目录设置 mode 0700，
    # 等于写了一个「禁止继承」的保护性 ACL，会挡住沙箱/其它账户的写入授权，
    # 导致在这个目录下连子目录都建不了（WinError 5）。
    # 这里用普通 mkdir，让它继承父目录的权限项。
    stamp = "%s_%d" % (time.strftime("%Y%m%d_%H%M%S"), os.getpid())
    tmp = SCRATCH_ROOT / ("voc_selftest_" + stamp)
    tmp.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("数据管线自检")
    print("=" * 74)
    print("临时目录：%s" % tmp)

    try:
        # ---------------- 构造合成数据集 ----------------
        voc = tmp / "cat_face"
        jpeg = voc / "JPEGImages"
        anno = voc / "Annotations"
        jpeg.mkdir(parents=True)
        anno.mkdir(parents=True)
        (voc / "ImageSets" / "Main").mkdir(parents=True)

        from PIL import Image
        W, H = 200, 150

        def make_image(stem):
            Image.new("RGB", (W, H), (120, 120, 120)).save(jpeg / (stem + ".jpg"))

        # 16 张正常图（1~2 个框）
        normal = ["%03d" % i for i in range(1, 17)]
        for i, stem in enumerate(normal):
            make_image(stem)
            objs = [("cat", 10, 10, 60, 60, 0)]
            if i % 3 == 0:
                objs.append(("cat", 100, 70, 180, 140, 0))
            write_xml(anno / (stem + ".xml"), stem + ".jpg", W, H, objs)

        # 边界情况
        make_image("d01")  # difficult=1 -> 应被跳过
        write_xml(anno / "d01.xml", "d01.jpg", W, H, [("cat", 10, 10, 60, 60, 1)])

        make_image("d02")  # 未定义类别 dog -> 应被跳过，图变空 -> 丢弃
        write_xml(anno / "d02.xml", "d02.jpg", W, H, [("dog", 10, 10, 60, 60, 0)])

        make_image("d03")  # 坐标反转 -> 应交换
        write_xml(anno / "d03.xml", "d03.jpg", W, H, [("cat", 60, 60, 10, 10, 0)])

        make_image("d04")  # 越界 -> 应裁剪到图内
        write_xml(anno / "d04.xml", "d04.jpg", W, H, [("cat", -20, -20, 500, 500, 0)])

        make_image("d05")  # 退化框（宽=0）-> 应跳过，图变空 -> 丢弃
        write_xml(anno / "d05.xml", "d05.jpg", W, H, [("cat", 40, 40, 40, 80, 0)])

        # 有图无标注：3 张
        for stem in ("m01", "m02", "m03"):
            make_image(stem)

        # 有标注无图：1 个孤立 xml
        write_xml(anno / "orphan.xml", "orphan.jpg", W, H, [("cat", 5, 5, 50, 50, 0)])

        n_img = len(list(jpeg.glob("*.jpg")))
        n_xml = len(list(anno.glob("*.xml")))
        print("")
        print("合成数据：%d 张图，%d 个 xml" % (n_img, n_xml))

        # ---------------- make_imagesets ----------------
        print("")
        print("-" * 74)
        print("第 1 步：make_imagesets.py")
        print("-" * 74)
        proc = run(HERE / "make_imagesets.py",
                   "--voc-root", str(voc),
                   "--out-dir", str(voc / "ImageSets" / "Main"),
                   cwd=HERE)
        out1 = proc.stdout + proc.stderr
        check(proc.returncode == 0, "退出码为 0", "实际 %d" % proc.returncode)
        check("204" not in out1, "没有出现错误的图片计数")
        check("有 3 张图片**没有标注**" in out1 or "有 3 张图片" in out1,
              "正确报告 3 张有图无标注", out1[out1.find("警告"):][:120])
        check("orphan" in out1, "正确报告 1 个孤立标注（无对应图片）")

        sets = voc / "ImageSets" / "Main"
        tr = read_lines(sets / "train.txt") or []
        va = read_lines(sets / "val.txt") or []
        te = read_lines(sets / "test.txt") or []
        tv = read_lines(sets / "trainval.txt") or []
        print("      trainval=%d train=%d val=%d test=%d" % (len(tv), len(tr), len(va), len(te)))

        # 配对成功的 stem：16 张正常 + d01..d05（共 21 个有 xml 的）。
        # make_imagesets 只看「有没有 xml」，不看框是否有效，所以 21 个全部进索引。
        # 21 * 0.9 = 18.9 -> round -> 19 进 trainval，余 2 进 test。
        # 19 * 0.9 = 17.1 -> round -> 17 train，余 2 val。
        all_paired = len(tv) + len(te)
        check(len(tv) == 19, "trainval = 19", "实际 %d" % len(tv))
        check(all_paired == 21, "配对成功总数 = 21", "实际 %d" % all_paired)
        check(len(tr) + len(va) == len(tv), "train + val == trainval",
              "%d + %d != %d" % (len(tr), len(va), len(tv)))
        check(len(tr) + len(va) + len(te) == all_paired,
              "train + val + test == 配对总数（互斥且完备）",
              "%d != %d" % (len(tr) + len(va) + len(te), all_paired))
        check(len(set(tr) & set(va)) == 0, "train 与 val 无重叠")
        check(len(set(tr) & set(te)) == 0, "train 与 test 无重叠")
        check(len(set(va) & set(te)) == 0, "val 与 test 无重叠")
        check(len(set(tr) | set(va) | set(te)) == all_paired,
              "三个划分合并后覆盖全部配对样本，无重复无遗漏",
              "并集 %d，应为 %d" % (len(set(tr) | set(va) | set(te)), all_paired))

        # ---------------- voc_annotation ----------------
        print("")
        print("-" * 74)
        print("第 2 步：voc_annotation.py")
        print("-" * 74)
        outdir = tmp / "code_out"
        outdir.mkdir()
        proc = run(HERE / "voc_annotation.py",
                   "--voc-root", str(voc),
                   "--imagesets-dir", str(sets),
                   "--out-dir", str(outdir),
                   "--absolute-paths",
                   cwd=HERE)
        out2 = proc.stdout + proc.stderr
        check(proc.returncode == 0, "退出码为 0", "实际 %d" % proc.returncode)

        trl = read_lines(outdir / "train.txt") or []
        val_ = read_lines(outdir / "val.txt") or []
        tel = read_lines(outdir / "test.txt") or []
        print("      行数 train=%d val=%d test=%d" % (len(trl), len(val_), len(tel)))

        # d01(difficult) / d02(dog) / d05(退化) 会被丢弃 → 有效图 18
        check("类别名未定义 1 处" in out2, "报告 1 处未定义类别 dog")
        check("difficult=1 被跳过 1 处" in out2, "报告 1 处 difficult 跳过")
        # d03 的框是 (60,60)-(10,10)，x 和 y 两个方向都反了，所以计 2 次交换
        check("坐标反转已交换 2 处" in out2, "报告 2 处坐标反转（一个框 x/y 各一次）")
        check("越界已裁剪 1 处" in out2, "报告 1 处越界裁剪")
        check("退化框 1 处" in out2, "报告 1 处退化框")
        # m01..m03 没有 xml，在 make_imagesets 阶段就被排除了，
        # 不会出现在索引里，所以这里不应该报「找不到标注」
        check("找不到标注" not in out2,
              "无标注的图片在 make_imagesets 阶段已被排除（voc_annotation 不报缺标注）")
        check("找不到图片" not in out2, "所有索引条目都能找到对应图片")

        total_lines = len(trl) + len(val_) + len(tel)
        check(total_lines == 18, "有效图共 18 张（21 - d01 - d02 - d05）",
              "实际 %d" % total_lines)

        # 逐行校验格式与坐标合法性
        # 行格式（keras-yolo3 口径）：<图片路径> <x0,y0,x1,y1,cls> <x0,y0,x1,y1,cls> ...
        #   路径与各框之间用**空格**分隔，逗号只在框内部
        bad_fields = 0
        bad_coord = 0
        img_missing = 0
        box_count = 0
        two_box_images = 0
        for ln in trl + val_ + tel:
            groups = ln.split()
            if len(groups) < 2:
                bad_fields += 1
                continue
            path, box_groups = groups[0], groups[1:]
            if not Path(path).is_file():
                img_missing += 1
            if len(box_groups) == 2:
                two_box_images += 1
            for g in box_groups:
                parts = g.split(",")
                if len(parts) != 5:
                    bad_fields += 1
                    continue
                # 顺序是 xmin,ymin,xmax,ymax,class_id
                x0, y0, x1, y1, cid = parts
                box_count += 1
                if int(x1) <= int(x0) or int(y1) <= int(y0):
                    bad_coord += 1
                if cid != "0":
                    bad_coord += 1
        check(bad_fields == 0, "所有行格式合法（路径 + 空格分隔的 5 字段框）",
              "%d 处非法" % bad_fields)
        check(img_missing == 0, "所有图片路径都存在", "%d 行路径无效" % img_missing)
        check(bad_coord == 0, "所有框坐标合法且类别 id 正确", "%d 处异常" % bad_coord)
        # 16 张正常图里 i%3==0 的有 6 张（i=0,3,6,9,12,15）是双框，其余 10 张单框
        # -> 6*2 + 10*1 = 22 个框；再加 d03、d04 各 1 个 = 24
        check(two_box_images == 6, "有 6 张双框图", "实际 %d" % two_box_images)
        check(box_count == 24, "目标框总数 = 24（22 + d03 + d04）", "实际 %d" % box_count)

        # 关键：行首必须是**纯路径**，能被 PIL 打开
        # get_random_data 里是 line = annotation_line.split() 然后 Image.open(line[0])
        first = (trl + val_ + tel)[0]
        first_path = first.split()[0]
        check("," not in first_path, "行首字段不含逗号（否则 get_random_data 会解析失败）",
              first_path)
        check(Path(first_path).is_file(), "行首字段是可直接打开的图片路径", first_path)
        try:
            from PIL import Image
            with Image.open(first_path) as im:
                check(True, "PIL 能打开行首路径（%dx%d）" % im.size)
        except Exception as e:
            check(False, "PIL 能打开行首路径", "%s: %s" % (type(e).__name__, e))

        def boxes_of(line):
            """从一行里取出所有框的坐标四元组（新格式：路径 + 空格分隔的框）。"""
            return [g.split(",")[:4] for g in line.split()[1:]]

        # d03 交换后的框应为 10,10,60,60
        d03 = [ln for ln in trl + val_ + tel if "d03" in ln]
        check(len(d03) == 1 and boxes_of(d03[0]) == [["10", "10", "60", "60"]],
              "d03 坐标反转已正确交换为 10,10,60,60",
              boxes_of(d03[0]) if d03 else "找不到 d03")

        # d04 裁剪后的框应为 0,0,199,149
        d04 = [ln for ln in trl + val_ + tel if "d04" in ln]
        check(len(d04) == 1 and boxes_of(d04[0]) == [["0", "0", "199", "149"]],
              "d04 越界已裁剪为 0,0,199,149",
              boxes_of(d04[0]) if d04 else "找不到 d04")

        # d01/d02/d05 不应出现
        for stem in ("d01", "d02", "d05", "m01", "orphan"):
            present = any(stem in ln for ln in trl + val_ + tel)
            check(not present, "%s 未被写入标注文件" % stem)

        # voc_classes.txt
        vc = HERE / "model_data" / "voc_classes.txt"
        check(vc.is_file() and read_lines(vc) == ["cat"],
              "model_data/voc_classes.txt 内容为 cat")

        # 覆盖率告警
        check("标注覆盖率" in out1, "输出了标注覆盖率")

    finally:
        if args.keep:
            print("")
            print("临时目录已保留：%s" % tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            # 顺手把空的父目录也清掉，别在工作区里留垃圾
            try:
                SCRATCH_ROOT.rmdir()
            except OSError:
                pass

    print("")
    print("=" * 74)
    print("通过 %d 项，失败 %d 项" % (len(PASSED), len(FAILED)))
    if FAILED:
        for f in FAILED:
            print("  [FAIL] %s" % f)
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
