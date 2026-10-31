#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按类别数修正 yolov3.cfg 里的 classes 与 filters。

对应《目标检测实验指导书》1.4.1 节：
> 打开文件后，ctrl+f，搜索 yolo，总共出现三处。每处都要修改：
>   filters: 3*(5+len(classes))
>   classes: len(classes) = 你标注的类别数
>   random：根据个人电脑配置决定，显存小则修改为 0

手工改很容易漏掉其中一处（三处 yolo 层各有一个 classes，
而每个 yolo 层**前面那个 [convolutional] 层**才有 filters），
本脚本自动完成并校验。

======================================================================
cfg 结构说明（为什么要这样找）
======================================================================
YOLOv3 的 cfg 里，检测头是这样成对出现的：

    [convolutional]          <- 这里有 filters
    size=1
    pad=1
    filters=255              <- 3*(5+80)=255（COCO 80 类）
    activation=linear

    [yolo]                   <- 这里有 classes
    mask = 6,7,8
    anchors = 10,13, ...     <- 共 9 组
    classes=80               <- 类别数
    ...

共出现 3 次（对应 3 个尺度）。所以：
  * `classes` 改 **3 处**（在 [yolo] 段里）
  * `filters` 改 **3 处**（在紧邻其前的 [convolutional] 段里）
  * 本实验只有 cat 一类：classes=1, filters=3*(5+1)=18

用法：
    python patch_yolov3_cfg.py                          # 用 config.py 里的类别
    python patch_yolov3_cfg.py --cfg model_data/yolov3.cfg
    python patch_yolov3_cfg.py --random 0               # 同时把 random 改成 0（显存小）
    python patch_yolov3_cfg.py --in-place               # 就地修改（会先备份）
    python patch_yolov3_cfg.py --check                  # 只检查，不修改
    python patch_yolov3_cfg.py --selftest               # 用合成 cfg 自检
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

try:
    from config import CLASSES, FILTERS, MODEL_DATA_DIR, NUM_CLASSES
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import CLASSES, FILTERS, MODEL_DATA_DIR, NUM_CLASSES

DEFAULT_CFG = MODEL_DATA_DIR / "yolov3.cfg"


# --------------------------------------------------------------------------
# 解析 / 修改
# --------------------------------------------------------------------------

def parse_sections(lines):
    """把 cfg 切成若干段：返回 [(section_name, start_idx, end_idx)]，end 为下一段的起点。

    cfg 的段以 `[name]` 开头，到下一个 `[name]` 之前结束。
    """
    starts = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("[") and s.endswith("]"):
            starts.append((s[1:-1].strip(), i))
    sections = []
    for k, (name, i) in enumerate(starts):
        end = starts[k + 1][1] if k + 1 < len(starts) else len(lines)
        sections.append((name, i, end))
    return sections


def find_yolo_layers(lines):
    """找到所有 [yolo] 段，以及各自**前面最近**的 [convolutional] 段。

    返回 [(yolo_idx, yolo_end, conv_idx, conv_end)]。
    """
    sections = parse_sections(lines)
    out = []
    for k, (name, i, end) in enumerate(sections):
        if name != "yolo":
            continue
        conv = None
        for j in range(k - 1, -1, -1):
            if sections[j][0] == "convolutional":
                conv = sections[j]
                break
            if sections[j][0] == "yolo":
                break
        out.append((i, end, conv[1] if conv else None, conv[2] if conv else None))
    return out


KEY_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)=(\s*)(.*?)(\s*)$")


def set_key(lines, start, end, key, value):
    """在 [start, end) 段里把 `key=...` 改成 `value`，返回 (是否找到, 旧值)。"""
    for i in range(start, end):
        m = KEY_RE.match(lines[i].rstrip("\n"))
        if m and m.group(2) == key:
            old = m.group(5)
            lines[i] = "%s%s%s=%s%s\n" % (m.group(1), key, m.group(3), value, m.group(6))
            return True, old
    return False, None


def get_key(lines, start, end, key):
    for i in range(start, end):
        m = KEY_RE.match(lines[i].rstrip("\n"))
        if m and m.group(2) == key:
            return m.group(5)
    return None


def patch_lines(lines, num_classes, random_value=None):
    """返回 (新 lines, 修改记录列表, 问题列表)。"""
    filters_want = 3 * (5 + num_classes)
    layers = find_yolo_layers(lines)
    changes = []
    problems = []

    if len(layers) != 3:
        problems.append("找到 %d 个 [yolo] 层，预期 3 个。cfg 可能不是标准 YOLOv3，"
                        "请人工确认。" % len(layers))

    for n, (yi, ye, ci, ce) in enumerate(layers, 1):
        # classes 在 [yolo] 段
        found, old = set_key(lines, yi, ye, "classes", str(num_classes))
        if not found:
            problems.append("第 %d 个 [yolo] 层里找不到 classes" % n)
        else:
            changes.append(("yolo#%d classes" % n, old, str(num_classes)))

        # filters 在紧邻其前的 [convolutional] 段
        if ci is None:
            problems.append("第 %d 个 [yolo] 层前面找不到 [convolutional]，无法改 filters" % n)
        else:
            found, old = set_key(lines, ci, ce, "filters", str(filters_want))
            if not found:
                problems.append("第 %d 个 [yolo] 层前的 [convolutional] 里找不到 filters" % n)
            else:
                changes.append(("conv#%d filters" % n, old, str(filters_want)))

        # mask / anchors 数量校验：每个 yolo 层的 mask 应指向 3 组 anchor
        mask = get_key(lines, yi, ye, "mask")
        anchors = get_key(lines, yi, ye, "anchors")
        if mask is not None:
            n_mask = len([x for x in mask.split(",") if x.strip()])
            if n_mask != 3:
                problems.append("第 %d 个 [yolo] 层的 mask 有 %d 个 anchor（通常应为 3 个）"
                                % (n, n_mask))
        if anchors is not None:
            n_anchor = len([x for x in anchors.split(",") if x.strip()])
            if n_anchor != 18:
                problems.append("第 %d 个 [yolo] 层的 anchors 有 %d 个数（9 组应为 18 个数）"
                                % (n, n_anchor))

        # random（可选）
        if random_value is not None:
            found, old = set_key(lines, yi, ye, "random", str(random_value))
            if found:
                changes.append(("yolo#%d random" % n, old, str(random_value)))
            else:
                problems.append("第 %d 个 [yolo] 层里找不到 random" % n)

    return lines, changes, problems


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

SYNTHETIC_CFG = """\
[net]
batch=1
subdivisions=1
width=416
height=416
channels=3

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
"""


def selftest() -> int:
    print("=" * 70)
    print("patch_yolov3_cfg.py 自检")
    print("=" * 70)
    failed = []
    passed = [0]

    def check(cond, label, detail=""):
        if cond:
            passed[0] += 1
            print("  [ OK ] %s" % label)
        else:
            print("  [FAIL] %s  %s" % (label, detail))
            failed.append(label)

    # 1) 解析出 3 个 yolo 层
    lines = SYNTHETIC_CFG.splitlines(keepends=True)
    layers = find_yolo_layers(lines)
    check(len(layers) == 3, "解析出 3 个 [yolo] 层", "实际 %d" % len(layers))
    check(all(ci is not None for (_a, _b, ci, _c) in layers),
          "每个 [yolo] 层都能找到紧邻其前的 [convolutional]")

    # 2) 打补丁：1 类
    lines2, changes, problems = patch_lines(list(lines), 1, random_value=0)
    check(not problems, "1 类时无任何问题", "; ".join(problems))
    check(len(changes) == 3 * 3, "共 9 处修改（3 层 × (classes+filters+random)）",
          "实际 %d" % len(changes))

    text = "".join(lines2)
    check(text.count("classes=1") == 3, "3 处 classes 都改成了 1",
          "出现 %d 次" % text.count("classes=1"))
    check(text.count("filters=18") == 3, "3 处 filters 都改成了 18",
          "出现 %d 次" % text.count("filters=18"))
    check(text.count("random=0") == 3, "3 处 random 都改成了 0",
          "出现 %d 次" % text.count("random=0"))
    check("classes=80" not in text, "旧的 classes=80 已全部替换")
    check("filters=255" not in text, "旧的 filters=255 已全部替换")

    # 3) 不同类别数
    lines3, _c, p3 = patch_lines(list(lines), 3)
    check(not p3, "3 类时无问题", "; ".join(p3))
    t3 = "".join(lines3)
    check(t3.count("classes=3") == 3, "3 类时 classes=3 出现 3 次")
    check(t3.count("filters=24") == 3, "3 类时 filters=3*(5+3)=24 出现 3 次",
          "出现 %d 次" % t3.count("filters=24"))

    # 4) random 不动时保持原值
    lines4, _c4, _p4 = patch_lines(list(lines), 1)
    check("random=1" in "".join(lines4), "不传 --random 时保持原值 random=1")

    # 5) 幂等：再打一次不应报错，且结果不变
    lines5, changes5, problems5 = patch_lines(list(lines2), 1, random_value=0)
    check(not problems5, "重复打补丁无问题", "; ".join(problems5))
    check("".join(lines5) == text, "结果幂等（重复运行不改变内容）")

    # 6) 异常 cfg：只有 2 个 yolo 层应被报出来
    parts = SYNTHETIC_CFG.split("[yolo]")
    short = parts[0] + "[yolo]" + parts[1] + "[yolo]" + parts[2]
    _l, _c6, p6 = patch_lines(short.splitlines(keepends=True), 1)
    check(any("3 个" in x for x in p6), "yolo 层数量不为 3 时给出提示", "; ".join(p6))

    # 7) mask 数量异常应被报出来
    bad_mask = SYNTHETIC_CFG.replace("mask = 6,7,8", "mask = 6,7")
    _l7, _c7, p7 = patch_lines(bad_mask.splitlines(keepends=True), 1)
    check(any("mask" in x for x in p7), "mask 里 anchor 数量异常时给出提示", "; ".join(p7))

    print("")
    print("=" * 70)
    if failed:
        print("通过 %d 项，失败 %d 项" % (passed[0], len(failed)))
        for f in failed:
            print("  [FAIL] %s" % f)
    else:
        print("通过 %d 项，全部通过" % passed[0])
    print("=" * 70)
    return 1 if failed else 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="按类别数修正 yolov3.cfg 的 classes 与 filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cfg", default=str(DEFAULT_CFG), help="yolov3.cfg 路径")
    ap.add_argument("--classes", type=int, default=NUM_CLASSES,
                    help="类别数（默认取 config.py 的 %d）" % NUM_CLASSES)
    ap.add_argument("--random", type=int, default=None, choices=[0, 1],
                    help="同时设置 random（显存小用 0）")
    ap.add_argument("--in-place", action="store_true",
                    help="就地修改（会先备份为 *.bak）")
    ap.add_argument("--out", default=None, help="输出路径（默认 <cfg>.patched）")
    ap.add_argument("--check", action="store_true", help="只检查当前 cfg，不修改")
    ap.add_argument("--selftest", action="store_true", help="用合成 cfg 自检")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.selftest:
        return selftest()

    cfg = Path(args.cfg).resolve()
    print("=" * 70)
    print("修正 yolov3.cfg 的 classes / filters")
    print("=" * 70)
    print("cfg 文件 : %s" % cfg)
    print("类别     : %d 类 -> %s" % (args.classes, CLASSES if args.classes == NUM_CLASSES
                                     else "(由 --classes 指定)"))
    print("filters  : 3 * (5 + %d) = %d" % (args.classes, 3 * (5 + args.classes)))
    print("")

    if not cfg.is_file():
        print("[错误] 找不到 %s" % cfg)
        print("")
        print("yolov3.cfg 是 keras-yolo3 仓库里的文件，需要先获取基础代码：")
        print("  git clone https://github.com/qqwweee/keras-yolo3")
        print("  然后把 yolov3.cfg 与 model_data/ 下的文件放到：")
        print("    %s" % MODEL_DATA_DIR)
        print("")
        print("也可以先跑 --selftest 验证本脚本功能：")
        print("  python patch_yolov3_cfg.py --selftest")
        return 1

    text = cfg.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    layers = find_yolo_layers(lines)
    print("检测到 %d 个 [yolo] 层" % len(layers))
    for n, (yi, ye, ci, ce) in enumerate(layers, 1):
        cur_c = get_key(lines, yi, ye, "classes")
        cur_f = get_key(lines, ci, ce, "filters") if ci is not None else None
        cur_r = get_key(lines, yi, ye, "random")
        print("  第 %d 层: classes=%-4s filters=%-5s random=%s"
              % (n, cur_c, cur_f, cur_r))
    print("")

    if args.check:
        want_f = 3 * (5 + args.classes)
        ok_all = True
        for n, (yi, ye, ci, ce) in enumerate(layers, 1):
            c = get_key(lines, yi, ye, "classes")
            f = get_key(lines, ci, ce, "filters") if ci is not None else None
            if c != str(args.classes) or f != str(want_f):
                print("  [不对] 第 %d 层应为 classes=%d filters=%d，实际 classes=%s filters=%s"
                      % (n, args.classes, want_f, c, f))
                ok_all = False
        if ok_all:
            print("  [ OK ] 3 处 classes 与 filters 都已正确设置为 %d / %d"
                  % (args.classes, want_f))
            return 0
        print("")
        print("运行不带 --check 的命令即可自动修正。")
        return 1

    new_lines, changes, problems = patch_lines(lines, args.classes, args.random)

    print("修改明细：")
    for name, old, new in changes:
        print("  %-20s %s -> %s" % (name, old, new))
    if not changes:
        print("  （没有需要修改的地方）")
    if problems:
        print("")
        print("需要注意：")
        for p in problems:
            print("  [警告] %s" % p)

    if args.in_place:
        out = cfg
        bak = cfg.with_suffix(cfg.suffix + ".bak")
        shutil.copy2(cfg, bak)
        print("")
        print("已备份原文件 -> %s" % bak)
    elif args.out:
        out = Path(args.out).resolve()
    else:
        out = cfg.with_suffix(cfg.suffix + ".patched")

    out.write_text("".join(new_lines), encoding="utf-8")
    print("")
    print("已写入 -> %s" % out)

    # 复核
    verify_lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    want_f = 3 * (5 + args.classes)
    print("")
    print("复核：")
    all_ok = True
    for n, (yi, ye, ci, ce) in enumerate(find_yolo_layers(verify_lines), 1):
        c = get_key(verify_lines, yi, ye, "classes")
        f = get_key(verify_lines, ci, ce, "filters") if ci is not None else None
        good = (c == str(args.classes) and f == str(want_f))
        all_ok = all_ok and good
        print("  第 %d 层: classes=%s filters=%s  %s" % (n, c, f, "OK" if good else "!! 不对"))
    if all_ok:
        print("")
        print("[ OK ] 三处 classes / filters 均已正确设置。")
    else:
        print("")
        print("[错误] 复核未通过，请人工检查 cfg。")
        return 1

    if not args.in_place:
        print("")
        print("下一步：把生成的文件替换到训练脚本使用的位置：")
        print("  copy \"%s\" \"%s\"" % (out.name, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
