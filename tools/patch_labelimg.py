#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修补 LabelImg 与新版 PyQt5 的 float/int 类型不兼容（不改其它行为）。

======================================================================
问题
======================================================================
labelImg 1.8.3 有多处把**浮点数**传给只接受 int 的 PyQt5 API。
`QPointF.x()/.y()` 返回 float，`-delta / (8*15)` 这种真除法也返回 float，
而 `QPainter.drawLine(int,int,int,int)`、`drawRect(int,int,int,int)`、
`drawText(int,int,str)`、`QScrollBar.setValue(int)`、`QSpinBox.setValue(int)`
都严格要求 int。

老版本 PyQt5 会默默把 float 转成 int，新版（sip 6.x）改成严格校验，
于是这些地方全部抛 TypeError。表现为**能启动、能开图，但一画框/一滚动/一缩放就崩**。

共 8 处，分布在 3 个文件：

    labelImg/labelImg.py  4 处（scrollRequest / setZoom / zoomRequest）
    libs/canvas.py        3 处（paintEvent 里画框和十字准线）
    libs/shape.py         1 处（paint 里画标签文字）

修法是在这些调用外面套一层 `int()`。

======================================================================
用法
======================================================================
    python tools/patch_labelimg.py            # 打补丁并复查
    python tools/patch_labelimg.py --check     # 只检查当前状态
    python tools/patch_labelimg.py --revert    # 从 .bak 还原

脚本是**幂等**的，重复运行不会重复插 int()。首次修改每个文件前会自动备份。
labelImg 已停止维护，直接改安装目录里的文件是常规做法；重装后再跑一次即可。
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

# 相对 site-packages 的路径 -> [(原始写法, 修补后写法), ...]
PATCHES = {
    "labelImg/labelImg.py": [
        ("bar.setValue(bar.value() + bar.singleStep() * units)",
         "bar.setValue(int(bar.value() + bar.singleStep() * units))"),
        ("self.zoomWidget.setValue(value)",
         "self.zoomWidget.setValue(int(value))"),
        ("h_bar.setValue(new_h_bar_value)",
         "h_bar.setValue(int(new_h_bar_value))"),
        ("v_bar.setValue(new_v_bar_value)",
         "v_bar.setValue(int(new_v_bar_value))"),
    ],
    "libs/canvas.py": [
        ("p.drawRect(leftTop.x(), leftTop.y(), rectWidth, rectHeight)",
         "p.drawRect(int(leftTop.x()), int(leftTop.y()), "
         "int(rectWidth), int(rectHeight))"),
        ("p.drawLine(self.prevPoint.x(), 0, self.prevPoint.x(), self.pixmap.height())",
         "p.drawLine(int(self.prevPoint.x()), 0, "
         "int(self.prevPoint.x()), self.pixmap.height())"),
        ("p.drawLine(0, self.prevPoint.y(), self.pixmap.width(), self.prevPoint.y())",
         "p.drawLine(0, int(self.prevPoint.y()), "
         "self.pixmap.width(), int(self.prevPoint.y()))"),
    ],
    "libs/shape.py": [
        ("painter.drawText(min_x, min_y, self.label)",
         "painter.drawText(int(min_x), int(min_y), self.label)"),
    ],
}


def site_packages() -> Path:
    """定位 labelImg 所在的 site-packages。"""
    spec = importlib.util.find_spec("labelImg")
    if spec is None or not spec.origin:
        raise SystemExit(
            "[错误] 找不到已安装的 labelImg。\n"
            "       先安装：\n"
            "         pip install --only-binary=:all: labelImg lxml")
    return Path(spec.origin).parent.parent


def patch_file(path: Path, pairs, check_only: bool):
    """返回 (需要修补数, 已修补数, 未匹配数, 是否改动)。"""
    if not path.is_file():
        return 0, 0, len(pairs), False

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    fixed_map = dict(pairs)
    need = already = missing = 0
    changed = False

    for i, ln in enumerate(lines):
        s = ln.strip()
        hit = None
        for orig, fixed in pairs:
            if s == fixed:
                hit = ("already", orig)
                break
            if s == orig:
                hit = ("need", orig)
                break
        if hit is None:
            continue
        kind, orig = hit
        if kind == "already":
            already += 1
        else:
            need += 1
            indent = ln[: len(ln) - len(ln.lstrip())]
            trailing = "\n" if ln.endswith("\n") else ""
            lines[i] = "%s%s%s" % (indent, fixed_map[orig], trailing)
            changed = True

    matched = {orig for orig, _ in pairs}  # noqa: F841
    for orig, _ in pairs:
        if not any(ln.strip() in (orig, fixed_map[orig]) for ln in
                   path.read_text(encoding="utf-8").splitlines()):
            missing += 1

    if check_only or not changed:
        return need, already, 0, False

    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.is_file():
        shutil.copy2(path, bak)
        print("      已备份 -> %s" % bak.name)
    path.write_text("".join(lines), encoding="utf-8")
    return need, already, 0, True


def verify_syntax(path: Path) -> bool:
    import py_compile
    try:
        py_compile.compile(str(path), doraise=True)
        return True
    except py_compile.PyCompileError as e:
        print("      [错误] 语法检查失败：%s" % e)
        return False


def main():
    ap = argparse.ArgumentParser(description="修补 LabelImg 的 float/int 类型不兼容")
    ap.add_argument("--check", action="store_true", help="只检查，不修改")
    ap.add_argument("--revert", action="store_true", help="从 .bak 还原所有改动")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    sp = site_packages()
    print("=" * 72)
    print("修补 LabelImg 的 float -> int 类型不兼容")
    print("=" * 72)
    print("site-packages : %s" % sp)
    print("")

    if args.revert:
        n = 0
        for rel in PATCHES:
            p = sp / rel
            bak = p.with_suffix(p.suffix + ".bak")
            if bak.is_file():
                shutil.copy2(bak, p)
                print("  已还原 %s" % rel)
                n += 1
        if n == 0:
            print("  没有找到任何 .bak 备份，无需还原。")
        return 0

    total_need = total_already = total_missing = 0
    touched = []

    for rel, pairs in PATCHES.items():
        p = sp / rel
        print("  %s" % rel)
        if not p.is_file():
            print("      [跳过] 文件不存在")
            total_missing += len(pairs)
            continue

        need, already, _miss, changed = patch_file(p, pairs, args.check)
        total_need += need
        total_already += already
        total_missing += _miss

        status = []
        if already:
            status.append("已修补 %d" % already)
        if need:
            status.append("待修补 %d" % need)
        if not status:
            status.append("未找到预期写法（labelImg 版本可能不同）")
        print("      %s" % "，".join(status))

        if changed:
            if verify_syntax(p):
                print("      语法检查通过")
                touched.append(rel)
            else:
                shutil.copy2(p.with_suffix(p.suffix + ".bak"), p)
                print("      已还原该文件")
        elif not args.check and need == 0 and already:
            verify_syntax(p)

    print("")
    print("-" * 72)
    print("合计：待修补 %d 处，已修补 %d 处，未匹配 %d 处"
          % (total_need, total_already, total_missing))

    if args.check:
        if total_need:
            print("")
            print("有 %d 处需要修补，运行不带 --check 的命令即可。" % total_need)
            return 1
        print("")
        print("全部已修补。")
        return 0

    if total_need == 0 and not touched:
        print("")
        print("无需改动。")
        return 0

    print("")
    print("已修补文件：%s" % "、".join(touched))
    print("")
    print("请**重新启动** labelImg（先关掉正在运行的那个），")
    print("滚动、缩放、画框、画标签文字应该都不再崩了。")
    print("")
    print("建议接着跑一遍环境检查确认：")
    print("  python tools/check_labelimg_env.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
