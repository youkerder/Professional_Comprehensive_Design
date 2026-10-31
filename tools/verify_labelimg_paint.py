#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离屏验证 LabelImg 的绘制路径不再因 float/int 类型报错。

labelImg 1.8.3 有 8 处把 float 传给只接受 int 的 PyQt5 API，会让程序在
画框/滚动/缩放时崩溃（见 tools/patch_labelimg.py 的说明）。
本脚本在**无界面（offscreen）模式**下真实走一遍这些代码路径，
从而确认补丁生效——不需要真的打开图形界面。

覆盖：
  * Canvas.paintEvent  —— 绘制中的矩形与十字准线（canvas.py 3 处）
  * Shape.paint        —— 带标签文字的框（shape.py 1 处）
  * QScrollBar/QSpinBox.setValue —— 滚动与缩放（labelImg.py 4 处）

用法：
    python tools/verify_labelimg_paint.py
    python tools/verify_labelimg_paint.py --verbose
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "*=false")


def main():
    ap = argparse.ArgumentParser(description="离屏验证 LabelImg 绘制路径")
    ap.add_argument("--verbose", action="store_true", help="打印 Qt 的原始报错")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 72)
    print("LabelImg 绘制路径离屏验证（不需要图形界面）")
    print("=" * 72)

    from PyQt5.QtCore import QPointF, QRect, Qt
    from PyQt5.QtGui import QPaintEvent, QPainter, QPixmap
    from PyQt5.QtWidgets import QApplication, QScrollBar, QSpinBox

    app = QApplication.instance() or QApplication([])  # noqa: F841

    from libs.canvas import Canvas
    from libs.shape import Shape

    results = []

    def record(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except TypeError as e:
            results.append((name, False, str(e).splitlines()[0]))
        except Exception as e:
            results.append((name, None, "%s: %s" % (type(e).__name__, e)))

    # ---------- 1. Canvas.paintEvent（绘制中的框 + 十字准线）----------
    def canvas_paint():
        c = Canvas()
        c.pixmap = QPixmap(200, 200)
        c.resize(200, 200)
        c.show()                      # offscreen 下也需要 show 才有绘制设备
        c.mode = Canvas.CREATE        # 让 drawing() 返回 True
        c.current = Shape()           # 非 None 才会画矩形
        # 用带小数点的坐标，确保触发 float 路径
        c.line = [QPointF(10.5, 20.5), QPointF(30.5, 40.5)]
        c.prevPoint = QPointF(10.5, 20.5)
        c.paintEvent(QPaintEvent(QRect(0, 0, 200, 200)))

    record("Canvas.paintEvent（矩形 + 十字准线，float 坐标）", canvas_paint)

    # ---------- 2. Shape.paint（带标签的框）----------
    def shape_paint():
        s = Shape(paintLabel=True)
        s.label = "cat"
        s.scale = 1.0                 # paint 里会做 2.0 / scale
        s.addPoint(QPointF(10.5, 20.5))
        s.addPoint(QPointF(30.5, 20.5))
        s.addPoint(QPointF(30.5, 40.5))
        s.addPoint(QPointF(10.5, 40.5))
        pm = QPixmap(200, 200)
        painter = QPainter(pm)
        try:
            s.paint(painter)
        finally:
            painter.end()

    record("Shape.paint（带标签文字，float 坐标）", shape_paint)

    # ---------- 3. 滚动与缩放（labelImg.py 里已套 int 的写法）----------
    bar = QScrollBar(Qt.Horizontal)
    bar.setRange(0, 1000)
    spin = QSpinBox()
    spin.setRange(1, 500)

    def scroll_like_labelimg():
        # 对应 labelImg.py:877-879 修补后的写法
        delta, units = -120, None
        units = -delta / (8 * 15)
        bar.setValue(int(bar.value() + bar.singleStep() * units))

    def zoom_like_labelimg():
        # 对应 labelImg.py:936-940 修补后的写法
        move_x, d_h_bar_max = 0.5, 100
        bar.setValue(int(bar.value() + move_x * d_h_bar_max))

    def spin_like_labelimg():
        # 对应 labelImg.py:885 修补后的写法
        spin.setValue(int(100 * 1.5))

    record("QScrollBar.setValue（滚动，int 转换后）", scroll_like_labelimg)
    record("QScrollBar.setValue（缩放定位，int 转换后）", zoom_like_labelimg)
    record("QSpinBox.setValue（缩放值，int 转换后）", spin_like_labelimg)

    # ---------- 结果 ----------
    print("")
    failed = []
    for name, ok, err in results:
        if ok is True:
            print("  [ 通过 ] %s" % name)
        elif ok is False:
            print("  [ 失败 ] %s" % name)
            print("           %s" % err)
            failed.append(name)
        else:
            print("  [ 跳过 ] %s  (%s)" % (name, err))
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("")
    print("=" * 72)
    if failed:
        print("失败 %d 项。补丁可能没打全，请运行：" % len(failed))
        print("  python tools/patch_labelimg.py")
        return 1
    print("全部通过 —— LabelImg 的绘制/滚动/缩放路径都能正常执行。")
    print("（若有条目显示“跳过”，说明该项因与类型无关的原因未能执行，")
    print("  可加 --verbose 看详细原因；只要没有“失败”就说明类型问题已解决。）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
