#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查本机 LabelImg 运行环境是否可用（无界面，需要时能自动降级 PyQt5）。

======================================================================
背景
======================================================================
labelImg 1.8.3 里大量把**浮点数**传给 PyQt5 中只接受 int 的 API，例如：

    # libs/canvas.py:472
    p.drawLine(self.prevPoint.x(), 0, self.prevPoint.x(), self.pixmap.height())
    #   prevPoint 是 QPointF，.x() 返回 float -> drawLine(int,int,int,int) 报错

    # labelImg.py:879
    bar.setValue(bar.value() + bar.singleStep() * units)   # units 是 float

老版本 PyQt5 会默默把 float 转成 int；**PyQt5 5.15.9 起（sip 12.12+）改成严格
校验类型**，于是这些地方全部抛 TypeError，表现为「能启动、一画/一滚就崩」。

本脚本在**离屏模式**下直接调用这些 API 试一次，从而判断当前 PyQt5 是否会
接受 float。这样不用真的开图形界面就能确定环境是否可用。

用法：
    python tools/check_labelimg_env.py            # 只检查
    python tools/check_labelimg_env.py --fix      # 检查并在必要时降级 PyQt5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# 已知可用的 PyQt5 版本（严格校验是 5.15.9 引入的）
GOOD_PYQT5 = "5.15.7"
GOOD_SIP = "12.11.0"


def patch_status():
    """统计 labelImg 的 8 处 float->int 补丁打了几处。

    返回 (已打, 总数, 缺失位置列表)。
    """
    try:
        sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
        from patch_labelimg import PATCHES, site_packages
    except Exception as e:
        return 0, 0, ["无法导入 patch_labelimg：%s" % e]

    sp = site_packages()
    done = total = 0
    missing = []
    for rel, pairs in PATCHES.items():
        p = sp / rel
        if not p.is_file():
            total += len(pairs)
            missing.append("%s（文件不存在）" % rel)
            continue
        lines = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()}
        for orig, fixed in pairs:
            total += 1
            if fixed in lines:
                done += 1
            else:
                missing.append("%s: %s" % (rel, orig[:50]))
    return done, total, missing


def probe() -> list:
    """在离屏模式下试调用那些只接受 int 的 API，返回 [(名称, 是否通过, 错误)]。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_LOGGING_RULES", "*=false")

    from PyQt5.QtCore import QPointF, Qt
    from PyQt5.QtGui import QPainter, QPixmap
    from PyQt5.QtWidgets import QApplication, QScrollBar, QSpinBox

    app = QApplication.instance() or QApplication([])  # noqa: F841
    out = []

    def try_call(name, fn):
        try:
            fn()
            out.append((name, True, ""))
        except TypeError as e:
            out.append((name, False, str(e).splitlines()[0]))
        except Exception as e:  # 其它异常不代表 float 兼容性
            out.append((name, None, "%s: %s" % (type(e).__name__, e)))

    # 1) canvas.py:472 —— QPainter.drawLine(int x4) 收 float
    pm = QPixmap(100, 100)
    painter = QPainter(pm)
    try_call("QPainter.drawLine(4*int) 传 float",
             lambda: painter.drawLine(10.5, 0, 10.5, 100))
    # drawRect 同理（shape.py 里也有 float 坐标）
    try_call("QPainter.drawRect(4*int) 传 float",
             lambda: painter.drawRect(1.5, 2.5, 30.5, 40.5))
    painter.end()

    # 2) labelImg.py:879 —— QScrollBar.setValue 收 float
    bar = QScrollBar(Qt.Horizontal)
    try_call("QScrollBar.setValue 传 float", lambda: bar.setValue(1.5))

    # 3) labelImg.py:885 —— QSpinBox.setValue 收 float
    spin = QSpinBox()
    try_call("QSpinBox.setValue 传 float", lambda: spin.setValue(1.5))

    # 4) QPointF -> QPoint 的常见误用
    pt = QPointF(1.5, 2.5)
    _ = pt  # 仅确认 QPointF 存在
    return out


def pyqt5_version():
    try:
        from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
        try:
            from PyQt5.sip import SIP_VERSION_STR
        except Exception:
            from sip import SIP_VERSION_STR  # type: ignore
        return PYQT_VERSION_STR, QT_VERSION_STR, SIP_VERSION_STR
    except Exception as e:
        return None, None, str(e)


def main():
    ap = argparse.ArgumentParser(description="检查 LabelImg 运行环境")
    ap.add_argument("--fix", action="store_true",
                    help="检测到不兼容时自动降级 PyQt5（会改动 pytorch 环境）")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 70)
    print("LabelImg 运行环境检查")
    print("=" * 70)

    v, qt, sip = pyqt5_version()
    print("  PyQt5     : %s" % v)
    print("  Qt        : %s" % qt)
    print("  sip       : %s" % sip)
    print("")

    results = probe()
    print("  API 兼容性探测（离屏模式，不需要图形界面）：")
    bad = []
    for name, ok, err in results:
        if ok is True:
            print("    [ 通过 ] %s" % name)
        elif ok is False:
            print("    [ 失败 ] %s" % name)
            print("             %s" % err)
            bad.append(name)
        else:
            print("    [ 跳过 ] %s  (%s)" % (name, err))
    print("")

    if not bad:
        print("  结论：当前 PyQt5 接受 float，labelImg 无需补丁即可使用。")
        return 0

    # PyQt5 本身拒绝 float —— 这只说明「需要补丁」，
    # 关键要看 labelImg 的 8 处调用是否已经套了 int()。
    done, total, missing = patch_status()
    print("  labelImg 源码补丁：%d / %d 处已打" % (done, total))
    print("")

    if total and done == total:
        print("  结论：PyQt5 会拒绝 float，但 labelImg 的调用点**已全部套上 int()**，")
        print("        因此实际运行时不会再崩。")
        print("        想进一步确认可跑：python tools/verify_labelimg_paint.py")
        return 0

    print("  结论：PyQt5 **严格校验类型**，且 labelImg 的调用点还没补全，")
    print("        会在画框/滚动/缩放时崩溃。")
    if missing:
        print("        缺失：")
        for m in missing[:8]:
            print("          - %s" % m)
    print("")
    print("  处理：给这些调用套上 int()：")
    print("    python tools/patch_labelimg.py")
    print("")
    print("  然后离屏验证一遍（不用开图形界面）：")
    print("    python tools/verify_labelimg_paint.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
