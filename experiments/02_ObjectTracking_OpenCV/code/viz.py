# -*- coding: utf-8 -*-
"""实验三 · 可视化工具：画框、画轨迹、贴状态面板。

约定：所有框都是 (x, y, w, h)，坐标可以是 float。
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from common import center


def draw_box(frame, box, color, label=None, ok=True, thickness=2):
    x, y, w, h = [int(round(v)) for v in box]
    if ok:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    else:
        # 跟踪失败：画虚线框（用短线段拼）+ 叉号，一眼能看出丢失
        _dashed_rect(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.line(frame, (x, y), (x + w, y + h), color, 1)
        cv2.line(frame, (x + w, y), (x, y + h), color, 1)
    if label:
        ty = y - 6 if y - 6 > 12 else y + h + 16
        cv2.putText(frame, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)


def _dashed_rect(frame, p1, p2, color, thickness=2, dash=6):
    x1, y1 = p1
    x2, y2 = p2
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x1, y2, x2, y2),
                             (x1, y1, x1, y2), (x2, y1, x2, y2)):
        L = max(abs(bx - ax), abs(by - ay))
        if L == 0:
            continue
        for s in range(0, int(L), dash * 2):
            t0, t1 = s / L, min(1.0, (s + dash) / L)
            p = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
            q = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
            cv2.line(frame, p, q, color, thickness)


def draw_trail(frame, points, color, maxlen=60, thickness=2):
    """把中心点连成轨迹，越旧越淡（BGR 颜色按距离线性衰减）。"""
    pts = list(points)[-maxlen:]
    n = len(pts)
    for i in range(1, n):
        a = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        b = (int(pts[i][0]), int(pts[i][1]))
        if a == b:
            continue
        k = i / max(1, n - 1)          # 0=旧, 1=新
        col = tuple(int(c * (0.35 + 0.65 * k)) for c in color)
        cv2.line(frame, a, b, col, thickness + (1 if k > 0.75 else 0), cv2.LINE_AA)


def draw_history(frame, trail, color, maxlen=60):
    """轨迹点存在 deque 里，取最近 maxlen 个点画线。"""
    draw_trail(frame, list(trail)[-maxlen:], color, maxlen=maxlen)


def panel(frame, lines, origin=(8, 8), font_scale=0.5, line_h=20):
    """左上角半透明黑底 + 多行文字（只支持 ASCII，OpenCV 不认中文）。"""
    if not lines:
        return frame
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for t in lines) + 16
    height = line_h * len(lines) + 10
    x0, y0 = origin
    x1, y1 = min(frame.shape[1], x0 + width), min(frame.shape[0], y0 + height)
    roi = frame[y0:y1, x0:x1]
    if roi.size:
        roi[:] = (roi * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (x0 + 8, y0 + 18 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def new_trail(maxlen=600):
    return deque(maxlen=maxlen)


def get_center(box):
    return center(box)
