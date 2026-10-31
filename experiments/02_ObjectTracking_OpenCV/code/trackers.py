# -*- coding: utf-8 -*-
"""实验三 · 跟踪器工厂：抹平 OpenCV 3.x / 4.x / 5.x 的命名差异。

要点
----
* OpenCV ≤ 4.5.1：`cv2.TrackerKCF_create()`、`cv2.MultiTracker_create()`
* OpenCV ≥ 4.5.2：跟踪器迁到 `cv2.legacy`，即
  `cv2.legacy.TrackerKCF_create()`、`cv2.legacy.MultiTracker_create()`
* 只装了 `opencv-python`（非 contrib）时，以上全都没有 → 直接给出明确报错

本模块对外只暴露 `create_tracker(name)` 与 `create_multi_tracker()`，
上层的实验代码不关心底层是哪个命名空间。
"""
from __future__ import annotations

import cv2

# 算法名 -> (中文说明, 候选构造函数名, 适用维度)
TRACKER_INFO = {
    "KCF": ("核相关滤波", "TrackerKCF_create", "精度/速度均衡"),
    "MOSSE": ("最小平方误差滤波", "TrackerMOSSE_create", "速度"),
    "CSRT": ("通道与空间可靠性", "TrackerCSRT_create", "精度/抗形变"),
    "MEDIANFLOW": ("中值光流", "TrackerMedianFlow_create", "失败检测"),
    "MIL": ("多示例学习", "TrackerMIL_create", "基线"),
    "BOOSTING": ("在线 AdaBoost", "TrackerBoosting_create", "基线"),
    "TLD": ("跟踪-学习-检测", "TrackerTLD_create", "长期遮挡"),
}


def _lookup(fn_name):
    """在 cv2.legacy 和 cv2 两个命名空间里找构造函数。"""
    for ns_name in ("legacy", "top"):
        ns = getattr(cv2, "legacy", None) if ns_name == "legacy" else cv2
        if ns is None:
            continue
        fn = getattr(ns, fn_name, None)
        if callable(fn):
            return fn
    return None


def create_tracker(name):
    name = name.upper()
    if name not in TRACKER_INFO:
        raise KeyError(f"未知跟踪器 {name}，可选：{', '.join(TRACKER_INFO)}")
    fn = _lookup(TRACKER_INFO[name][1])
    if fn is None:
        raise RuntimeError(
            f"当前 OpenCV {cv2.__version__} 中找不到 {TRACKER_INFO[name][1]}。\n"
            f"  * 需要 opencv-contrib-python（不是 opencv-python）\n"
            f"  * OpenCV ≥ 4.5.2 请用 cv2.legacy.*，本模块已自动兼容"
        )
    return fn()


def create_multi_tracker():
    """题目要求的 cv2.MultiTracker_create()，按版本自动落到 cv2.legacy。"""
    mt = getattr(getattr(cv2, "legacy", None), "MultiTracker_create", None)
    if callable(mt):
        return mt()
    mt = getattr(cv2, "MultiTracker_create", None)
    if callable(mt):
        return mt()
    raise RuntimeError(
        "找不到 MultiTracker_create：请安装 opencv-contrib-python（新版用 cv2.legacy.MultiTracker_create）"
    )


def available_trackers():
    """实际能构造出来的跟踪器列表（用于命令行校验与报告）。"""
    ok = []
    for name in TRACKER_INFO:
        try:
            create_tracker(name)
            ok.append(name)
        except Exception:
            pass
    return ok


def describe():
    lines = [f"OpenCV {cv2.__version__}  (legacy 命名空间：{hasattr(cv2, 'legacy')})"]
    for name, (cn, fn, role) in TRACKER_INFO.items():
        lines.append(f"  {name:11s} {cn:12s} {role:8s} {'OK' if _lookup(fn) else '缺失'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
