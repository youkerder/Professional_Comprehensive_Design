# -*- coding: utf-8 -*-
"""实验三 · 目标跟踪（OpenCV）公共工具。

包含：
  * 路径常量（工作区 / 数据 / 输出）
  * 视频读写
  * 真值（OTB 每目标 txt、MOT gt.txt）读写
  * 预测轨迹 CSV 读写
  * IoU / 中心距 / 轨迹平滑等数值工具
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
EXP_DIR = CODE_DIR.parent                       # experiments/02_ObjectTracking_OpenCV
ROOT = EXP_DIR.parents[1]                       # 工作区根目录
DATA_DIR = ROOT / "datasets" / "tracking_videos"
VIDEO_DIR = DATA_DIR / "videos"
ANN_DIR = DATA_DIR / "annotations"
OUT_DIR = EXP_DIR / "outputs"
TRACE_DIR = OUT_DIR / "traces"
VIDEO_OUT_DIR = OUT_DIR / "videos"
METRIC_DIR = OUT_DIR / "metrics"
FIG_DIR = OUT_DIR / "figures"                   # 实验中产生的图
REPORT_FIG_DIR = ROOT / "reports" / "figures"   # 报告引用的图（拷贝一份）
REPORT_TAB_DIR = ROOT / "reports" / "tables"

# BGR 调色板：每个目标一种颜色（与报告插图一致）
PALETTE = [
    (60, 60, 230),    # 红
    (230, 120, 40),   # 蓝
    (60, 190, 60),    # 绿
    (30, 200, 240),   # 黄
    (220, 60, 220),   # 紫
    (200, 200, 40),   # 青
]

# 主实验的四种算法（题目要求 ≥3 种；这几种在速度/精度/鲁棒性上各有代表）
MAIN_TRACKERS = ["KCF", "CSRT", "MEDIANFLOW"]
# 用于“效率对比”补充跑分的基准算法（较慢，仅做 FPS/精度参考）
EXTRA_TRACKERS = ["MIL", "BOOSTING", "TLD"]

# 注意：MOSSE 在 opencv-contrib-python 4.8.1.78 里不可用（update 因 PSR 门限直接返回
# False 并把框置零），详见 EXPERIMENT_LOG.md 与 code/tools/check_trackers.py 的复现输出。
# 代码仍保留 MOSSE 支持，换到能用的 OpenCV 版本后无需改代码。
BROKEN_IN_THIS_BUILD = {"MOSSE": "update() 恒返回 False，(x,y,w,h) 被置零"}


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_console():
    """Windows 控制台默认 GBK，统一改成 UTF-8 输出，避免中文报错。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 视频
# ----------------------------------------------------------------------------
def load_frames(video_path):
    """把视频整段读成 ndarray 列表。合成视频 300 帧 640x360 约 200MB，可接受。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"视频没有可读帧：{video_path}")
    return frames


def video_meta(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def make_writer(path, fps, size, fourcc="mp4v"):
    ensure_dir(Path(path).parent)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter 打开失败：{path}")
    return vw


# ----------------------------------------------------------------------------
# 真值 IO
# ----------------------------------------------------------------------------
def load_otb_gt(path):
    """OTB 格式：每行 `x,y,w,h`，行号即帧号（从 0 开始），NaN 表示该帧不存在。"""
    boxes = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace("\t", ",").split(",") if p != ""]
            if len(parts) < 4:
                continue
            try:
                vals = [float(v) for v in parts[:4]]
            except ValueError:
                continue
            if any(np.isnan(vals)):
                continue
            boxes[i] = tuple(vals)
    return boxes


def load_mot_gt(path):
    """MOTChallenge gt.txt：`frame,id,x,y,w,h,conf,class,visibility` -> {frame: {id: box}}"""
    seq = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split(",")
            if len(p) < 6:
                continue
            frame, tid = int(float(p[0])), int(float(p[1]))
            box = (float(p[2]), float(p[3]), float(p[4]), float(p[5]))
            if len(p) >= 9:
                try:
                    if float(p[8]) <= 0:
                        continue
                except ValueError:
                    pass
            seq.setdefault(frame - 1, {})[tid] = box   # 统一成 0 基帧号
    return seq


def load_seq_gt(seq):
    """读取某序列的真值：返回 (per_target_otb, mot_per_frame, meta)

    per_target_otb : {target_name: {frame: box}}
    mot_per_frame  : {frame: {target_id: box}}
    """
    seq_dir = ANN_DIR / seq
    target_files = sorted((seq_dir / "otb").glob("*.txt")) if (seq_dir / "otb").is_dir() else []
    per_target = {}
    for tf in target_files:
        per_target[tf.stem] = load_otb_gt(tf)
    mot = load_mot_gt(seq_dir / "gt.txt") if (seq_dir / "gt.txt").exists() else {}
    meta = {}
    meta_path = seq_dir / "seqinfo.ini"
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
    meta["targets"] = target_files and [tf.stem for tf in target_files] or []
    return per_target, mot, meta


def initial_boxes(seq):
    """首帧框：优先用 seqinfo/gt，其次用标注的首帧。返回 {target_name: (x,y,w,h)}"""
    p = ANN_DIR / seq / "first_boxes.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: tuple(float(v) for v in box) for k, box in raw.items()}
    per_target, _, _ = load_seq_gt(seq)
    out = {}
    for name, boxes in per_target.items():
        if boxes:
            out[name] = boxes[min(boxes)]
    return out


# ----------------------------------------------------------------------------
# 预测轨迹 CSV
# ----------------------------------------------------------------------------
TRACE_FIELDS = ["frame", "target", "x", "y", "w", "h", "ok", "ms", "cx", "cy"]


def save_trace(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRACE_FIELDS})


def load_trace(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {
                "frame": int(r["frame"]),
                "target": r["target"],
                "box": (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])),
                "ok": str(r["ok"]).lower() in ("1", "true", "yes"),
                "ms": float(r["ms"]) if r["ms"] else 0.0,
            }
            rows.append(row)
    return rows


def trace_to_gt(rows):
    """把预测行整理成 {frame: {target: box}}"""
    out = {}
    for r in rows:
        out.setdefault(r["frame"], {})[r["target"]] = r["box"]
    return out


# ----------------------------------------------------------------------------
# 数值工具
# ----------------------------------------------------------------------------
def xywh_to_xyxy(box):
    x, y, w, h = box
    return x, y, x + w, y + h


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def center_distance(box_a, box_b):
    ax, ay = center(box_a)
    bx, by = center(box_b)
    return float(np.hypot(ax - bx, ay - by))


def hungarian(cost):
    """Kuhn-Munkres 匈牙利算法（方阵成本最小化），用于帧内匹配与 IDF1 的 ID 分配。

    返回 (row_ind, col_ind) 的配对列表；对非方阵先补齐。
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    n, m = cost.shape
    transposed = n > m
    if transposed:
        cost = cost.T
        n, m = cost.shape
    # 补成方阵
    size = max(n, m)
    padded = np.full((size, size), cost.max() * 10 + 1.0)
    padded[:n, :m] = cost
    # 经典 O(n^3) 实现（带势函数）
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = padded[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] <= n and j <= m:
            pairs.append((p[j] - 1, j - 1))
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return pairs


def fmt_float(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)
