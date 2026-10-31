# -*- coding: utf-8 -*-
"""实验三 · 合成多目标跟踪视频（带逐帧真值）。

为什么要合成
------------
公开数据集体积大、且本机暂时没有跟踪视频；合成序列可以拿到**逐帧精确真值**，
OTB / MOT / VOT 三类指标都能严格计算，还能精确控制三类难点：

  * cross   ：三个目标交叉运动（目标间互相遮挡）
  * occlude ：目标钻到静态遮挡条后面，一个**完全消失 28 帧**、一个只剩**一半可见**，
              第三个不受影响（三档可见性，用于遮挡鲁棒性对比）

画面：640x360、25 fps、300 帧（12 s），背景有纹理与噪声，
目标本身带高对比纹理（否则 MOSSE / MEDIANFLOW 这类依赖灰度纹理的算法
在纯色块上会直接失败，测出来的结论是假的）。

生成文件
--------
  datasets/tracking_videos/videos/<seq>.mp4
  datasets/tracking_videos/annotations/<seq>/otb/<目标>.txt   # OTB: x,y,w,h
  datasets/tracking_videos/annotations/<seq>/gt.txt           # MOT: frame,id,x,y,w,h,conf,cls,vis
  datasets/tracking_videos/annotations/<seq>/seqinfo.ini
  datasets/tracking_videos/annotations/<seq>/first_boxes.json  # 首帧框
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from common import ANN_DIR, FIG_DIR, PALETTE, VIDEO_DIR, ensure_dir, setup_console

W, H = 640, 360
FPS = 25
N_FRAMES = 300


# ---------------------------------------------------------------------------
# 目标外观（带纹理，保证 KCF/MOSSE/CSRT/MedianFlow 都能初始化）
# ---------------------------------------------------------------------------
def _textured_sprite(radius, color, seed, kind="disc"):
    size = 2 * radius + 1
    img = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), np.uint8)
    if kind == "disc":
        cv2.circle(img, (radius, radius), radius, color, -1)
        cv2.circle(mask, (radius, radius), radius, 255, -1)
    else:
        cv2.rectangle(img, (0, 0), (size - 1, size - 1), color, -1)
        cv2.rectangle(mask, (0, 0), (size - 1, size - 1), 255, -1)

    rng = np.random.default_rng(seed)
    light = tuple(int(min(255, c + 80)) for c in color)
    dark = tuple(int(max(0, c - 80)) for c in color)
    # 内部画随机小方块 / 圆，制造角点纹理
    for i in range(28):
        cx, cy = rng.integers(4, size - 4, 2)
        r = int(rng.integers(2, max(3, radius // 3)))
        col = light if i % 2 == 0 else dark
        if i % 3 == 0:
            cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), col, -1)
        else:
            cv2.circle(img, (cx, cy), r, col, -1)
    # 白色高光边，便于目视辨认
    if kind == "disc":
        cv2.circle(img, (radius, radius), radius - 1, (245, 245, 245), 2)
    else:
        cv2.rectangle(img, (1, 1), (size - 2, size - 2), (245, 245, 245), 2)

    img[mask == 0] = 0
    return img, mask


def _make_background(rng):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.zeros((H, W, 3), np.float32)
    base[..., 0] = 70 + 25 * (yy / H)          # B
    base[..., 1] = 78 + 20 * (xx / W)          # G
    base[..., 2] = 72 + 15 * ((xx + yy) / (W + H))  # R
    # 低频噪声（放大后高斯模糊）→ 墙面/地面质感
    low = rng.normal(0, 18, (H // 16, W // 16)).astype(np.float32)
    low = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
    base += low[..., None] * 0.9
    bg = np.clip(base, 0, 255).astype(np.uint8)

    # 静态纹理：若干灰色矩形/圆 + 细网格线
    for _ in range(9):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        w, h = int(rng.integers(30, 110)), int(rng.integers(20, 80))
        col = int(rng.integers(45, 110))
        cv2.rectangle(bg, (x, y), (x + w, y + h), (col, col, col), -1)
        cv2.rectangle(bg, (x, y), (x + w, y + h), (200, 200, 200), 1)
    for x in range(0, W, 40):
        cv2.line(bg, (x, 0), (x, H), (95, 95, 95), 1)
    for y in range(0, H, 40):
        cv2.line(bg, (0, y), (W, y), (95, 95, 95), 1)
    return bg


# ---------------------------------------------------------------------------
# 两条序列的运动学定义
# ---------------------------------------------------------------------------
def path_cross(t):
    """t ∈ [0,1] → ({target: (cx, cy, half_size)}, None)"""
    objs = {
        # 左上 → 右下
        "target1": (60 + 520 * t, 80 + 190 * t + 18 * math.sin(2 * math.pi * 2 * t), 20),
        # 右下 → 左上，与 target1 在中点交叉
        "target2": (580 - 520 * t, 280 - 190 * t + 18 * math.cos(2 * math.pi * 2 * t), 22),
        # 绕中心做小圆运动，逐渐变大（尺度变化）
        "target3": (320 + 130 * math.sin(2 * math.pi * t),
                    170 + 55 * math.cos(2 * math.pi * 2 * t),
                    20 + 10 * t),
    }
    return objs, None


OCCLUDER = (280, 0, 90, 250)   # x, y, w, h 的静态遮挡条（draw 在目标之上）


def path_occlude(t):
    objs = {
        "target1": (40 + 560 * t, 140, 20),   # 完全进入遮挡条 → 完全不可见约 28 帧
        "target2": (600 - 560 * t, 250, 22),  # 只有下半部露出 → 约 50% 可见
        "target3": (110 + 420 * (0.5 - 0.5 * math.cos(2 * math.pi * t)), 320, 20),  # 不受遮挡（对照）
    }
    return objs, OCCLUDER


SEQS = {
    "cross": (path_cross, "三目标交叉：目标间互相遮挡 + 尺度变化"),
    "occlude": (path_occlude, "静态遮挡条（280,0,90,250）：target1 完全消失 28 帧，target2 约 50% 可见 100 帧，target3 不受影响"),
}


# ---------------------------------------------------------------------------
# 主渲染
# ---------------------------------------------------------------------------
def render(seq, out_dir=None, n_frames=N_FRAMES, preview=True):
    func, desc = SEQS[seq]
    rng = np.random.default_rng(20241118)
    bg0 = _make_background(rng)

    target_colors = {"target1": PALETTE[0], "target2": PALETTE[1], "target3": PALETTE[2]}
    sprites = {k: _textured_sprite(40, c, seed=100 + i, kind="disc" if i < 2 else "square")
               for i, (k, c) in enumerate(target_colors.items())}

    videos_dir = ensure_dir(VIDEO_DIR)
    ann_dir = ensure_dir(ANN_DIR / seq)
    ensure_dir(ann_dir / "otb")
    video_path = Path(videos_dir) / f"{seq}.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter 打开失败")

    gt_rows = []          # MOT 行
    per_target = {k: [] for k in target_colors}
    first_boxes = {}
    occ_frac = {k: [] for k in target_colors}

    for i in range(n_frames):
        t = i / (n_frames - 1)
        objs, occ = func(t)
        frame = bg0.copy()
        # 轻微亮度/噪声波动（模拟真实相机）
        alpha = 1.0 + 0.015 * math.sin(i * 0.21)
        frame = np.clip(frame.astype(np.float32) * alpha + rng.normal(0, 1.5, frame.shape),
                        0, 255).astype(np.uint8)

        vismask = np.full((H, W), 255, np.uint8)
        for name in ("target3", "target2", "target1"):   # 固定绘制顺序
            cx, cy, hs = objs[name]
            sprite = sprites[name]
            size = int(round(hs * 2))
            patch = cv2.resize(sprite[0], (size, size), interpolation=cv2.INTER_AREA)
            pmask = cv2.resize(sprite[1], (size, size), interpolation=cv2.INTER_NEAREST)
            x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
            x1, y1 = x0 + size, y0 + size
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(W, x1), min(H, y1)
            sx0, sy0 = x0c - x0, y0c - y0
            if x1c > x0c and y1c > y0c:
                roim = frame[y0c:y1c, x0c:x1c]
                m = pmask[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
                patch_roi = patch[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
                roim[m > 0] = patch_roi[m > 0]
                vismask[y0c:y1c, x0c:x1c][m > 0] = 255
            box = (x0, y0, size, size)
            per_target[name].append(box)
        if occ:
            ox, oy, ow, oh = occ
            # 遮挡条画在目标之后 → 目标被完全盖住
            bar = np.full((oh, ow, 3), 58, np.uint8)
            bar[::7, :] = 78
            bar[:, ::9] = 72
            frame[oy:oy + oh, ox:ox + ow] = bar
            vismask[oy:oy + oh, ox:ox + ow] = 0

        # 计算每个目标的可见比例（用于 MOT 的 visibility 列）
        for name in target_colors:
            x, y, w, h = per_target[name][-1]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 <= x0 or y1 <= y0:
                vis = 0.0
            else:
                m = vismask[y0:y1, x0:x1]
                vis = float(np.count_nonzero(m)) / float(w * h)
            occ_frac[name].append(vis)

        if i == 0:
            for name in target_colors:
                first_boxes[name] = [int(v) for v in per_target[name][0]]
        writer.write(frame)
        if i == 0:
            cv2.imwrite(str(ensure_dir(FIG_DIR) / f"seq_{seq}_first_frame.png"), frame)

    writer.release()

    # ---- 写真值 ----
    for name in target_colors:
        lines = [f"{x},{y},{w},{h}" for (x, y, w, h) in per_target[name]]
        (ann_dir / "otb" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(ann_dir / "gt.txt", "w", encoding="utf-8") as f:
        for i in range(n_frames):
            for tid, name in enumerate(target_colors, start=1):
                x, y, w, h = per_target[name][i]
                f.write(f"{i + 1},{tid},{x},{y},{w},{h},1,1,{occ_frac[name][i]:.3f}\n")

    (ann_dir / "first_boxes.json").write_text(
        json.dumps(first_boxes, ensure_ascii=False, indent=2), encoding="utf-8")
    (ann_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={seq}\nimDir=../videos\nimExt=.mp4\n"
        f"seqLength={n_frames}\nframeRate={FPS}\n"
        f"imWidth={W}\nimHeight={H}\nnote={desc}\n", encoding="utf-8")
    (ann_dir / "meta.json").write_text(json.dumps({
        "seq": seq, "desc": desc, "size": [W, H], "fps": FPS, "frames": n_frames,
        "occluder": list(OCCLUDER) if occ else None,
        "occluded_frames": {k: int(sum(1 for v in occ_frac[k] if v < 0.05)) for k in target_colors},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {seq}: {video_path}  {n_frames} 帧 {W}x{H}@{FPS}fps")
    print(f"     完全遮挡帧数: " + ", ".join(
        f"{k}={sum(1 for v in occ_frac[k] if v < 0.05)}" for k in target_colors))
    if preview:
        _preview(seq, per_target, target_colors, n_frames)
    return video_path, per_target


def _preview(seq, per_target, colors, n_frames):
    """生成一张三联图：第 1 / 中间 / 最后 帧，画真值框。"""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{seq}.mp4"))
    picks = [0, n_frames // 2, n_frames - 1]
    tiles = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx > picks[-1]:
            break
        if idx in picks:
            f = frame.copy()
            for name, col in colors.items():
                x, y, w, h = per_target[name][idx]
                cv2.rectangle(f, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)
                cv2.putText(f, name.replace("target", "T"), (int(x), int(y) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            cv2.putText(f, f"frame {idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(f)
        idx += 1
    cap.release()
    if tiles:
        out = np.hstack(tiles)
        cv2.imwrite(str(ensure_dir(FIG_DIR) / f"seq_{seq}_gt_preview.png"), out)


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="生成合成多目标跟踪视频与真值")
    ap.add_argument("--seq", default="cross,occlude", help="cross / occlude / all")
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    args = ap.parse_args()
    seqs = ["cross", "occlude"] if args.seq in ("all", "cross,occlude") else args.seq.split(",")
    for s in seqs:
        render(s.strip(), n_frames=args.frames)


if __name__ == "__main__":
    main()
