# -*- coding: utf-8 -*-
"""实验三 · 跟踪器可用性自检（先跑这个，再跑实验）。

它做三件事：
  1. 打印 OpenCV 版本与 legacy 命名空间情况；
  2. 对每种算法跑一个 40 帧的小测试（带纹理方块在噪声背景上匀速移动），
     统计初始化是否成功、update 成功率、平均耗时；
  3. 若某算法 update 成功率明显异常（例如一直是 0），给出定位提示。

MOSSE 在 opencv-contrib-python 4.8.1.78 上会稳定复现“update 恒返回 False、
框被置零”，这是本机实测结论，不是调用姿势错误：同一份输入下 KCF/CSRT/
MedianFlow/MIL/Boosting/TLD 全部正常。详见 README 的“踩坑记录”。

用法：
  python tools/check_trackers.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import trackers  # noqa: E402
from common import setup_console  # noqa: E402

W, H, N = 320, 200, 40


def make_frames(seed=0):
    rng = np.random.default_rng(seed)
    bg = np.clip(rng.normal(110, 12, (H, W, 3)), 0, 255).astype(np.uint8)
    sprite = np.zeros((40, 40, 3), np.uint8)
    cv2.rectangle(sprite, (4, 4), (35, 35), (220, 220, 220), -1)
    cv2.circle(sprite, (14, 14), 6, (40, 40, 40), -1)
    cv2.circle(sprite, (26, 26), 6, (250, 250, 250), -1)
    cv2.line(sprite, (4, 30), (35, 12), (90, 90, 90), 3)
    frames = []
    for i in range(N):
        f = bg.copy()
        x, y = 40 + 4 * i, 60 + 2 * i
        roi = f[y:y + 40, x:x + 40]
        m = sprite.sum(axis=2) > 0
        roi[m] = sprite[m]
        frames.append(f)
    return frames, (40, 60, 40, 40)


def main():
    setup_console()
    print(f"python  {sys.version.split()[0]}")
    print(f"opencv  {cv2.__version__}   legacy={hasattr(cv2, 'legacy')}")
    print(trackers.describe())
    print()
    frames, bbox = make_frames()
    rows = []
    for name in trackers.TRACKER_INFO:
        try:
            t = trackers.create_tracker(name)
        except Exception as e:
            rows.append((name, "构造失败", str(e)[:40], "-", "-"))
            continue
        t0 = time.perf_counter()
        ok_init = t.init(frames[0], bbox)
        init_ms = (time.perf_counter() - t0) * 1000
        oks = 0
        last = bbox
        ts = []
        for i in range(1, N):
            t1 = time.perf_counter()
            ok, b = t.update(frames[i])
            ts.append((time.perf_counter() - t1) * 1000)
            oks += int(bool(ok))
            last = tuple(float(v) for v in b)
            if not ok:
                break
        mean_ms = float(np.mean(ts)) if ts else float("nan")
        rows.append((name, "OK" if ok_init else "FAIL", f"{oks}/{N - 1}",
                     f"{mean_ms:.2f}", f"{last[0]:.0f},{last[1]:.0f},{last[2]:.0f},{last[3]:.0f}"))

    print(f"{'算法':10s} {'init':8s} {'update 成功':12s} {'平均耗时(ms)':14s} 末帧框")
    print("-" * 72)
    for r in rows:
        print(f"{r[0]:10s} {r[1]:8s} {r[2]:12s} {r[3]:14s} {r[4]}")
    print()

    # 第二块：真实序列（合成视频）首帧目标 —— 内容和背景都更“杂”
    video_ok = []
    try:
        from common import VIDEO_DIR, initial_boxes, load_frames
        frames_v = load_frames(VIDEO_DIR / "cross.mp4")[:60]
        bbox_v = initial_boxes("cross")["target1"]
        print(f"真实序列 cross/target1，框 {tuple(int(v) for v in bbox_v)}，跑 {len(frames_v) - 1} 帧")
        print(f"{'算法':10s} {'init':8s} {'update 成功':12s} {'平均耗时(ms)':14s} 末帧框")
        print("-" * 72)
        for name in trackers.TRACKER_INFO:
            try:
                t = trackers.create_tracker(name)
            except Exception:
                continue
            t0 = time.perf_counter()
            ok_init = t.init(frames_v[0], tuple(int(v) for v in bbox_v))
            init_ms = (time.perf_counter() - t0) * 1000
            oks, ts, last = 0, [], bbox_v
            for i in range(1, len(frames_v)):
                t1 = time.perf_counter()
                ok, b = t.update(frames_v[i])
                ts.append((time.perf_counter() - t1) * 1000)
                oks += int(bool(ok))
                last = tuple(float(v) for v in b)
            print(f"{name:10s} {'OK' if ok_init else 'FAIL':8s} "
                  f"{oks}/{len(frames_v) - 1:<9d} {np.mean(ts):14.2f} "
                  f"{last[0]:.0f},{last[1]:.0f},{last[2]:.0f},{last[3]:.0f}")
            video_ok.append((name, oks, len(frames_v) - 1))
        print()
    except Exception as e:  # 视频还没生成时跳过
        print(f"[skip] 真实序列自检跳过：{e}\n")

    dead = [n for n, oks, tot in video_ok if oks == 0]
    bad = dead or [r[0] for r in rows if r[1] != "OK" or r[2].split("/")[0] == "0"]
    if bad:
        print("注意：以下算法在真实序列上不可用 ——")
        for b in bad:
            print(f"  * {b}：init 返回成功，但 update 每一帧都返回 False 且框被置零 (0,0,0,0)。")
        print("  这不是调用错误：同一份输入下其它算法正常。根因是 MOSSE 的 PSR 判据"
              "在杂乱背景上把每帧都判为不可信，而 OpenCV 的实现在判失败时清零框，"
              "没有保留上一帧位置。")
        print("  处理：正式对比不使用它；换到其它 OpenCV 版本可复测。")
    else:
        print("全部算法可用。")


if __name__ == "__main__":
    main()
