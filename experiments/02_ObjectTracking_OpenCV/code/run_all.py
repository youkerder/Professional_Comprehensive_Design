# -*- coding: utf-8 -*-
"""实验三 · 一键复现（环境自检 → 造视频 → 单目标 demo → 多目标对比 → 评价 → 图表）。

用法
----
  python run_all.py              # 全流程（含 MIL/BOOSTING/TLD，约 6~8 分钟）
  python run_all.py --fast       # 只跑 KCF/CSRT/MEDIANFLOW（约 1 分钟）
  python run_all.py --regen      # 强制重造合成视频

每一步都打印命令与耗时；任一步非零退出立即停止并给出原因。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from common import CODE_DIR, DATA_DIR, MAIN_TRACKERS, EXTRA_TRACKERS, VIDEO_DIR, setup_console

PY = sys.executable


def run(desc, script, *args):
    cmd = [PY, str(CODE_DIR / script), *map(str, args)]
    print(f"\n=== {desc}\n$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(CODE_DIR))
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"[FAIL] {desc} 退出码 {p.returncode}（耗时 {dt:.1f}s）")
    print(f"--- {desc} 完成，{dt:.1f}s")


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    print("=" * 70)
    print("实验三 · 基于 OpenCV 的多目标跟踪对比实验")
    print("=" * 70)

    # 0. 环境自检
    import cv2
    print(f"[env] python {sys.version.split()[0]}")
    print(f"[env] opencv {cv2.__version__}  legacy={hasattr(cv2, 'legacy')}")
    if not hasattr(cv2, "legacy"):
        raise SystemExit("[FAIL] 没有 cv2.legacy，请装 opencv-contrib-python 4.5.2+")
    import trackers
    print(trackers.describe())

    # 1. 数据
    need = [s for s in ("cross", "occlude") if not (VIDEO_DIR / f"{s}.mp4").exists()]
    if need or args.regen:
        run("生成合成多目标视频", "make_synthetic_video.py")
    else:
        print(f"[data] 已存在：{VIDEO_DIR / 'cross.mp4'}、{VIDEO_DIR / 'occlude.mp4'}")

    # 2. 单目标 demo（题目要求的 selectROI 流程，这里用真值首帧框保证可复现）
    for algo in MAIN_TRACKERS:
        run(f"单目标 demo {algo}", "single_target_demo.py",
            "--seq", "cross", "--algo", algo, "--target", "target1")

    # 3. 多目标：主算法（前端 MultiTracker）
    run("多目标跟踪：主算法", "run_multitracker.py",
        "--seq", "cross,occlude", "--algos", ",".join(MAIN_TRACKERS),
        *(["--frames", args.frames] if args.frames else []))

    # 4. 多目标：补充算法（慢，用于效率/精度基线）
    if not args.fast:
        run("多目标跟踪：补充算法", "run_multitracker.py",
            "--seq", "cross,occlude", "--algos", ",".join(EXTRA_TRACKERS),
            *(["--frames", args.frames] if args.frames else []))

    # 5. 一个 MultiTracker 内混用三种算法（演示用）
    run("多目标跟踪：混用三种算法", "run_multitracker.py",
        "--seq", "cross", "--mixed", "--algos", ",".join(MAIN_TRACKERS))

    # 6. 指标
    run("OTB / MOT / VOT 指标评价", "evaluate.py")

    # 7. 图表
    run("生成图表与表格", "make_figures.py")

    print("\n" + "=" * 70)
    print("全部完成。产物：")
    print("  outputs/videos/   跟踪结果视频")
    print("  outputs/traces/   逐帧轨迹 CSV")
    print("  outputs/metrics/  指标 JSON + summary.csv")
    print("  outputs/figures/  图；reports/figures/、reports/tables/ 已同步一份")
    print("=" * 70)


if __name__ == "__main__":
    main()
