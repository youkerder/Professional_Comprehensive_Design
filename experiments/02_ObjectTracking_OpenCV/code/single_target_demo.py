# -*- coding: utf-8 -*-
"""实验三 · 单目标跟踪 demo（对应“执行步骤 2”）。

流程与题目要求一致：
  1. 读取视频；
  2. **第一帧用 cv2.selectROI 手工框选目标**（也可以用 --target 直接取真值框，方便批量复现）；
  3. 用 KCF / MOSSE / CSRT / MEDIANFLOW / MIL / BOOSTING / TLD 之一逐帧 update；
  4. 画框 + 拖尾轨迹，写出视频与 CSV 轨迹；
  5. 打印每帧耗时、FPS、成功帧率、与真值的平均 IoU / 中心误差。

用法
----
  # 交互式：弹出窗口，鼠标框选（本机有桌面时才可用）
  python single_target_demo.py --seq cross --algo KCF --select

  # 非交互：直接取第一帧真值框（批量复现用）
  python single_target_demo.py --seq cross --algo KCF --target target1

  # 自己给框
  python single_target_demo.py --seq occlude --algo CSRT --roi 20,120,40,40
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

import trackers
import viz
from common import (FIG_DIR, MAIN_TRACKERS, TRACE_DIR, VIDEO_DIR, VIDEO_OUT_DIR,
                    center_distance, ensure_dir, iou, load_frames, make_writer,
                    save_trace, setup_console)


def pick_roi(frame, args):
    """返回 (x, y, w, h)。优先级：--roi > --select > --target(真值) > 报错。"""
    if args.roi:
        vals = [int(float(v)) for v in args.roi.replace(",", " ").split()]
        if len(vals) != 4:
            raise ValueError("--roi 需要 4 个整数：x,y,w,h")
        return tuple(vals)
    if args.select:
        print("[提示] 在弹出的窗口中用鼠标框选目标，按 空格/回车 确认，按 c 取消。")
        box = cv2.selectROI("select ROI", frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("select ROI")
        if box == (0, 0, 0, 0):
            raise RuntimeError("没有选中目标")
        return tuple(int(v) for v in box)
    # 退化为真值框（批量化复现）
    from common import initial_boxes
    boxes = initial_boxes(args.seq)
    name = args.target or sorted(boxes)[0]
    if name not in boxes:
        raise KeyError(f"{args.seq} 没有目标 {name}，可选：{sorted(boxes)}")
    print(f"[提示] 未指定 --roi/--select，使用真值首帧框 {name}={boxes[name]}")
    return tuple(float(v) for v in boxes[name])


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="单目标跟踪 demo")
    ap.add_argument("--seq", default="cross", help="序列名（cross / occlude）")
    ap.add_argument("--algo", default="KCF", help="跟踪算法：" + "/".join(trackers.TRACKER_INFO))
    ap.add_argument("--target", default=None, help="用哪个目标的真值框做初值")
    ap.add_argument("--roi", default=None, help="直接给初值框 x,y,w,h")
    ap.add_argument("--select", action="store_true", help="弹窗手工 selectROI")
    ap.add_argument("--frames", type=int, default=0, help="只跑前 N 帧，0=全片")
    ap.add_argument("--trail", type=int, default=50, help="拖尾长度（帧）")
    ap.add_argument("--show", action="store_true", help="实时显示 + 按 q 退出")
    ap.add_argument("--no-write", action="store_true", help="不写视频文件")
    args = ap.parse_args()

    video = VIDEO_DIR / f"{args.seq}.mp4"
    frames = load_frames(video)
    if args.frames:
        frames = frames[:args.frames]
    h, w = frames[0].shape[:2]

    roi = pick_roi(frames[0], args)
    tracker = trackers.create_tracker(args.algo.upper())
    t0 = time.perf_counter()
    ok = tracker.init(frames[0], tuple(int(v) for v in roi))
    init_ms = (time.perf_counter() - t0) * 1000
    if not ok:
        raise RuntimeError(f"{args.algo} 在第 1 帧初始化失败：请检查框内是否有足够纹理（{roi}）")
    print(f"[init] {args.algo} 初始化成功 roi={tuple(int(v) for v in roi)} 用时 {init_ms:.2f} ms")

    out_path = VIDEO_OUT_DIR / f"single_{args.seq}_{args.algo}.mp4"
    writer = None if args.no_write else make_writer(out_path, 25.0, (w, h))
    rows = []
    trail = viz.new_trail(1000)
    boxes = {0: tuple(float(v) for v in roi)}
    update_ms = []
    ok_count = 0

    from common import load_otb_gt, ANN_DIR
    gt_name = args.target or "target1"
    gt_path = ANN_DIR / args.seq / "otb" / f"{gt_name}.txt"
    gt = load_otb_gt(gt_path) if gt_path.exists() else {}

    for i in range(len(frames)):
        frame = frames[i]
        if i > 0:
            t1 = time.perf_counter()
            ok, box = tracker.update(frame)
            dt = (time.perf_counter() - t1) * 1000
            update_ms.append(dt)
            box = tuple(float(v) for v in box)
            boxes[i] = box
        else:
            dt = init_ms
            box = boxes[0]
        ok_count += int(bool(ok))
        trail.append(viz.get_center(box))

        iou_v = iou(box, gt[i]) if i in gt else float("nan")
        cdist = center_distance(box, gt[i]) if i in gt else float("nan")
        rows.append({"frame": i, "target": gt_name, "x": box[0], "y": box[1],
                     "w": box[2], "h": box[3], "ok": int(bool(ok)),
                     "ms": round(dt, 3), "cx": round(box[0] + box[2] / 2, 2),
                     "cy": round(box[1] + box[3] / 2, 2)})

        vis = frame.copy()
        viz.draw_history(vis, [viz.get_center(boxes[k]) for k in sorted(boxes)
                               if k >= max(0, i - args.trail)], (60, 220, 240))
        viz.draw_box(vis, box, (60, 60, 230), f"{args.algo} T1 {'OK' if ok else 'LOST'}",
                     ok=bool(ok))
        viz.panel(vis, [
            f"{args.algo}  frame {i}/{len(frames) - 1}",
            f"update {dt:.2f} ms   ~{1000.0 / max(dt, 1e-6):.0f} FPS",
            f"IoU {iou_v:.3f}" if i in gt else "IoU n/a",
            f"center err {cdist:.1f} px" if i in gt else "",
        ])
        if writer is not None:
            writer.write(vis)
        if args.show:
            cv2.imshow("single target tracking", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        if args.no_write and not args.show and i % 30 == 0:
            cv2.imwrite(str(ensure_dir(FIG_DIR) / f"single_{args.seq}_{args.algo}_f{i:03d}.png"), vis)

    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    save_trace(TRACE_DIR / f"single_{args.seq}_{args.algo}.csv", rows)
    n = len(update_ms)
    mean_ms = float(np.mean(update_ms)) if n else float("nan")
    mean_iou = float(np.nanmean([iou(boxes[k], gt[k]) for k in gt if k in boxes])) if gt else float("nan")
    mce = float(np.nanmean([center_distance(boxes[k], gt[k]) for k in gt if k in boxes])) if gt else float("nan")
    print(f"[result] {args.algo} on {args.seq}")
    print(f"  init {init_ms:.2f} ms | update 平均 {mean_ms:.2f} ms "
          f"({1000.0 / mean_ms:.1f} FPS) | 最快 {min(update_ms):.2f} 最慢 {max(update_ms):.2f} ms")
    print(f"  跟踪成功帧 {ok_count}/{len(frames)} = {ok_count / len(frames) * 100:.1f}%")
    print(f"  平均 IoU {mean_iou:.4f} | 平均中心误差 {mce:.2f} px")
    if not args.no_write:
        print(f"  视频已写出：{out_path}")
    print(f"  轨迹 CSV：{TRACE_DIR / f'single_{args.seq}_{args.algo}.csv'}")


if __name__ == "__main__":
    main()
