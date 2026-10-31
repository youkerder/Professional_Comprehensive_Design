# -*- coding: utf-8 -*-
"""实验三 · 多目标跟踪（对应“执行步骤 3、4”）。

两条后端，都保留：
  * ``multi``（默认）：``cv2.legacy.MultiTracker_create()`` + 逐个 ``add()``，
    一次 ``update()`` 推进所有目标 —— 题目明确要求的方式。
    注意：legacy MultiTracker 的 update 只返回一个“全体是否都成功”的布尔值，
    单个目标是否丢失拿不到（这是 OpenCV 4.x 绑定层的真实限制），
    所以逐目标的成败在评价阶段统一用与真值的 IoU 判定。
  * ``single``：每个目标各建一个 tracker 实例自己循环，能拿到逐目标 ``ok``，
    耗时口径与 ``multi`` 一致，用于对照“MultiTracker 封装本身有没有额外开销”。

另外支持 ``--mixed``：仍是**一个** MultiTracker，但三个目标分别塞入
KCF / MOSSE / CSRT —— 直接演示“用至少三种算法同时跟踪多目标”。

用法
----
  python run_multitracker.py --seq cross --algos KCF,MOSSE,CSRT,MEDIANFLOW
  python run_multitracker.py --seq occlude --algos KCF --backend single
  python run_multitracker.py --seq cross --mixed
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

import trackers
import viz
from common import (ANN_DIR, MAIN_TRACKERS, METRIC_DIR, PALETTE, TRACE_DIR, VIDEO_DIR,
                    VIDEO_OUT_DIR, center_distance, ensure_dir, iou, load_frames,
                    load_otb_gt, make_writer, save_trace, setup_console)


# ---------------------------------------------------------------------------
# 后端封装
# ---------------------------------------------------------------------------
def broadcast_algos(algo_names, n):
    """把算法名列表按目标数补齐：只给一个算法名时所有目标都用它。"""
    algo_names = list(algo_names)
    if len(algo_names) == 1:
        return algo_names * n
    if len(algo_names) != n:
        raise ValueError(f"算法数 {len(algo_names)} 与目标数 {n} 不匹配")
    return algo_names


class MultiBackend:
    """题目要求的 cv2.legacy.MultiTracker_create() 路径。"""

    name = "multi"

    def __init__(self, algo_names):
        self.algo_names = list(algo_names)
        self.mt = trackers.create_multi_tracker()
        self.inited = False

    def init(self, frame, boxes):
        for name, box in zip(broadcast_algos(self.algo_names, len(boxes)), boxes):
            self.mt.add(trackers.create_tracker(name), frame, tuple(int(v) for v in box))
        self.inited = True
        return True

    def update(self, frame):
        ok, boxes = self.mt.update(frame)
        return bool(ok), [tuple(float(v) for v in b) for b in boxes]


class SingleBackend:
    """逐目标独立实例：能拿到每个目标的 ok，便于分析“谁先丢”。"""

    name = "single"

    def __init__(self, algo_names):
        self.algo_names = list(algo_names)
        self.ts = []
        self.inited = False

    def init(self, frame, boxes):
        self.ts = [trackers.create_tracker(n)
                   for n in broadcast_algos(self.algo_names, len(boxes))]
        oks = [t.init(frame, tuple(int(v) for v in b)) for t, b in zip(self.ts, boxes)]
        self.inited = True
        return all(oks)

    def update(self, frame):
        oks, boxes = [], []
        for t in self.ts:
            ok, b = t.update(frame)
            oks.append(bool(ok))
            boxes.append(tuple(float(v) for v in b))
        return all(oks), boxes


def build_backend(algo_names, backend):
    return SingleBackend(algo_names) if backend == "single" else MultiBackend(algo_names)


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def run_seq(seq, algos, backend="multi", frames_limit=0, trail=45,
            show=False, write_video=True, tag=None, mixed=False):
    video_path = VIDEO_DIR / f"{seq}.mp4"
    frames = load_frames(video_path)
    if frames_limit:
        frames = frames[:frames_limit]
    h, w = frames[0].shape[:2]

    from common import initial_boxes
    init_boxes = initial_boxes(seq)
    target_names = sorted(init_boxes)
    boxes0 = [init_boxes[t] for t in target_names]

    if mixed:
        algo_names = [a.upper() for a in algos][:len(target_names)]
        while len(algo_names) < len(target_names):
            algo_names.append(algo_names[-1])
        tag = tag or "mixed_" + "-".join(algo_names)
        backend = "multi"
        print(f"[mixed] 一个 MultiTracker 内混用：{dict(zip(target_names, algo_names))}")
    else:
        algo_names = [a.upper() for a in algos]
        tag = tag or "-".join(algo_names)

    be = build_backend(algo_names, backend)
    colored = make_colors(len(target_names))

    # 预热：OpenCV 第一次 init 会做内存/FFT 计划等一次性分配，
    # 不预热的话第一条序列的 init 耗时会虚高一个数量级（实测 1482ms vs 79ms）。
    if len(frames) > 1:
        try:
            warm = build_backend(algo_names, backend)
            warm.init(frames[0], boxes0)
            warm.update(frames[1])
        except Exception:
            pass

    out_path = VIDEO_OUT_DIR / f"mt_{seq}_{tag}.mp4"
    writer = make_writer(out_path, 25.0, (w, h)) if write_video else None

    t_init0 = time.perf_counter()
    be.init(frames[0], boxes0)
    init_ms = (time.perf_counter() - t_init0) * 1000

    rows = []
    trails = [viz.new_trail(2000) for _ in target_names]
    per_frame_ms = []
    per_target_ms = {t: [] for t in target_names}

    for i in range(len(frames)):
        frame = frames[i]
        if i == 0:
            boxes, ok = boxes0, True
            dt = init_ms
        else:
            t0 = time.perf_counter()
            ok, boxes = be.update(frame)
            dt = (time.perf_counter() - t0) * 1000
        per_frame_ms.append(dt)
        if not ok and i > 0:
            pass  # 整体失败：具体是哪个目标由评价阶段用 IoU 判定

        for k, name in enumerate(target_names):
            b = boxes[k]
            if i > 0:
                per_target_ms[name].append(dt / max(1, len(target_names)))
            trails[k].append(viz.get_center(b))
            rows.append({"frame": i, "target": name, "x": b[0], "y": b[1],
                         "w": b[2], "h": b[3], "ok": int(bool(ok)),
                         "ms": round(dt / max(1, len(target_names)), 3),
                         "cx": round(b[0] + b[2] / 2, 2), "cy": round(b[1] + b[3] / 2, 2)})

        vis = frame.copy()
        for k, name in enumerate(target_names):
            viz.draw_history(vis, list(trails[k])[-trail:], colored[k], maxlen=trail)
            algo = algo_names[k] if mixed else algo_names[0]
            viz.draw_box(vis, boxes[k], colored[k], f"{name[-1]}:{algo}", ok=bool(ok))
        fps_now = 1000.0 / max(dt, 1e-6)
        viz.panel(vis, [
            f"{'MIXED' if mixed else '-'.join(algo_names)}  {seq}  frame {i}/{len(frames) - 1}",
            f"update {dt:.2f} ms   ~{fps_now:.0f} FPS",
            f"{'all ok' if ok else 'SOME LOST'}",
        ])
        if writer is not None:
            writer.write(vis)
        if show:
            cv2.imshow(f"{seq} {tag}", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    trace_path = TRACE_DIR / f"mt_{seq}_{tag}.csv"
    save_trace(trace_path, rows)

    # 与真值对比的即时概览（完整评价在 evaluate.py）
    gt = {}
    for name in target_names:
        p = ANN_DIR / seq / "otb" / f"{name}.txt"
        if p.exists():
            gt[name] = load_otb_gt(p)
    traj = {}
    for r in rows:
        traj.setdefault(r["target"], {})[r["frame"]] = (
            r["x"], r["y"], r["w"], r["h"])

    stats = {}
    for name in target_names:
        ious = [iou(traj[name][f], gt[name][f]) for f in gt.get(name, {})
                if f in traj.get(name, {})]
        if ious:
            ious = np.asarray(ious)
            stats[name] = {
                "mean_iou": float(ious.mean()),
                "sr50": float((ious >= 0.5).mean()),
                "sr70": float((ious >= 0.7).mean()),
                "lost": int((ious < 0.1).sum()),
            }
    mean_ms = float(np.mean(per_frame_ms)) if per_frame_ms else float("nan")
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("nan")
    print(f"[done] {seq} / {tag} / backend={be.name}")
    print(f"  init {init_ms:.2f} ms | 每帧平均 {mean_ms:.2f} ms | {fps:.1f} FPS "
          f"(P95 {np.percentile(per_frame_ms, 95):.2f} ms)")
    for name, s in stats.items():
        print(f"  {name}: meanIoU {s['mean_iou']:.4f}  SR@0.5 {s['sr50'] * 100:5.1f}%  "
              f"SR@0.7 {s['sr70'] * 100:5.1f}%  丢失帧 {s['lost']}")
    if write_video:
        print(f"  视频：{out_path}")
    print(f"  CSV ：{trace_path}")

    result = {"seq": seq, "tag": tag, "backend": be.name, "algos": algo_names,
              "frames": len(frames), "fps": fps,
              "init_ms": init_ms, "mean_ms": mean_ms,
              "p95_ms": float(np.percentile(per_frame_ms, 95)),
              "per_target": stats,
              "video": str(out_path) if write_video else None,
              "trace": str(trace_path)}
    (ensure_dir(METRIC_DIR) / f"run_mt_{seq}_{tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


COLORS = [(40, 40, 220), (220, 120, 40), (50, 190, 50), (30, 200, 240), (200, 60, 200)]


def make_colors(n):
    out = []
    for i in range(n):
        base = COLORS[i % len(COLORS)]
        out.append(base)
    return out


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="多目标跟踪（MultiTracker）")
    ap.add_argument("--seq", default="cross,occlude")
    ap.add_argument("--algos", default="KCF,MOSSE,CSRT",
                    help="逗号分隔，可写多个；配合 --mixed 时表示每个目标一个算法")
    ap.add_argument("--backend", default="multi", choices=["multi", "single"])
    ap.add_argument("--mixed", action="store_true", help="一个 MultiTracker 内混用多种算法")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--trail", type=int, default=45)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    seqs = [s.strip() for s in args.seq.split(",") if s.strip()]
    if args.mixed:
        for s in seqs:
            run_seq(s, args.algos.split(","), backend="multi", frames_limit=args.frames,
                    trail=args.trail, show=args.show, write_video=not args.no_write,
                    mixed=True)
        return
    for s in seqs:
        for algo in [a.strip() for a in args.algos.split(",") if a.strip()]:
            run_seq(s, [algo], backend=args.backend, frames_limit=args.frames,
                    trail=args.trail, show=args.show, write_video=not args.no_write)


if __name__ == "__main__":
    main()
