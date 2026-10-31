# -*- coding: utf-8 -*-
"""根据模型预测的逐帧得分生成摘要视频、关键帧图片与得分柱状图。

相对上游 gen_summary.py 的改动（都是为跑通本实验）：
1. 视频目录 / 标注文件按工作区实际布局传参，不再写死 ydata-tvsum50-v1_1 层级；
2. 不再把整段视频读进内存（上游会把所有帧堆进一个 ndarray，几 GB），改为单次解码边读边写；
3. 关键帧用 cv2.imwrite 落盘，柱状图 savefig 后立即 close，避免句柄泄漏；
4. 去掉 seaborn 依赖，柱状图用 matplotlib 原生实现：橙色 = 关键帧；
5. 支持 --ids 指定 10 段测试视频，逐视频输出 score_bar.png，并额外输出总览图。

用法：
  python gen_summary.py --h5_path ... --json_path ... --video_dir ... --save_dir ... --ids ... -b
"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_bar(ax, pred_score, pred_summary, title):
    """画 320 根柱：橙色 = 被选为关键帧，青绿 = 未选中。"""
    n = 320
    idx = np.linspace(0, len(pred_score) - 1, n).astype(int)
    height = np.asarray(pred_score)[idx]
    sel = np.asarray(pred_summary)[idx]
    colors = ["orange" if s > 0 else "lightseagreen" for s in sel]
    ax.bar(x=np.arange(n), height=height, color=colors, width=1.0, linewidth=0)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("sampled frame index", fontsize=8)
    ax.set_ylabel("importance score", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, n)


def gen_one(f_data, vid, json_dict, video_path, save_dir, fps_out=24, save_frames=True):
    g = f_data["video_" + vid]
    cps = g["change_points"][()]
    pred_score = json_dict[vid]["pred_score"]
    pred_selected = json_dict[vid]["pred_selected"]

    # 关键帧所在帧号：每个被选段内得分最高的帧
    keyframe_idx = []
    for sel in pred_selected:
        s, e = cps[sel][0], cps[sel][1]
        keyframe_idx.append(int(np.argmax(pred_score[s:e]) + s))
    keyframe_idx = sorted(set(keyframe_idx))

    # 选中段覆盖的帧号区间
    ranges = sorted((int(cps[s][0]), int(cps[s][1])) for s in pred_selected)

    out_dir = os.path.join(save_dir, vid)
    kf_dir = os.path.join(out_dir, "keyframes")
    os.makedirs(kf_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [WARN] 打不开视频:", video_path)
        return None
    ok, frame = cap.read()
    if not ok:
        print("  [WARN] 读不到帧:", video_path)
        cap.release()
        return None
    h, w = frame.shape[:2]

    writer = cv2.VideoWriter(os.path.join(out_dir, "summary.avi"),
                             cv2.VideoWriter_fourcc(*"XVID"), fps_out, (w, h))
    kf_set = set(keyframe_idx)
    n_written = 0
    i = 0
    while ok:
        if any(a <= i < b for a, b in ranges):
            writer.write(frame)
            n_written += 1
        if save_frames and i in kf_set:
            cv2.imwrite(os.path.join(kf_dir, "%05d.jpg" % i), frame)
        ok, frame = cap.read()
        i += 1
    cap.release()
    writer.release()

    n_frames = int(g["length"][()])
    fig, ax = plt.subplots(figsize=(12, 3.2))
    build_bar(ax, pred_score, json_dict[vid]["pred_summary"],
              "%s  (%.1f%% frames selected)" % (vid, 100.0 * n_written / max(n_frames, 1)))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "score_bar.png"), dpi=110)
    plt.close(fig)

    print("  %-12s 段数=%2d 关键帧=%3d 摘要帧=%5d/%d (%.1f%%)"
          % (vid, len(pred_selected), len(keyframe_idx), n_written, n_frames,
             100.0 * n_written / max(n_frames, 1)))
    return dict(vid=vid, n_summary=n_written, n_frames=n_frames,
                n_keyshots=len(keyframe_idx), n_segments=len(pred_selected))


def combined_bar(json_dict, vids, save_path):
    cols = 2
    rows = int(np.ceil(len(vids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, vid in zip(axes, vids):
        build_bar(ax, json_dict[vid]["pred_score"], json_dict[vid]["pred_summary"], vid)
    for ax in axes[len(vids):]:
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print("总览柱状图:", save_path)


def main():
    ap = argparse.ArgumentParser(description="Generate keyshots, keyframes and score bar.")
    ap.add_argument("--h5_path", required=True, help="fcsn_tvsum.h5")
    ap.add_argument("-j", "--json_path", required=True, help="模型输出的逐视频预测 json")
    ap.add_argument("-r", "--video_dir", required=True, help="原始 mp4 目录")
    ap.add_argument("-s", "--save_dir", required=True, help="结果输出根目录")
    ap.add_argument("--ids", default=None, help="测试视频 id 列表文件（每行一个）")
    ap.add_argument("--n", type=int, default=10, help="最多生成前 N 个视频")
    ap.add_argument("--fps", type=int, default=24, help="摘要视频帧率")
    ap.add_argument("-b", "--bar", action="store_true", help="画得分柱状图")
    args = ap.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        json_dict = json.load(f)
    if args.ids:
        with open(args.ids, "r", encoding="utf-8") as f:
            vids = [l.strip().replace("video_", "") for l in f if l.strip()]
    else:
        vids = list(json_dict.keys())
    vids = [v for v in vids if v in json_dict][:args.n]

    os.makedirs(args.save_dir, exist_ok=True)
    results = []
    with h5py.File(args.h5_path, "r") as f_data:
        for vid in vids:
            video_path = os.path.join(args.video_dir, vid + ".mp4")
            res = gen_one(f_data, vid, json_dict, video_path, args.save_dir,
                          fps_out=args.fps, save_frames=True)
            if res:
                results.append(res)
    if args.bar and results:
        combined_bar(json_dict, [r["vid"] for r in results],
                     os.path.join(args.save_dir, "score_bars_all.png"))


if __name__ == "__main__":
    main()
