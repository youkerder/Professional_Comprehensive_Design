# -*- coding: utf-8 -*-
"""实验 04 · 检索演示：输入一张图片，同时检出相关图像与相关视频。

  以图搜图：查询图特征 vs 图像库特征（余弦相似度）→ Top-K 图像
  以图搜视频：查询图特征 vs 视频特征（mean-pool 为主）→ Top-K 视频，
             并把命中视频里与查询最相似的一帧截出来做缩略图。

输出：outputs/results/<查询图名>/
  grid_images.jpg  以图搜图 Top-8 拼图（绿框 = 同场景，红框 = 干扰）
  grid_videos.jpg  以图搜视频 Top-3 拼图（含命中帧缩略图与相似度）
  result.json      排序列表

用法：
  python code/search.py --query sunset_sea_q0
  python code/search.py --query path/to/xxx.jpg --task both --topk 8
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_IMAGES, DATA_QUERIES, DATA_VIDEOS, FEATS_DIR,
                    FRAMES_PER_VIDEO, RESULTS_DIR, TOPK_I2I, TOPK_I2V)
from utils import cosine_scores, load_meta


def find_query_file(q):
    if os.path.isfile(q):
        return q
    p = os.path.join(DATA_QUERIES, q if q.endswith(".jpg") else q + ".jpg")
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"找不到查询图：{q}")


def label_img(img, text, hit, similarity=None):
    img = img.copy()
    h, w = img.shape[:2]
    color = (60, 200, 80) if hit else (50, 50, 230)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6)
    tag = text if similarity is None else f"{text}  {similarity:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(tag, font, 0.52, 1)
    cv2.rectangle(img, (0, h - th - 14), (tw + 10, h), (30, 30, 30), -1)
    cv2.putText(img, tag, (5, h - 7), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def thumb(path, size=(240, 135)):
    img = cv2.imread(path)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="查询图路径或 queries/ 下的文件名（不含扩展名也可）")
    ap.add_argument("--task", default="both", choices=["i2i", "i2v", "both"])
    ap.add_argument("--topk", type=int, default=TOPK_I2I)
    ap.add_argument("--mode", default="mean-pool",
                    choices=["mean-pool", "max-pool", "frame-max"],
                    help="以图搜视频的聚合方式")
    args = ap.parse_args()

    qpath = find_query_file(args.query)
    qstem = os.path.splitext(os.path.basename(qpath))[0]
    qimg = cv2.imread(qpath)

    q_feats = np.load(os.path.join(FEATS_DIR, "queries_resnet50.npy"))
    q_meta = load_meta(os.path.join(FEATS_DIR, "queries_meta.json"))
    q_paths = q_meta["paths"]
    qi = q_paths.index(qpath) if qpath in q_paths else -1
    if qi >= 0:
        q = q_feats[qi]
    else:  # 任意外部图片：现场提特征
        from extract_features import DeepExtractor
        ext = DeepExtractor()
        q = ext.batch([qimg])[0]
        from utils import l2norm
        q = l2norm(q)
    q_scene = q_meta["scenes"][qi] if qi >= 0 else "unknown"

    outdir = os.path.join(RESULTS_DIR, qstem)
    os.makedirs(outdir, exist_ok=True)
    result = {"query": qpath, "query_scene": q_scene}

    size = (224, 126)
    cells = [label_img(cv2.resize(qimg, size), "QUERY", True)]

    if args.task in ("i2i", "both"):
        g_feats = np.load(os.path.join(FEATS_DIR, "images_resnet50.npy"))
        g_meta = load_meta(os.path.join(FEATS_DIR, "images_meta.json"))
        order = np.argsort(-cosine_scores(q, g_feats))[:args.topk]
        result["i2i"] = [{"path": g_meta["paths"][i], "scene": g_meta["scenes"][i],
                          "score": float(cosine_scores(q, g_feats[i:i + 1])[0])}
                         for i in order]
        for rank, i in enumerate(order, 1):
            img = thumb(os.path.join(DATA_IMAGES, g_meta["paths"][i]), size)
            hit = g_meta["scenes"][i] == q_scene
            cells.append(label_img(img, f"#{rank} {g_meta['scenes'][i]}", hit,
                                   result["i2i"][rank - 1]["score"]))

    if args.task in ("i2v", "both"):
        v_meta = load_meta(os.path.join(FEATS_DIR, "videos_meta.json"))
        vnames = v_meta["names"]
        fv = np.array(v_meta["frame_video_idx"])
        frames = np.load(os.path.join(FEATS_DIR, "video_frames_resnet50.npy"))
        if args.mode == "mean-pool":
            vscore = cosine_scores(q, np.load(
                os.path.join(FEATS_DIR, "videos_resnet50_mean.npy")))
        elif args.mode == "max-pool":
            vscore = cosine_scores(q, np.load(
                os.path.join(FEATS_DIR, "videos_resnet50_max.npy")))
        else:
            sim = cosine_scores(q, frames)
            vscore = np.stack([sim[:, fv == vi].max(1) for vi in range(len(vnames))],
                              axis=1)[0]
        order = np.argsort(-vscore)[:TOPK_I2V]
        result["i2v"] = [{"video": vnames[i] + ".mp4", "score": float(vscore[i])}
                         for i in order]
        vcells = [label_img(cv2.resize(qimg, size), "QUERY", True)]
        for rank, i in enumerate(order, 1):
            # 命中视频里与查询最相似的一帧
            sim = cosine_scores(q, frames[fv == i])
            best_local = int(np.argmax(sim))
            frame_no = int(round(best_local * 191 / (FRAMES_PER_VIDEO - 1)))
            cap = cv2.VideoCapture(os.path.join(DATA_VIDEOS, vnames[i] + ".mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                frame = np.zeros((126, 224, 3), np.uint8)
            hit = vnames[i] == q_scene
            vcells.append(label_img(cv2.resize(frame, size),
                                    f"#{rank} {vnames[i]}", hit, float(vscore[i])))
        _grid(vcells, os.path.join(outdir, "grid_videos.jpg"), 4)
        print(f"[i2v] Top-{TOPK_I2V}: " +
              ", ".join(f"{vnames[i]}({vscore[i]:.3f})" for i in order))

    if args.task in ("i2i", "both"):
        print(f"[i2i] Top-{args.topk}: " +
              ", ".join(f"{os.path.basename(r['path'])}({r['score']:.3f})"
                        for r in result["i2i"][:5]) + " ...")
        _grid(cells, os.path.join(outdir, "grid_images.jpg"), 5)

    with open(os.path.join(outdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"结果已保存：{outdir}")


def _grid(cells, out_path, ncol):
    rows = []
    for i in range(0, len(cells), ncol):
        row = cells[i:i + ncol]
        while len(row) < ncol:
            row.append(np.full_like(row[0], 255))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    grid = cv2.copyMakeBorder(grid, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.imwrite(out_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
