# -*- coding: utf-8 -*-
"""实验 04 · 生成实验报告插图（reports/figures/04_*.png）。"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EXP, FIGS_DIR, METRICS_DIR, REPO, DATA_IMAGES, DATA_QUERIES, DATA_VIDEOS

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

C_DEEP = "#4C72B0"     # 蓝
C_HIST = "#DD8452"     # 橙
C_POOL = ["#8C8C8C", "#5F9ED1", "#59A14F"]


def read_csv(name):
    with open(os.path.join(METRICS_DIR, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- 1. 精度柱状图
def fig_accuracy():
    rows = read_csv("accuracy.csv")
    get = lambda task, feat, m: next(float(r[m]) for r in rows
                                     if r["task"] == task and r["feature"] == feat)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=160)
    # i2i
    labels = ["mAP", "P@1", "P@5", "P@10", "R@10"]
    deep = [get("i2i", "deep_resnet50", m) for m in labels]
    hist = [get("i2i", "hist_hsv", m) for m in labels]
    x = np.arange(len(labels))
    ax = axes[0]
    ax.bar(x - 0.19, deep, 0.38, label="ResNet50 深度特征", color=C_DEEP)
    ax.bar(x + 0.19, hist, 0.38, label="HSV 颜色直方图", color=C_HIST)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("以图搜图（图像库 120 张）", fontsize=10)
    ax.axhline(1.0, color="#BBBBBB", lw=0.6, ls="--")
    for xi, v in zip(x - 0.19, deep):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.2)
    for xi, v in zip(x + 0.19, hist):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.2)
    # i2v（单相关项：R@1 = Top-1 命中率；P@3 上限为 1/3，故只画 P@1）
    labels2 = ["mAP", "P@1", "R@1"]
    deep2 = [get("i2v", "deep_resnet50", "mAP"),
             get("i2v", "deep_resnet50", "P@1"),
             get("i2v", "deep_resnet50", "R@1")]
    hist2 = [get("i2v", "hist_hsv", "mAP"),
             get("i2v", "hist_hsv", "P@1"),
             get("i2v", "hist_hsv", "R@1")]
    ax = axes[1]
    x2 = np.arange(len(labels2))
    ax.bar(x2 - 0.19, deep2, 0.38, label="ResNet50 深度特征", color=C_DEEP)
    ax.bar(x2 + 0.19, hist2, 0.38, label="HSV 颜色直方图", color=C_HIST)
    ax.set_xticks(x2, labels2)
    ax.set_ylim(0.5, 1.08)
    ax.set_title("以图搜视频（视频库 12 段，mean-pool）", fontsize=10)
    for xi, v in zip(x2 - 0.19, deep2):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.2)
    for xi, v in zip(x2 + 0.19, hist2):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.2)
    axes[0].legend(fontsize=7.5, loc="lower left")
    axes[0].set_ylabel("指标值")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "04_accuracy_bars.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 2. 每场景 AP
def fig_scene_ap():
    rows = read_csv("accuracy_per_scene.csv")
    scenes = sorted({r["scene"] for r in rows if r["task"] == "i2i"})
    deep = [np.mean([float(r["AP"]) for r in rows if r["task"] == "i2i"
                     and r["feature"] == "deep_resnet50" and r["scene"] == s])
            for s in scenes]
    hist = [np.mean([float(r["AP"]) for r in rows if r["task"] == "i2i"
                     and r["feature"] == "hist_hsv" and r["scene"] == s])
            for s in scenes]
    fig, ax = plt.subplots(figsize=(7.6, 2.7), dpi=160)
    x = np.arange(len(scenes))
    ax.bar(x - 0.19, deep, 0.38, label="ResNet50 深度特征", color=C_DEEP)
    ax.bar(x + 0.19, hist, 0.38, label="HSV 颜色直方图", color=C_HIST)
    ax.set_xticks(x, [s.replace("_", "\n") for s in scenes], fontsize=7)
    ax.set_ylim(0.55, 1.06)
    ax.set_ylabel("以图搜图 AP")
    ax.axhline(1.0, color="#BBBBBB", lw=0.6, ls="--")
    for xi, v in zip(x - 0.19, deep):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=6)
    for xi, v in zip(x + 0.19, hist):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=6)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "04_scene_ap.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 3. 延迟与召回
def fig_latency():
    rows = read_csv("efficiency.csv")
    pools = [("images_lib", "图像库 N=120"), ("video_frames", "视频帧库 N=192"),
             ("big_20000", "放大库 N=20000")]
    methods = ["brute", "faiss_flat", "faiss_ivf", "hnsw"]
    names = {"brute": "暴力(numpy)", "faiss_flat": "FAISS Flat",
             "faiss_ivf": "FAISS IVF", "hnsw": "HNSW"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=160)
    ax = axes[0]
    x = np.arange(len(methods))
    for pi, (pool, label) in enumerate(pools):
        lat = [float(next(r["latency_ms"] for r in rows
                          if r["pool"] == pool and r["index"] == m)) for m in methods]
        ax.bar(x + (pi - 1) * 0.27, lat, 0.26, label=label, color=C_POOL[pi])
    ax.set_yscale("log")
    ax.set_xticks(x, [names[m].replace(" ", "\n") for m in methods], fontsize=7.5)
    ax.set_ylabel("单次查询延迟 (ms，对数轴)")
    ax.set_title("四种索引方式的查询延迟", fontsize=10)
    ax.legend(fontsize=6.8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    ax = axes[1]
    big = [r for r in rows if r["pool"] == "big_20000"]
    lat = [float(r["latency_ms"]) for r in big]
    rec = [float(r["recall_at_k"]) * 100 for r in big]
    colors = [C_POOL[0], C_POOL[1], C_POOL[2], "#C44E52"]
    bars = ax.bar(np.arange(4), lat, 0.55, color=colors)
    for i, (b, r) in enumerate(zip(bars, rec)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.06,
                f"{lat[i]:.2f} ms\nR@10={r:.1f}%", ha="center", fontsize=6.6)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(4), [names[m["index"]] for m in big], fontsize=7.5)
    ax.set_ylabel("N=20000 时延迟 (ms，对数轴)")
    ax.set_title("放大库上的效率-精度权衡", fontsize=10)
    ax.set_ylim(0.5, 60)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "04_latency.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 4. 流程图
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=160)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF0F8", ec="#4C72B0", fs=8.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2,
                                   joinstyle="round", zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3)

    def arrow(x0, y0, x1, y1, color="#666666"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1))

    # 在线检索主链（上行：以图搜图；下行：以图搜视频）
    box(0.25, 2.42, 1.55, 0.78, "查询图像\n（1 张）", fc="#FFF3E6", ec="#DD8452")
    box(2.42, 2.42, 2.30, 0.78, "ResNet50 特征提取\n2048 维，L2 归一化")
    box(5.45, 2.42, 2.55, 0.78, "以图搜图\n与 120 张库图算余弦", fc="#EAF6EC", ec="#59A14F")
    box(8.85, 2.42, 2.05, 0.78, "Top-K 相关图像")
    box(5.45, 1.28, 2.55, 0.78, "以图搜视频\n与 12 段视频算余弦", fc="#EAF6EC", ec="#59A14F")
    box(8.85, 1.28, 2.05, 0.78, "Top-K 相关视频")

    # 库侧离线建库（底部灰色，同一 ResNet50 → 特征库）
    box(2.42, 0.14, 5.58, 0.72,
        "库侧（离线，一次）：图像库 120 张 / 每段视频均匀抽 16 帧\n"
        "→ 同一 ResNet50 提特征（视频做 mean/max 聚合）→ 特征库落盘 .npy",
        fc="#F4F4F4", ec="#999999", fs=7.6)

    arrow(1.80, 2.81, 2.42, 2.81)
    arrow(4.72, 2.86, 5.45, 2.86)
    arrow(4.72, 2.76, 5.45, 1.80)
    arrow(8.00, 2.81, 8.85, 2.81)
    arrow(8.00, 1.67, 8.85, 1.67)
    arrow(3.57, 0.86, 3.57, 2.42, color="#999999")
    ax.text(3.72, 1.55, "同一特征空间", fontsize=6.8, color="#888888",
            va="center", rotation=90)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 3.35)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "04_pipeline.png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 5. 数据集样例
def fig_dataset():
    def thumb(path, w=200):
        img = cv2.imread(path)
        h = int(img.shape[0] * w / img.shape[1])
        return cv2.resize(img, (w, h))

    rows_img = []
    for f in sorted(glob.glob(os.path.join(DATA_VIDEOS, "*.mp4"))):
        cap = cv2.VideoCapture(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 96)
        ok, img = cap.read()
        cap.release()
        t = thumb_img(img)
        cv2.putText(t, os.path.basename(f)[:-4], (3, t.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rows_img.append(t)
    grid_rows = [np.hstack(rows_img[i:i + 4]) for i in range(0, 12, 4)]
    dcells = []
    for f in sorted(glob.glob(os.path.join(DATA_IMAGES, "d*", "*.jpg")))[:4]:
        t = thumb_img(cv2.imread(f))
        cv2.putText(t, os.path.basename(os.path.dirname(f)), (3, t.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 255), 1, cv2.LINE_AA)
        dcells.append(t)
    grid_rows.append(np.hstack(dcells))
    qcells = []
    for f in sorted(glob.glob(os.path.join(DATA_QUERIES, "*.jpg")))[:4]:
        t = thumb_img(cv2.imread(f))
        cv2.putText(t, "QUERY " + os.path.basename(f)[:-4].replace("_q", "#"),
                    (3, t.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (80, 255, 120), 1, cv2.LINE_AA)
        qcells.append(t)
    grid_rows.append(np.hstack(qcells))
    grid = np.vstack(grid_rows)
    cv2.imwrite(os.path.join(FIGS_DIR, "04_dataset_samples.png"), grid,
                [cv2.IMWRITE_JPEG_QUALITY, 95])


def thumb_img(img, w=200):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 6. 演示结果拼图
def fig_demo():
    """把 4 个查询的 i2i / i2v 演示拼图合并为两页大图。"""
    res = os.path.join(EXP, "outputs", "results")
    fname = {"i2i": "grid_images.jpg", "i2v": "grid_videos.jpg"}
    pairs = [("sunset_sea_q0", "aquarium_q1"), ("city_night_q1", "fireworks_q0")]
    for name, (q1, q2) in zip(("i2i", "i2v"), pairs):
        imgs = []
        for q in (q1, q2):
            p = os.path.join(res, q, fname[name])
            imgs.append(cv2.imread(p))
        w = min(i.shape[1] for i in imgs)
        imgs = [cv2.resize(i, (w, int(i.shape[0] * w / i.shape[1]))) for i in imgs]
        grid = np.vstack(imgs)
        cv2.imwrite(os.path.join(FIGS_DIR, f"04_demo_{name}.png"), grid,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    fig_accuracy()
    fig_scene_ap()
    fig_latency()
    fig_pipeline()
    fig_dataset()
    fig_demo()
    for f in sorted(glob.glob(os.path.join(FIGS_DIR, "04_*.png"))):
        print(os.path.basename(f), os.path.getsize(f) // 1024, "KB")
