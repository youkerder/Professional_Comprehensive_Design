# -*- coding: utf-8 -*-
"""实验 04 · 合成本地小型检索测试库。

由于实验指导书对应的 Stanford videosearch 测试集需要编译 VLFEAT/YAEL（Windows 下
非常困难）且原始数据托管在境外，本实验改为用 OpenCV 程化生成一个可控、可复现的
小型库：

  videos/   12 段 8 s、640x360@24fps 的合成短视频（12 个视觉场景）
  images/   12 个视频场景各抽 8 帧（96 张）+ 6 个纯图像干扰场景各 4 张（24 张）= 120 张
  queries/  每个视频场景 2 张查询图（取库外帧并做裁剪/亮度/翻转增强）= 24 张

查询图与库内图像同场景不同帧（且做了增强），因此「自己搜自己」被天然排除；
场景 id 即类别，相关性真值可由目录名自动推导，无需人工标注。

用法：python code/make_dataset.py
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_IMAGES, DATA_QUERIES, DATA_VIDEOS, FRAMES_PER_VIDEO_IN_LIB,
                    IMGS_PER_DISTRACTOR, N_DISTRACTOR_SCENES, QUERIES_PER_SCENE,
                    VIDEO_FPS, VIDEO_H, VIDEO_SECONDS, VIDEO_W)

W, H = VIDEO_W, VIDEO_H
FPS, DUR = VIDEO_FPS, VIDEO_SECONDS


# ------------------------------------------------------------------ 通用工具
def vgrad(top, bottom, h=H, w=W):
    """垂直渐变背景（BGR）。"""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.float32(top)[None, None, :]
    bottom = np.float32(bottom)[None, None, :]
    return (top * (1 - t) + bottom * t).astype(np.uint8).repeat(w, axis=1)


def noise(img, rng, sigma=2.5):
    n = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def seg(canvas, p1, p2, color, th):
    cv2.line(canvas, tuple(np.intp(p1)), tuple(np.intp(p2)), color, th, cv2.LINE_AA)


# ------------------------------------------------------------------ 12 个视频场景
def plan_rng(name):
    return np.random.default_rng(abs(hash(("plan", name))) % (2**32))


def sc_sunset_sea(t, rng):
    horizon = int(H * 0.55)
    img = vgrad((90, 70, 200), (120, 190, 250), h=horizon)
    sea = vgrad((130, 90, 50), (150, 130, 80), h=H - horizon)
    img = np.vstack([img, sea]).astype(np.uint8)
    sun_x = int(W * (0.30 + 0.06 * t))
    sun_y = int(H * (0.30 - 0.04 * t))
    cv2.circle(img, (sun_x, sun_y), 26, (80, 170, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (sun_x, sun_y), 34, (120, 200, 255), 2, cv2.LINE_AA)
    # 海面波纹
    for i in range(14):
        y = int(H * 0.55) + 6 + i * 10
        amp, ph = 2.5 + i * 0.3, i * 1.1 + 6.0 * t
        xs = np.arange(0, W, 4)
        ys = (y + amp * np.sin(xs / 34.0 + ph)).astype(int)
        pts = np.stack([xs, ys], axis=1)
        cv2.polylines(img, [pts], False, (200, 150 + 4 * i, 110), 1, cv2.LINE_AA)
    # 太阳倒影
    for i in range(9):
        y = int(H * 0.55) + 8 + i * 11
        wdt = 26 - i * 2
        x = sun_x + int(6 * np.sin(3 * t + i))
        cv2.ellipse(img, (x, y), (wdt, 2), 0, 0, 360, (90, 180, 250), -1, cv2.LINE_AA)
    # 帆船自左向右
    bx = int(W * (0.12 + 0.72 * t))
    by = int(H * 0.55) + 14 + int(2 * np.sin(5 * t))
    cv2.fillPoly(img, [np.array([[bx - 30, by], [bx + 30, by], [bx + 20, by + 9],
                                 [bx - 20, by + 9]])], (60, 50, 70))
    seg(img, (bx, by), (bx, by - 34), (50, 40, 60), 2)
    cv2.fillPoly(img, [np.array([[bx, by - 34], [bx + 17, by - 10], [bx, by - 10]])],
                 (240, 240, 245))
    return noise(img, rng)


def _city_plan():
    r = plan_rng("city")
    buildings = []
    x = -10
    while x < W + 10:
        bw = int(r.integers(46, 90))
        bh = int(r.integers(90, 190))
        buildings.append((x, H - 46 - bh, bw, bh,
                          tuple(int(v) for v in r.integers(20, 70, 3))))
        x += bw + int(r.integers(4, 14))
    windows = []
    for (bx, by, bw, bh, col) in buildings:
        for wy in range(by + 8, H - 52, 16):
            for wx in range(bx + 6, bx + bw - 8, 13):
                if r.random() < 0.42:
                    windows.append((wx, wy, r.random() < 0.25,
                                    tuple(int(v) for v in r.integers(60, 110, 3))))
    return buildings, windows, r


def _city_render(t, rng):
    img = vgrad((70, 45, 25), (140, 90, 55), h=H - 46)
    img = np.vstack([img, np.full((46, W, 3), (35, 30, 25), np.uint8)])
    for moon in [(int(W * 0.82), int(H * 0.14))]:
        cv2.circle(img, moon, 17, (200, 220, 235), -1, cv2.LINE_AA)
        cv2.circle(img, (moon[0] + 7, moon[1] - 4), 14, (70, 45, 25), -1)
    buildings, windows, r = _city_plan()
    for (bx, by, bw, bh, col) in buildings:
        cv2.rectangle(img, (bx, by), (bx + bw, H - 46), col, -1)
    for (wx, wy, blink, col) in windows:
        if blink and (int(t * FPS) // 9) % 2 == 0:
            continue
        cv2.rectangle(img, (wx, wy), (wx + 6, wy + 8), col, -1)
    # 马路上的车灯
    for i in range(6):
        cx = int((W * (t * (0.35 + 0.05 * i) + i * 0.17)) % (W + 60)) - 30
        cy = H - 18 - (i % 2) * 12
        cv2.circle(img, (cx, cy), 3, (235, 245, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx - 8, cy), 2, (60, 60, 220), -1, cv2.LINE_AA)
        seg(img, (cx, cy), (cx + 16, cy), (150, 170, 255), 1)
    return noise(img, rng)


def sc_forest(t, rng):
    img = vgrad((40, 110, 50), (70, 170, 90))
    r = plan_rng("forest")
    trunks = sorted([float(r.uniform(0, 1.6)) for _ in range(9)])
    for i, base in enumerate(trunks):
        x = int(((base - t * (0.05 + 0.012 * (i % 3))) % 1.3) * (W + 120) - 60)
        tw = 10 + (i % 3) * 7
        cv2.rectangle(img, (x, 52), (x + tw, H), (30, 75, 90), -1)
        for b in range(3):
            cv2.circle(img, (x + tw // 2 + int((b - 1) * 14), 40 + b * 26 + (i % 4) * 8),
                       30 - b * 5, (50, 110 + 20 * b, 45), -1, cv2.LINE_AA)
    for i in range(26):  # 飘落的叶子
        lf = r.random()
        y = int(((lf + t * (0.05 + 0.04 * (lf > 0.5))) % 1.0) * H)
        x = int(lf * W + 24 * np.sin(6 * t + lf * 12))
        col = [(60, 120, 250), (80, 160, 240), (90, 190, 60)][i % 3]
        cv2.ellipse(img, (x, y), (5, 2), 60 * t * 40 + i * 30, 0, 360, col, -1, cv2.LINE_AA)
    return noise(img, rng)


def sc_snow_mount(t, rng):
    img = vgrad((250, 215, 175), (255, 245, 235))
    r = plan_rng("snow")
    for i, (cx, hw, hh) in enumerate([(140, 200, 150), (420, 260, 190), (620, 180, 120)]):
        base = H
        pts = np.array([[cx - hw, base], [cx, base - hh], [cx + hw, base]])
        cv2.fillPoly(img, [pts], (200 + 15 * i, 200 + 10 * i, 215 + 10 * i))
        cv2.fillPoly(img, [np.array([[cx - 40, base - hh + 52], [cx, base - hh],
                                     [cx + 44, base - hh + 56]])], (250, 250, 252))
    for i in range(9):  # 松树
        x = int(r.integers(10, W - 10))
        s = int(r.integers(20, 38))
        cv2.fillPoly(img, [np.array([[x - s, H], [x, H - 2 * s], [x + s, H]])],
                     (60, 80, 45))
    for i in range(70):  # 雪花
        f = r.random()
        y = int(((f + t * (0.10 + 0.08 * (f > 0.6))) % 1.0) * H)
        x = int(f * W + 18 * np.sin(4 * t + f * 20))
        sz = 1 + int(2 * (f > 0.75))
        cv2.circle(img, (x % W, y), sz, (255, 255, 255), -1, cv2.LINE_AA)
    return noise(img, rng)


def _aquarium_render(t, rng):
    img = vgrad((120, 80, 25), (210, 150, 70))
    for i in range(4):  # 光柱
        x0 = int(W * (0.15 + 0.2 * i) + 10 * np.sin(1.5 * t + i))
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array([[x0, 0], [x0 + 46, 0], [x0 + 130, H], [x0 + 40, H]])],
                     (90, 70, 40))
        img = cv2.addWeighted(img, 0.88, overlay, 0.12, 0)
    r = plan_rng("aqua")
    for i in range(4):  # 海草
        x0 = int(W * (0.08 + 0.22 * i))
        for k in range(7):
            xx = x0 + int(9 * np.sin(2.2 * t + k * 0.8 + i))
            cv2.line(img, (x0 + int(6 * np.sin(2.2 * t + (k - 1) * 0.8 + i)), H - k * 12),
                     (xx, H - (k + 1) * 12), (70, 130, 50), 3, cv2.LINE_AA)
    fish_cols = [(40, 130, 250), (70, 170, 255), (200, 200, 240), (30, 100, 220)]
    for i in range(4):
        f = r.random()
        y0 = int(H * (0.25 + 0.55 * f))
        x = int(((f * 0.6 + t * (0.06 + 0.03 * (i % 2))) % 1.25) * (W + 80) - 40)
        y = int(y0 + 16 * np.sin(3.2 * t * (1 + 0.2 * i) + i * 2))
        s = 16 + (i % 3) * 7
        cv2.ellipse(img, (x, y), (s, s // 2 + 3), 0, 0, 360, fish_cols[i], -1, cv2.LINE_AA)
        tail = np.array([[x - s, y], [x - s - 10, y - 8], [x - s - 10, y + 8]])
        cv2.fillPoly(img, [tail], fish_cols[i])
        cv2.circle(img, (x + s // 2, y - 2), 2, (40, 40, 40), -1)
    for i in range(24):  # 气泡
        f = r.random()
        y = int((1 - ((f + t * (0.07 + 0.05 * (f > 0.7))) % 1.0)) * H)
        x = int(f * W + 8 * np.sin(5 * t + f * 30))
        cv2.circle(img, (x, y), 2 + int(3 * (f > 0.8)), (230, 220, 200), 1, cv2.LINE_AA)
    return noise(img, rng)


def sc_desert(t, rng):
    img = vgrad((150, 190, 235), (190, 220, 245), h=int(H * 0.42))
    img = np.vstack([img, vgrad((110, 185, 245), (140, 205, 250), h=H - int(H * 0.42))])
    cv2.circle(img, (int(W * 0.75), int(H * 0.16)), 15, (120, 200, 250), -1, cv2.LINE_AA)
    r = plan_rng("desert")
    for i, (cx, rw, rh, col) in enumerate([(90, 300, 60, (135, 200, 248)),
                                           (400, 360, 80, (120, 190, 246)),
                                           (640, 320, 55, (100, 175, 242))]):
        cv2.ellipse(img, (cx, H - 6), (rw, rh), 0, 180, 360, col, -1, cv2.LINE_AA)
    n_cactus = 3
    for i in range(n_cactus):
        base = r.random()
        x = int(((base + 0.06 * t * (1 + i * 0.15)) % 1.15) * (W + 80) - 40)
        yb = H - 26 + (i % 2) * 10
        ch = 42 + (i % 3) * 14
        cv2.rectangle(img, (x, yb - ch), (x + 12, yb), (40, 110, 60), -1)
        cv2.rectangle(img, (x - 10, yb - ch + 12), (x, yb - ch + 18), (40, 110, 60), -1)
        cv2.rectangle(img, (x - 10, yb - ch + 12), (x - 5, yb - ch - 4), (40, 110, 60), -1)
        cv2.rectangle(img, (x + 12, yb - ch + 20), (x + 22, yb - ch + 26), (40, 110, 60), -1)
        cv2.rectangle(img, (x + 17, yb - ch - 2), (x + 22, yb - ch + 26), (40, 110, 60), -1)
    return noise(img, rng)


def _fireworks_render(t, rng):
    img = np.full((H, W, 3), (14, 8, 6), np.uint8)
    r = plan_rng("fireworks")
    bursts = [(0.15, 0.30, 0), (0.45, 0.22, 1.1), (0.70, 0.35, 2.2),
              (0.30, 0.20, 3.3), (0.58, 0.30, 4.4), (0.82, 0.24, 5.5), (0.25, 0.34, 6.6)]
    cols = [(60, 90, 250), (70, 200, 250), (90, 220, 120), (50, 120, 250), (150, 180, 250)]
    for (bx, by, t0) in bursts:
        age = (t * DUR - t0) % DUR
        if age > 2.4:
            continue
        cx, cy = int(bx * W), int(by * H)
        col = cols[int(bx * 10) % len(cols)]
        n_p = 26
        for p in range(n_p):
            ang = 2 * np.pi * p / n_p + 0.35 * int(bx * 7)
            rad = age * (58 + 14 * ((p * 7) % 5))
            if rad > 96:
                continue
            x = int(cx + rad * np.cos(ang))
            y = int(cy + rad * np.sin(ang) * 0.92)
            fade = max(0, int(255 * (1 - age / 2.4)))
            cv2.circle(img, (x, y), 2, (int(col[0] * fade / 255),
                                        int(col[1] * fade / 255),
                                        int(col[2] * fade / 255)), -1, cv2.LINE_AA)
    cv2.rectangle(img, (0, H - 22), (W, H), (10, 6, 5), -1)
    return noise(img, rng, sigma=2.0)


def sc_traffic(t, rng):
    img = vgrad((240, 210, 150), (250, 235, 220), h=int(H * 0.30))
    img = np.vstack([img, vgrad((80, 82, 86), (60, 62, 66), h=H - int(H * 0.30))])
    for lane in (1, 2):  # 车道虚线滚动
        y = int(H * 0.30) + lane * int(H * 0.23)
        for k in range(-1, 9):
            x = int((k * 90 + t * 130) % (W + 90)) - 45
            cv2.rectangle(img, (x, y - 2), (x + 44, y + 2), (200, 205, 210), -1)
    r = plan_rng("traffic")
    car_cols = [(60, 60, 220), (40, 140, 250), (70, 170, 90), (180, 190, 230),
                (30, 90, 200), (100, 100, 110)]
    for i in range(5):
        lane = i % 2
        y = int(H * 0.30) + 22 + lane * int(H * 0.40)
        speed = (0.10 + 0.045 * i) * (1 if i % 2 == 0 else -1)
        x = int(((i * 0.23 + t * speed) % 1.3) * (W + 140) - 70)
        col = car_cols[i]
        wcar, hcar = 74 + 6 * i, 26
        cv2.rectangle(img, (x, y - hcar), (x + wcar, y), col, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x + 16, y - hcar - 12), (x + wcar - 16, y - hcar + 2),
                      col, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x + 20, y - hcar - 8), (x + wcar - 20, y - hcar), (60, 70, 80), -1)
        for wx in (x + 14, x + wcar - 14):
            cv2.circle(img, (wx, y), 7, (30, 30, 30), -1)
            cv2.circle(img, (wx, y), 3, (110, 110, 110), -1)
    return noise(img, rng)


def sc_balloons(t, rng):
    img = vgrad((240, 190, 100), (250, 235, 200))
    r = plan_rng("balloons")
    for i in range(5):  # 云
        f = r.random()
        cx = int(((f + 0.012 * t) % 1.15) * W)
        cy = int(H * (0.12 + 0.3 * f))
        for k, (dx, dy, rr) in enumerate([(-28, 4, 20), (0, 0, 26), (30, 5, 19)]):
            cv2.ellipse(img, (cx + dx, cy + dy), (rr + 12, rr - 4), 0, 0, 360,
                        (252, 248, 244), -1, cv2.LINE_AA)
    cols = [(60, 80, 230), (50, 160, 240), (70, 170, 60), (40, 120, 245),
            (90, 100, 220), (120, 160, 90)]
    for i in range(7):
        f = r.random()
        bx = int(W * (0.08 + 0.84 * f) + 10 * np.sin(2 * t + i))
        by = int(H * (1.08 - ((f * 0.7 + t * (0.05 + 0.03 * (f > 0.5))) % 1.25)))
        rad = 22 + int(10 * r.random())
        col = cols[i % len(cols)]
        cv2.ellipse(img, (bx, by), (rad, int(rad * 1.18)), 0, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(img, (bx - rad // 3, by - rad // 2), (rad // 5, rad // 4), -25,
                    0, 360, (235, 235, 245), -1)
        pts = []
        for k in range(6):
            pts.append((bx + int(4 * np.sin(t * 3 + k)), by + int(rad * 1.18) + k * 11))
        cv2.polylines(img, [np.array(pts)], False, (120, 120, 130), 1, cv2.LINE_AA)
        cv2.fillPoly(img, np.array([[bx - 5, by + int(rad * 1.18) - 2],
                                    [bx + 5, by + int(rad * 1.18) - 2],
                                    [bx, by + int(rad * 1.18) + 7]]).reshape(1, 3, 2), col)
    return noise(img, rng)


def sc_rain_glass(t, rng):
    img = vgrad((150, 135, 105), (185, 160, 135))
    r = plan_rng("rain")
    for i in range(12):  # 失焦光斑
        f = r.random()
        cx, cy = int(f * W), int(r.random() * H * 0.7)
        col = [(90, 150, 220), (80, 190, 235), (70, 130, 180)][i % 3]
        blur = np.zeros_like(img)
        cv2.circle(blur, (cx, cy), 18 + int(26 * r.random()), col, -1)
        img = cv2.addWeighted(img, 0.96, cv2.GaussianBlur(blur, (0, 0), 9), 0.22, 0)
    for i in range(60):  # 雨丝
        f = r.random()
        x = int(((f + 0.13 * t) % 1.0) * (W + 60)) - 30
        y = int(((f * 0.9 + t * (0.35 + 0.25 * (f > 0.5))) % 1.05) * H) - 10
        seg(img, (x, y), (x - 7, y + 22), (210, 215, 225), 1)
    for i in range(15):  # 玻璃上的水珠
        f = r.random()
        x = int(f * W)
        y = int((((f * 3.1) % 1.0) + t * (0.02 + 0.05 * (f > 0.66))) % 1.0 * H)
        rad = 3 + int(5 * (f > 0.7))
        cv2.circle(img, (x, y), rad, (200, 195, 180), -1, cv2.LINE_AA)
        cv2.circle(img, (x - rad // 3, y - rad // 3), max(1, rad // 3), (240, 240, 235), -1)
    return noise(img, rng, sigma=2.0)


def sc_jelly(t, rng):
    img = vgrad((110, 95, 30), (170, 140, 75))
    r = plan_rng("jelly")
    for i in range(40):  # 浮游颗粒
        f = r.random()
        x, y = int(f * W), int((1 - ((f * 0.8 + t * 0.03) % 1.0)) * H)
        cv2.circle(img, (x, y), 1, (150, 140, 110), -1)
    for i in range(3):
        f = r.random()
        jx = int(W * (0.2 + 0.3 * f) + 14 * np.sin(1.8 * t + i * 2))
        pulse = np.sin(2.4 * t + i * 1.5)
        jy = int(H * (0.85 - ((f * 0.5 + t * (0.035 + 0.012 * i)) % 0.95)))
        rw = 34 + int(9 * (f > 0.5))
        rh = max(10, int(rw * 0.72 + 7 * pulse))
        col = (170, 170, 205) if i != 1 else (150, 150, 230)
        cv2.ellipse(img, (jx, jy), (rw, rh), 0, 180, 360, col, 2, cv2.LINE_AA)
        cv2.ellipse(img, (jx, jy + 2), (rw - 4, max(6, rh - 4)), 0, 180, 360, col, -1)
        for k in range(5):
            x0 = jx - rw + (2 * rw) * k / 4.0
            pts = [(int(x0 + 6 * np.sin(3 * t + k + s * 0.7)), jy + 8 + s * 11)
                   for s in range(6)]
            cv2.polylines(img, [np.array(pts)], False, col, 1, cv2.LINE_AA)
    return noise(img, rng, sigma=2.0)


def sc_autumn(t, rng):
    img = vgrad((70, 130, 210), (110, 180, 235))
    r = plan_rng("autumn")
    for i, tx in enumerate([90, 260, 470, 600]):  # 树
        th = 90 + (i % 2) * 30
        cv2.rectangle(img, (tx - 7, H - th - 30), (tx + 7, H - 30), (35, 60, 70), -1)
        for b in range(6):
            cv2.circle(img, (tx + int((b % 3) - 1) * 26, H - th - 34 - (b // 3) * 24),
                       34, [(50, 110, 220), (60, 140, 235), (80, 100, 200)][b % 3],
                       -1, cv2.LINE_AA)
    cv2.rectangle(img, (330, H - 58), (470, H - 40), (30, 45, 60), -1)   # 长椅
    cv2.rectangle(img, (330, H - 40), (470, H - 34), (25, 40, 50), -1)
    for lx in (340, 460):
        cv2.rectangle(img, (lx, H - 34), (lx + 8, H - 16), (25, 40, 50), -1)
    cv2.rectangle(img, (0, H - 30), (W, H), (45, 90, 120), -1)
    for i in range(34):  # 落叶
        f = r.random()
        y = int(((f + t * (0.06 + 0.05 * (f > 0.55))) % 1.0) * H)
        x = int(f * W + 20 * np.sin(5 * t + f * 18))
        col = [(50, 130, 250), (70, 160, 240), (90, 190, 120)][i % 3]
        cv2.ellipse(img, (x, y), (6, 3), 40 * t * 30 + i * 47, 0, 360, col, -1, cv2.LINE_AA)
    return noise(img, rng)


VIDEO_SCENES = {
    "sunset_sea": sc_sunset_sea,
    "city_night": _city_render,
    "forest": sc_forest,
    "snow_mountain": sc_snow_mount,
    "aquarium": _aquarium_render,
    "desert_road": sc_desert,
    "fireworks": _fireworks_render,
    "traffic": sc_traffic,
    "balloons": sc_balloons,
    "rain_window": sc_rain_glass,
    "jellyfish": sc_jelly,
    "autumn_park": sc_autumn,
}


# ------------------------------------------------------------------ 干扰图像场景
def d_mosaic(seed):
    r = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    base = r.integers(0, 180, 3)
    for gy in range(0, H, 40):
        for gx in range(0, W, 40):
            col = np.clip(base + r.integers(-60, 60, 3), 0, 255)
            cv2.rectangle(img, (gx, gy), (gx + 39, gy + 39), tuple(int(v) for v in col), -1)
    return img


def d_smooth_noise(seed):
    r = np.random.default_rng(seed)
    small = r.integers(0, 255, (6, 11, 3), dtype=np.uint8)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (0, 0), 3)


def d_gradient_art(seed):
    r = np.random.default_rng(seed)
    img = vgrad(tuple(int(v) for v in r.integers(30, 220, 3)),
                tuple(int(v) for v in r.integers(30, 220, 3)))
    for i in range(7):
        cx, cy = int(r.integers(0, W)), int(r.integers(0, H))
        rad = int(r.integers(40, 130))
        col = tuple(int(v) for v in r.integers(40, 255, 3))
        ov = img.copy()
        cv2.circle(ov, (cx, cy), rad, col, -1, cv2.LINE_AA)
        img = cv2.addWeighted(img, 0.62, ov, 0.38, 0)
    return img


def d_shapes_flat(seed):
    r = np.random.default_rng(seed)
    img = np.full((H, W, 3), tuple(int(v) for v in r.integers(35, 200, 3)), np.uint8)
    for i in range(12):
        col = tuple(int(v) for v in r.integers(30, 255, 3))
        kind = r.integers(0, 3)
        x, y = int(r.integers(0, W)), int(r.integers(0, H))
        s = int(r.integers(24, 90))
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + s, y + int(s * 0.7)), col, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.circle(img, (x, y), s // 2, col, -1, cv2.LINE_AA)
        else:
            cv2.fillPoly(img, [np.array([[x, y], [x + s, y + s // 2], [x, y + s]])], col)
    return img


def d_stripes(seed):
    r = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    c1 = tuple(int(v) for v in r.integers(40, 220, 3))
    c2 = tuple(int(v) for v in r.integers(40, 220, 3))
    width = int(r.integers(18, 44))
    ang = float(r.uniform(15, 75))
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    for k in range(-W, 2 * W, width * 2):
        cv2.rectangle(img, (k, -H), (k + width, 2 * H), c1, -1)
        cv2.rectangle(img, (k + width, -H), (k + 2 * width, 2 * H), c2, -1)
    big = cv2.warpAffine(img, M, (W, H))
    xs = np.arange(W)[None, :, None].repeat(H, 0).astype(np.float32)
    ys = np.arange(H)[:, None, None].repeat(W, 1).astype(np.float32)
    wave = np.sin(xs / 46 + ys / 90) * 0.5 + 0.5          # (H, W, 1)
    return (big * (0.75 + 0.25 * wave)).astype(np.uint8)


def d_blur_blobs(seed):
    r = np.random.default_rng(seed)
    img = np.full((H, W, 3), (25, 25, 25), np.uint8)
    for i in range(9):
        cx, cy = int(r.integers(0, W)), int(r.integers(0, H))
        col = tuple(int(v) for v in r.integers(50, 255, 3))
        cv2.circle(img, (cx, cy), int(r.integers(60, 150)), col, -1)
    return cv2.GaussianBlur(img, (0, 0), 21)


DISTRACTOR_SCENES = {
    "d1_mosaic": d_mosaic,
    "d2_noise": d_smooth_noise,
    "d3_gradient": d_gradient_art,
    "d4_shapes": d_shapes_flat,
    "d5_stripes": d_stripes,
    "d6_blobs": d_blur_blobs,
}


# ------------------------------------------------------------------ 主流程
def augment(frame_bgr, rng):
    """查询图增强：随机裁剪 + 亮度/对比度扰动 + 随机水平翻转。"""
    h, w = frame_bgr.shape[:2]
    s = rng.uniform(0.78, 0.92)
    ch, cw = int(h * s), int(w * s)
    y0 = int(rng.integers(0, h - ch + 1))
    x0 = int(rng.integers(0, w - cw + 1))
    out = frame_bgr[y0:y0 + ch, x0:x0 + cw].copy()
    alpha = rng.uniform(0.88, 1.14)
    beta = rng.uniform(-16, 16)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = out[:, ::-1]
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)


def main():
    manifest, q_rows = [], []
    n_frames_total = 0

    # ---- 1) 合成 12 段视频 + 抽帧入库 ----
    for si, (name, fn) in enumerate(VIDEO_SCENES.items()):
        vpath = os.path.join(DATA_VIDEOS, f"{name}.mp4")
        vw = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        assert vw.isOpened(), f"VideoWriter 打开失败：{vpath}"
        total = FPS * DUR
        lib_idx = set(np.linspace(8, total - 9, FRAMES_PER_VIDEO_IN_LIB).astype(int))
        for fi in range(total):
            rng = np.random.default_rng(1000 * si + fi)
            frame = fn(fi / total, rng)
            vw.write(frame)
            if fi in lib_idx:
                ipath = os.path.join(DATA_IMAGES, name, f"{name}_f{fi:03d}.jpg")
                os.makedirs(os.path.dirname(ipath), exist_ok=True)
                cv2.imwrite(ipath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                manifest.append((os.path.relpath(ipath, DATA_IMAGES), name, "video", fi))
        n_frames_total += total
        vw.release()
        print(f"[video] {name}.mp4  ({total} 帧)")

    # ---- 2) 纯图像干扰场景 ----
    for si, (name, fn) in enumerate(DISTRACTOR_SCENES.items()):
        os.makedirs(os.path.join(DATA_IMAGES, name), exist_ok=True)
        for k in range(IMGS_PER_DISTRACTOR):
            seed = 7000 + 100 * si + k
            img = fn(seed)
            ipath = os.path.join(DATA_IMAGES, name, f"{name}_v{k}.jpg")
            cv2.imwrite(ipath, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest.append((os.path.relpath(ipath, DATA_IMAGES), name, "image", -1))

    # ---- 3) 查询图（库外帧 + 增强）----
    used = {m[0].split(os.sep)[-1].rsplit("_f", 1)[1][:3] for m in manifest if m[2] == "video"}
    for si, name in enumerate(VIDEO_SCENES):
        total = FPS * DUR
        for k in range(QUERIES_PER_SCENE):
            rng = np.random.default_rng(9000 + 100 * si + k)
            while True:
                fi = int(rng.integers(0, total))
                if f"{fi:03d}" not in used:
                    break
            frame = VIDEO_SCENES[name](fi / total, np.random.default_rng(1000 * si + fi))
            qimg = augment(frame, rng)
            qpath = os.path.join(DATA_QUERIES, f"{name}_q{k}.jpg")
            cv2.imwrite(qpath, qimg, [cv2.IMWRITE_JPEG_QUALITY, 88])
            q_rows.append((f"{name}_q{k}.jpg", name, fi))

    with open(os.path.join(DATA_IMAGES, "images_manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["relpath", "scene", "source", "frame_idx"])
        wcsv.writerows(manifest)
    with open(os.path.join(DATA_QUERIES, "relevance.csv"), "w", newline="",
              encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["query_file", "scene", "source_frame"])
        wcsv.writerows(q_rows)

    print(f"\n完成：视频 {len(VIDEO_SCENES)} 段（{n_frames_total} 帧），"
          f"图像库 {len(manifest)} 张，查询图 {len(q_rows)} 张")
    print(f"  {DATA_VIDEOS}\n  {DATA_IMAGES}\n  {DATA_QUERIES}")


if __name__ == "__main__":
    main()
