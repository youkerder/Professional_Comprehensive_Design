# -*- coding: utf-8 -*-
"""生成实验报告用的插图（输出到 reports/figures/03_*.png）。

图 1：训练损失 + 评测期 F-score / AUC 曲线（数据来自 logs/train_bce.log）
图 2：10 段测试视频的 Precision / Recall / F-score 柱状图（来自 outputs/metrics/prf_test10.csv）
图 3：FCSN / 线性基线 / 两种 Oracle 上界的 F-score 对比（来自 outputs/metrics/baseline_compare.csv）
图 4：某段视频的关键帧拼图（来自 outputs/summary/<id>/keyframes/）
图 5：摘要视频成片抽样帧（来自 outputs/summary/<id>/summary.avi）
图 6：两段视频的得分柱状图（直接复制 outputs/summary/<id>/score_bar.png）
"""
import csv
import os
import re
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 仓库根目录（由本文件位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根）
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

EXP = os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN")
FIGS = os.path.join(BASE, "reports", "figures")
os.makedirs(FIGS, exist_ok=True)
LOG = os.path.join(EXP, 'logs', 'train_bce.log')

COLORS = ['#2f7ed8', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']


def parse_log(path):
    """返回 (epochs, losses, eval_epochs, P, R, F, AUC)。"""
    losses = []
    evals = []
    cur_epoch = None
    txt = open(path, 'r', encoding='utf-8', errors='replace').read().replace('\r', '')
    for line in txt.split('\n'):
        m = re.search(r'Save parameters at .*epoch-(\d+)\.pkl', line)
        if m:
            cur_epoch = int(m.group(1))
        m = re.search(r'\|\s+mean\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|', line)
        if m and cur_epoch is not None:
            evals.append([cur_epoch, float(m.group(1)), float(m.group(2)), float(m.group(3)), None])
        m = re.search(r'帧级 AUC = ([\d.]+)', line)
        if m and evals and evals[-1][4] is None:
            evals[-1][4] = float(m.group(1))
        # 每个 epoch 的进度条最终行：N/150 [.., loss=X]
        m = re.search(r'(\d+)/150 \[[^\]]*?loss=([\d.]+)\]', line)
        if m:
            ep = int(m.group(1)) - 1
            if ep < 0:
                continue
            while len(losses) <= ep:
                losses.append(float('nan'))
            losses[ep] = float(m.group(2))
    losses = [v for v in losses if not (isinstance(v, float) and np.isnan(v))] or []
    return losses, evals


def fig_curves(losses, evals):
    fig, ax1 = plt.subplots(figsize=(9.2, 4.6), dpi=150)
    ax1.plot(range(len(losses)), losses, color=COLORS[0], lw=1.6, label='训练损失 (BCE, pos_weight≈4.7)')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('训练损失', color=COLORS[0])
    ax1.tick_params(axis='y', labelcolor=COLORS[0])
    ax1.grid(alpha=0.3, ls=':')

    ax2 = ax1.twinx()
    ee = [e[0] for e in evals]
    l2, = ax2.plot(ee, [e[3] for e in evals], 'o-', color=COLORS[1], lw=1.5, ms=4,
                   label='F-score (10 段测试)')
    l3, = ax2.plot(ee, [e[4] if e[4] else np.nan for e in evals], 's--', color=COLORS[2], lw=1.4, ms=4,
                   label='帧级 AUC')
    ax2.axhline(0.5, color='gray', lw=1, ls='-.')
    ax2.text(2, 0.512, 'AUC = 0.5（随机）', fontsize=8, color='gray')
    ax2.set_ylabel('评测指标')
    ax2.set_ylim(0.15, 0.75)
    ax1.legend([ax1.get_lines()[0], l2, l3],
               ['训练损失 (BCE, pos_weight≈4.7)', 'F-score (10 段测试)', '帧级 AUC'],
               loc='upper right', fontsize=8, framealpha=0.9)
    plt.title('FCSN 训练过程（150 epoch，每 10 epoch 评测一次）', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGS, '03_curves_train.png')
    plt.savefig(out)
    plt.close()
    print('saved', out)


def fig_prf():
    rows = list(csv.DictReader(open(os.path.join(EXP, 'outputs/metrics/prf_test10.csv'), encoding='utf-8-sig')))
    ids = [r['video_id'] for r in rows]
    P = [float(r['Precision']) for r in rows]
    R = [float(r['Recall']) for r in rows]
    F = [float(r['F-score']) for r in rows]
    x = np.arange(len(ids))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10.2, 4.2), dpi=150)
    ax.bar(x - w, P, w, label='Precision', color=COLORS[0])
    ax.bar(x, R, w, label='Recall', color=COLORS[1])
    ax.bar(x + w, F, w, label='F-score', color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('指标值')
    ax.set_ylim(0, 0.75)
    ax.grid(axis='y', alpha=0.3, ls=':')
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    plt.title('10 段测试视频的 Precision / Recall / F-score', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGS, '03_prf_bars.png')
    plt.savefig(out)
    plt.close()
    print('saved', out)


def fig_compare():
    rows = [r for r in csv.DictReader(open(os.path.join(EXP, 'outputs/metrics/baseline_compare.csv'),
                                           encoding='utf-8-sig'))]
    ids = sorted({r['video_id'] for r in rows})
    methods = ['FCSN', 'Ridge', 'Oracle320', 'OracleFull']
    label_map = {'FCSN': 'FCSN（本实验）', 'Ridge': '线性岭回归基线',
                 'Oracle320': 'Oracle（gtscore 320 采样）', 'OracleFull': 'Oracle（gtscore 全帧）'}
    fig, ax = plt.subplots(figsize=(10.2, 4.2), dpi=150)
    w = 0.2
    for i, m in enumerate(methods):
        vals = []
        for vid in ids:
            hit = [r for r in rows if r['video_id'] == vid and r['method'] == m]
            vals.append(float(hit[0]['F-score']) if hit else np.nan)
        ax.bar(np.arange(len(ids)) + (i - 1.5) * w, vals, w, label=label_map[m], color=COLORS[i])
    ax.set_xticks(np.arange(len(ids)))
    ax.set_xticklabels(ids, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('F-score')
    ax.grid(axis='y', alpha=0.3, ls=':')
    ax.legend(fontsize=8, ncol=2)
    plt.title('FCSN 与线性基线、Oracle 上界的逐视频 F-score 对比', fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGS, '03_compare_bars.png')
    plt.savefig(out)
    plt.close()
    print('saved', out)


def fig_keyframes(vid='xxdtq8mxegs'):
    d = os.path.join(EXP, 'outputs/summary', vid, 'keyframes')
    files = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    n = len(files)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for ax, fn in zip(axes, files):
        img = cv2.cvtColor(cv2.imread(os.path.join(d, fn)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title('第 %s 帧' % os.path.splitext(fn)[0].lstrip('0'), fontsize=8)
        ax.axis('off')
    for ax in axes[len(files):]:
        ax.axis('off')
    plt.suptitle('视频 %s 的 %d 个关键帧（每个选中镜头取一帧）' % (vid, n), fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGS, '03_keyframes.png')
    plt.savefig(out)
    plt.close()
    print('saved', out, '(%d 帧)' % n)


def fig_summary_strip(vid='xxdtq8mxegs'):
    path = os.path.join(EXP, 'outputs/summary', vid, 'summary.avi')
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), 8).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    fig, axes = plt.subplots(2, 4, figsize=(12, 3.6), dpi=150)
    for ax, fr, idx in zip(axes.ravel(), frames, idxs):
        ax.imshow(fr)
        ax.set_title('摘要第 %d 帧' % idx, fontsize=8)
        ax.axis('off')
    plt.suptitle('摘要视频 %s/summary.avi 抽样帧（共 %d 帧，约 30 s）' % (vid, total), fontsize=11)
    plt.tight_layout()
    out = os.path.join(FIGS, '03_summary_frames.png')
    plt.savefig(out)
    plt.close()
    print('saved', out)


def copy_bars():
    for vid, name in [('xxdtq8mxegs', '03_score_bar_good.png'), ('JKpqYvAdIsw', '03_score_bar_bad.png')]:
        src = os.path.join(EXP, 'outputs/summary', vid, 'score_bar.png')
        shutil.copyfile(src, os.path.join(FIGS, name))
        print('copied', name)


if __name__ == '__main__':
    losses, evals = parse_log(LOG)
    print('解析到 %d 个 epoch 的 loss，%d 次评测' % (len(losses), len(evals)))
    print('评测点：', [(e[0], round(e[3], 3), e[4]) for e in evals])
    fig_curves(losses, evals)
    fig_prf()
    fig_compare()
    fig_keyframes()
    fig_summary_strip()
    copy_bars()
