# -*- coding: utf-8 -*-
"""指标校准：在 10 段测试视频上对比三种预测的 P/R/F。

1. oracle  —— 用 eccv16 h5 的 gtscore（20 位标注者均值）直接选段，作为「上界参考」；
2. ridge   —— 用训练集 40 段拟合的岭回归（纯线性基线）；
3. fcsn    —— 训练好的 FCSN 权重。

输出 outputs/metrics/baseline_compare.csv。
"""
import argparse
import csv
import os

import h5py
import numpy as np
import torch

import eval as eval_utils
from fcsn import FCSN

# 仓库根目录（由本文件位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根）
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

DATA = os.path.join(BASE, "datasets", "tvsum50")
OUT = os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN", "outputs", "metrics")


def load_ids(path):
    with open(path, 'r', encoding='utf-8') as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    # test_ids.txt 里可能是 video_xxx 或纯 youtube id，统一去掉前缀
    return [i[len('video_'):] if i.startswith('video_') else i for i in ids]


def metrics_vs_users(video_info, pred_score_np, pred_selected):
    """按 select_keyshots 的口径算 P/R/F：只有 15% 被选中为 1。"""
    pred_summary = np.zeros(int(video_info['length'][()]), dtype=np.int64)
    cps = video_info['change_points'][()]
    for i in pred_selected:
        pred_summary[cps[i][0]:cps[i][1]] = 1
    true_summary_arr = video_info['user_summary'][()]
    res = np.mean([eval_utils.eval_metrics(pred_summary, ts) for ts in true_summary_arr], axis=0)
    return res, pred_summary


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    return float((rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else 0.5


def select_by_score(video_info, score, n):
    """与 eval.select_keyshots 同样的 knapsack 选段，但分数由外部给。"""
    cps = video_info['change_points'][()]
    weight = video_info['n_frame_per_seg'][()]
    pred_value = [float(np.mean(score[c[0]:c[1]])) for c in cps]
    _, selected = eval_utils.knapsack(pred_value, weight, int(0.15 * n))
    return selected[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN", "models", "fcsn_best.pkl"))
    ap.add_argument('--ids', default=os.path.join(BASE, "experiments", "03_VideoSummarization_FCSN", "logs", "test_ids.txt"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = load_ids(args.ids)

    model = FCSN(2).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))
    model.eval()

    rows = []
    with h5py.File(os.path.join(DATA, 'fcsn_tvsum.h5'), 'r') as f, \
            h5py.File(os.path.join(DATA, 'eccv16_dataset_tvsum_google_pool5_local.h5'), 'r') as f16:
        # ---- 线性基线：训练集 40 段拟合 ridge ----
        tr_keys = [k for k in sorted(f.keys()) if k[len('video_'):] not in ids]
        X = np.concatenate([f[k]['feature'][()] for k in tr_keys]).astype(np.float64)
        y = np.concatenate([f[k]['label'][()] for k in tr_keys]).astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - mu) / sd
        lam = 10.0
        w = np.linalg.solve(Xn.T @ Xn + lam * np.eye(Xn.shape[1]), Xn.T @ (y - y.mean()))
        b = y.mean() - ((0 - 0) if False else 0)
        print('岭回归训练完成：%d 帧 × %d 维' % Xn.shape)

        for vid in ids:
            key = 'video_' + vid
            g = f[key]
            fe = g['feature'][()].astype(np.float64)
            label = g['label'][()]
            total = int(f16[key]['length'][()])
            picks = f16[key]['picks'][()]
            gts = f16[key]['gtscore'][()]

            # FCSN
            with torch.no_grad():
                logits = model(torch.tensor(fe).float().cuda().t().unsqueeze(0))
                p_fcsn = torch.softmax(logits.squeeze(0), dim=0)[1].cpu().numpy()
            p_fcsn_full = eval_utils.upsample(p_fcsn, total)
            sel_fcsn = select_by_score(g, p_fcsn_full, total)
            met_fcsn, _ = metrics_vs_users(g, p_fcsn_full, sel_fcsn)

            # ridge
            p_ridge = ((fe - mu) / sd) @ w
            p_ridge_full = eval_utils.upsample(p_ridge, total)
            sel_ridge = select_by_score(g, p_ridge_full, total)
            met_ridge, _ = metrics_vs_users(g, p_ridge_full, sel_ridge)

            # oracle：用 gtscore 在 320 采样点上选
            gts320 = gts[picks]
            gts_full = eval_utils.upsample(gts320, total)
            sel_oracle = select_by_score(g, gts_full, total)
            met_oracle, _ = metrics_vs_users(g, gts_full, sel_oracle)

            # 真正上界：直接在全长 gtscore 上选
            sel_top = select_by_score(g, gts, total)
            met_top, _ = metrics_vs_users(g, gts, sel_top)

            auc_fcsn = auc(np.repeat(p_fcsn, total // 320)[:total], np.repeat(label, total // 320)[:total])
            auc_ridge = auc(np.repeat(p_ridge, total // 320)[:total], np.repeat(label, total // 320)[:total])

            rows.append([vid, total, 'FCSN', auc_fcsn] + [float(v) for v in met_fcsn])
            rows.append([vid, total, 'Ridge', auc_ridge] + [float(v) for v in met_ridge])
            rows.append([vid, total, 'Oracle320', auc_ridge] + [float(v) for v in met_oracle])
            rows.append([vid, total, 'OracleFull', 1.0] + [float(v) for v in met_top])
            print('%s N=%d | FCSN F=%.3f AUC=%.3f | Ridge F=%.3f | Oracle320 F=%.3f | OracleFull F=%.3f'
                  % (vid, total, met_fcsn[2], auc_fcsn, met_ridge[2], met_oracle[2], met_top[2]))

    path = os.path.join(OUT, 'baseline_compare.csv')
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.writer(fh)
        wr.writerow(['video_id', 'n_frames', 'method', 'AUC', 'Precision', 'Recall', 'F-score'])
        for r in rows:
            wr.writerow([r[0], r[1], r[2], '%.4f' % r[3], '%.4f' % r[4], '%.4f' % r[5], '%.4f' % r[6]])
        for m in ['FCSN', 'Ridge', 'Oracle320', 'OracleFull']:
            sub = [r for r in rows if r[2] == m]
            wr.writerow(['MEAN', '', m, '',
                         '%.4f' % np.mean([r[4] for r in sub]),
                         '%.4f' % np.mean([r[5] for r in sub]),
                         '%.4f' % np.mean([r[6] for r in sub])])
    print('已保存', path)


if __name__ == '__main__':
    main()
