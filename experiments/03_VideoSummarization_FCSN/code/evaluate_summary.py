# -*- coding: utf-8 -*-
"""对指定的测试视频跑训练好的 FCSN，计算 Precision / Recall / F-score 并导出表格。

输出：
  --json_out  逐视频的 pred_score / pred_selected / pred_summary（供 gen_summary.py 使用）
  --csv_out   逐视频 P/R/F 明细 + 平均值

用法：
  python evaluate_summary.py --ckpt ../models/fcsn_final.pkl --ids ../logs/test_ids.txt \
      --json_out ../outputs/scores/final_test.json --csv_out ../outputs/metrics/prf.csv
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np
import torch

import eval as eval_utils
from fcsn import FCSN


def load_ids(path, n=10):
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip().replace("video_", "") for line in f if line.strip()]
    return ids[:n] if n else ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=None, help="fcsn_tvsum.h5（默认取 config.Config）")
    ap.add_argument("--ckpt", required=True, help="模型权重 .pkl")
    ap.add_argument("--ids", default=None, help="测试视频 id 列表文件（每行一个）")
    ap.add_argument("--n", type=int, default=10, help="取前 N 个测试视频")
    ap.add_argument("--json_out", required=True)
    ap.add_argument("--csv_out", required=True)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    from config import Config
    cfg = Config()
    data_path = args.data_path or cfg.data_path
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")

    ids = load_ids(args.ids, args.n) if args.ids else None
    if ids is None:
        with h5py.File(data_path, "r") as f:
            ids = sorted(k[len("video_"):] for k in f.keys())[:args.n]

    model = FCSN(cfg.n_class).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("加载权重: %s | 设备: %s | 测试视频: %d" % (args.ckpt, device, len(ids)))

    out_dict, rows = {}, []
    with h5py.File(data_path, "r") as f, torch.no_grad():
        for vid in ids:
            g = f["video_" + vid]
            feature = torch.tensor(g["feature"][()], dtype=torch.float32).t().unsqueeze(0).to(device)
            logits = model(feature).squeeze(0)
            pred_score = torch.softmax(logits, dim=0)[1].cpu()

            pred_score, pred_selected, pred_summary = eval_utils.select_keyshots(g, pred_score)
            pred_summary = np.asarray(pred_summary)
            true_summary_arr = g["user_summary"][()]
            per_user = np.array([eval_utils.eval_metrics(pred_summary, u) for u in true_summary_arr])
            p, r, f1 = per_user.mean(axis=0).tolist()

            out_dict[vid] = {
                "pred_score": [float(v) for v in pred_score],
                "pred_selected": [int(s) for s in pred_selected],
                "pred_summary": [int(v) for v in pred_summary],
            }
            rows.append([vid, len(g["change_points"][()]), int(pred_summary.sum()),
                         float(np.mean(true_summary_arr)), p, r, f1])
            print("  %-12s segs=%3d 关键帧=%5d  P=%.3f R=%.3f F=%.3f"
                  % (vid, rows[-1][1], rows[-1][2], p, r, f1))

    mean_row = ["平均", "", "", "", np.mean([r_[4] for r_ in rows]),
                np.mean([r_[5] for r_ in rows]), np.mean([r_[6] for r_ in rows])]

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv_out)), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out_dict, f)
    with open(args.csv_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "num_segments", "num_keyshots", "user_summary_mean",
                    "Precision", "Recall", "F-score"])
        for row in rows:
            w.writerow([row[0], row[1], row[2], "%.4f" % row[3], "%.4f" % row[4],
                        "%.4f" % row[5], "%.4f" % row[6]])
        w.writerow([mean_row[0], "", "", "", "%.4f" % mean_row[4],
                    "%.4f" % mean_row[5], "%.4f" % mean_row[6]])

    print("\n== 10 段视频平均 ==  Precision=%.4f  Recall=%.4f  F-score=%.4f"
          % (mean_row[4], mean_row[5], mean_row[6]))
    print("json:", args.json_out)
    print("csv :", args.csv_out)


if __name__ == "__main__":
    main()
