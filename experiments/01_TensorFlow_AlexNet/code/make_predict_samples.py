#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成报告用图 1-5「预测结果示例」，顺带演示题目要求的"输入一张图像输出类别"。

做两件事
--------
1. **本数据集样本**：从 CIFAR-10 测试集里挑若干张（既有判对的、也有判错的），
   画出预测结果，拼成网格图。
2. **跨数据集演示**：把实验一（目标检测）的 `datasets/cat_face/JPEGImages/` 里的
   真实猫照片喂给这个分类模型，看它是否也判为 `cat`。
   这批图**完全没参与训练**，是真正的域外数据，能反映模型的泛化能力。

输出
----
    reports/figures/01b_predict_samples.png     报告用图
    experiments/01_TensorFlow_AlexNet/outputs/predict/  单张结果图与导出的样例

用法
----
    python make_predict_samples.py
    python make_predict_samples.py --model ../models/relu-lr0.01_best.h5 --act relu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from config import (CAT_FACE_JPEG_DIR, CLASS_NAMES, CLASS_NAMES_ZH,
                        DEFAULT_NORM, DEFAULT_SEED, IMAGE_SHAPE, MODELS_DIR,
                        PREDICT_DIR, REPORT_FIG_DIR, VAL_SIZE, setup_matplotlib)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (CAT_FACE_JPEG_DIR, CLASS_NAMES, CLASS_NAMES_ZH,
                        DEFAULT_NORM, DEFAULT_SEED, IMAGE_SHAPE, MODELS_DIR,
                        PREDICT_DIR, REPORT_FIG_DIR, VAL_SIZE, setup_matplotlib)

SAMPLE_DIR = PREDICT_DIR / "samples"


def pick_default_model():
    cands = sorted(MODELS_DIR.glob("*_best.h5"))
    return cands[0] if cands else None


def load_cifar_test():
    import data as dm
    splits = dm.load_splits(val_size=VAL_SIZE, seed=DEFAULT_SEED, quiet=True)
    return splits["x_test"], splits["y_test"], splits["mean"], splits["std"]


def choose_samples(model, x, y, mean, std, n_correct=6, n_wrong=2, seed=0):
    """挑若干张测试图：既有判对的，也有判错的，让示例图更有信息量。"""
    from alexnet import build_alexnet  # noqa: F401  （仅用于类型提示）

    import predict as pd

    correct, wrong = [], []
    # 分批预测，避免一次吃太多内存
    B = 512
    preds = []
    for i in range(0, len(x), B):
        batch = np.stack([pd.preprocess(im, mean, std)[0] for im in x[i:i + B]])
        preds.append(model.predict(batch, verbose=0))
    probs_all = np.concatenate(preds, axis=0)
    pred = np.argmax(probs_all, axis=1)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(x))
    for i in order:
        if pred[i] == y[i]:
            if len(correct) < n_correct:
                correct.append(i)
        else:
            if len(wrong) < n_wrong:
                wrong.append(i)
        if len(correct) >= n_correct and len(wrong) >= n_wrong:
            break
    return [(i, probs_all[i]) for i in correct + wrong]


def main():
    ap = argparse.ArgumentParser(description="生成预测结果示例图（图 1-5）")
    ap.add_argument("--model", default=None)
    ap.add_argument("--act", default="relu")
    ap.add_argument("--norm", default=DEFAULT_NORM)
    ap.add_argument("--n-correct", type=int, default=6)
    ap.add_argument("--n-wrong", type=int, default=2)
    ap.add_argument("--n-cat", type=int, default=4,
                    help="跨数据集演示用几张猫照片")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import predict as pd
    from alexnet import build_alexnet

    model_path = Path(args.model) if args.model else pick_default_model()
    if model_path is None or not Path(model_path).is_file():
        raise SystemExit("找不到权重。先训练，或用 --model 指定。")

    print("=" * 74)
    print("生成预测结果示例图")
    print("=" * 74)
    print("权重 : %s" % model_path)
    print("")

    print("正在加载模型与测试集 ...")
    model = build_alexnet(act=args.act, norm=args.norm)
    model.load_weights(str(model_path))
    x_test, y_test, mean, std = load_cifar_test()
    print("  测试集 %d 张，参数量 %.2f M" % (len(x_test), model.count_params() / 1e6))

    print("正在挑样例（判对 %d 张 + 判错 %d 张）..."
          % (args.n_correct, args.n_wrong))
    picked = choose_samples(model, x_test, y_test, mean, std,
                            args.n_correct, args.n_wrong)
    print("  挑出 %d 张" % len(picked))

    # ---------------- 跨数据集：真实猫照片 ----------------
    cat_items = []
    if CAT_FACE_JPEG_DIR.is_dir():
        files = sorted(CAT_FACE_JPEG_DIR.glob("*.jpg"))[:args.n_cat]
        for f in files:
            try:
                img = pd.load_image_array(f)
                probs = model.predict(pd.preprocess(img, mean, std), verbose=0)[0]
                cat_items.append((f.name, img, probs))
            except Exception as e:
                print("  [跳过] %s：%s" % (f.name, e))
        print("  跨数据集演示：取到 %d 张猫照片（未参与训练）" % len(cat_items))
    else:
        print("  [提示] 找不到 %s，跳过跨数据集演示" % CAT_FACE_JPEG_DIR)

    # ---------------- 画图 ----------------
    print("")
    print("正在画图 ...")
    plt = setup_matplotlib()
    from PIL import Image

    cols = 4
    rows = 1 + (1 if cat_items else 0)
    n_main = len(picked)
    n_main_rows = int(np.ceil(n_main / cols))
    rows = n_main_rows + (1 if cat_items else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")

    def show(ax, img, probs, title_color_default=True):
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        ax.imshow(img)
        ax.axis("off")
        zh = CLASS_NAMES_ZH.get(CLASS_NAMES[pred], "")
        ax.set_title("%s（%s）  %.3f" % (CLASS_NAMES[pred], zh, conf),
                     fontsize=10, color="#b91c1c" if not title_color_default else "#1f2937")

    # 第一组：CIFAR-10 测试集
    for k, (idx, probs) in enumerate(picked):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        gt = int(y_test[idx])
        ok = int(np.argmax(probs)) == gt
        show(ax, x_test[idx], probs, title_color_default=ok)
        # 标题上加真实类别，判错的用红字醒目提示
        zh = CLASS_NAMES_ZH.get(CLASS_NAMES[int(np.argmax(probs))], "")
        gt_zh = CLASS_NAMES_ZH.get(CLASS_NAMES[gt], "")
        ax.set_title("预测 %s（%s）%.3f\n真实 %s（%s）"
                     % (CLASS_NAMES[int(np.argmax(probs))], zh,
                        float(probs[int(np.argmax(probs))]),
                        CLASS_NAMES[gt], gt_zh),
                     fontsize=8.5, color="#1f2937" if ok else "#b91c1c")
        ax.axis("off")

    # 第二组：跨数据集的真实猫照片
    if cat_items:
        base = n_main_rows * cols
        for k, (name, img, probs) in enumerate(cat_items):
            r, c = divmod(base + k, cols)
            ax = axes[r][c]
            pred = int(np.argmax(probs))
            zh = CLASS_NAMES_ZH.get(CLASS_NAMES[pred], "")
            ax.imshow(img)
            ax.axis("off")
            ax.set_title("预测 %s（%s）%.3f\n%s（未参与训练）"
                         % (CLASS_NAMES[pred], zh, float(probs[pred]), name),
                         fontsize=8.5,
                         color="#1f2937" if CLASS_NAMES[pred] == "cat" else "#b45309")

    fig.suptitle("图 1-5 预测结果示例　（第一行起：CIFAR-10 测试集；"
                 "末行：跨数据集真实猫照片）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_FIG_DIR / "01b_predict_samples.png"
    fig.savefig(out)
    plt.close(fig)
    print("  已写入 %s" % out)

    # ---------------- 同时导出单张样例，便于让人自己跑 predict.py ----------------
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for k, (idx, probs) in enumerate(picked):
        Image.fromarray(x_test[idx]).resize((128, 128), Image.NEAREST).save(
            SAMPLE_DIR / ("cifar_test_%02d_gt%s.png" % (k, CLASS_NAMES[int(y_test[idx])])))
    print("  单张样例已导出到 %s（可直接喂给 predict.py）" % SAMPLE_DIR)

    # ---------------- 跨数据集结论 ----------------
    if cat_items:
        n_cat = sum(1 for _, _, p in cat_items if int(np.argmax(p)) == CLASS_NAMES.index("cat"))
        print("")
        print("跨数据集结果：%d / %d 张真实猫照片被判为 cat"
              % (n_cat, len(cat_items)))
        for name, _, p in cat_items:
            pred = int(np.argmax(p))
            print("  %-10s -> %-12s %.4f" % (name, CLASS_NAMES[pred], float(p[pred])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
