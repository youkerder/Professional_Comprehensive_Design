#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 CIFAR-10 测试集上评测训练好的 AlexNet。

输出
----
* 整体准确率、top-3 准确率
* 每一类的 Precision / Recall / F1（由混淆矩阵推出）
* macro 与 weighted 平均
* 10x10 混淆矩阵：PNG 图 + CSV

指标计算部分（`metrics_from_confusion`）是**纯 numpy** 的，不依赖 TensorFlow，
因此 `--selftest` 在任何环境下都能跑：用手工构造的混淆矩阵校验公式。

用法
----
    python evaluate.py --model ../models/relu-lr0.01_best.h5 --act relu --norm bn
    python evaluate.py --selftest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from config import (CLASS_NAMES, CLASS_NAMES_ZH, DEFAULT_BATCH_SIZE,
                        DEFAULT_NORM, DEFAULT_SEED, EVAL_DIR, IMAGE_SHAPE,
                        NUM_CLASSES, REPORT_FIG_DIR, REPORT_TAB_DIR, VAL_SIZE)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (CLASS_NAMES, CLASS_NAMES_ZH, DEFAULT_BATCH_SIZE,
                        DEFAULT_NORM, DEFAULT_SEED, EVAL_DIR, IMAGE_SHAPE,
                        NUM_CLASSES, REPORT_FIG_DIR, REPORT_TAB_DIR, VAL_SIZE)


# --------------------------------------------------------------------------
# 指标（纯 numpy，可独立测试）
# --------------------------------------------------------------------------

def confusion_matrix(y_true, y_pred, num_classes: int = NUM_CLASSES):
    """按行是真值、列是预测构造混淆矩阵（与 sklearn 的约定一致）。"""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true 与 y_pred 长度不一致：%s vs %s"
                         % (y_true.shape, y_pred.shape))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    np.add.at(cm, (y_true[valid], y_pred[valid]), 1)
    return cm


def metrics_from_confusion(cm, class_names=None):
    """由混淆矩阵算出每类的 Precision / Recall / F1 以及 macro / weighted 平均。

    定义（与常用目标检测/分类评测一致）：

        Precision_c = TP_c / (TP_c + FP_c)      # FP_c = 第 c 列之和 - TP_c
        Recall_c    = TP_c / (TP_c + FN_c)      # FN_c = 第 c 行之和 - TP_c
        F1_c        = 2 * P * R / (P + R)

    macro 平均：各类等权
    weighted 平均：按各类真值数量加权，能反映整体表现
    某类样本数为 0 时该类的 P/R/F1 记为 0，且不计入 macro（避免除零污染平均）。
    """
    cm = np.asarray(cm, dtype=np.int64)
    n = cm.shape[0]
    if cm.shape[0] != cm.shape[1]:
        raise ValueError("混淆矩阵必须是方阵，实际为 %s" % (cm.shape,))
    names = list(class_names) if class_names else ["class%d" % i for i in range(n)]

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)      # 每类真值数
    predicted = cm.sum(axis=0).astype(np.float64)    # 每类被预测数
    total = float(cm.sum())

    per_class = []
    for i in range(n):
        p = tp[i] / predicted[i] if predicted[i] > 0 else 0.0
        r = tp[i] / support[i] if support[i] > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_class.append({
            "index": i, "name": names[i],
            "name_zh": CLASS_NAMES_ZH.get(names[i], ""),
            "support": int(support[i]), "predicted": int(predicted[i]),
            "tp": int(tp[i]),
            "precision": float(p), "recall": float(r), "f1": float(f1),
        })

    present = [c for c in per_class if c["support"] > 0]
    macro = {
        "precision": float(np.mean([c["precision"] for c in present])) if present else 0.0,
        "recall": float(np.mean([c["recall"] for c in present])) if present else 0.0,
        "f1": float(np.mean([c["f1"] for c in present])) if present else 0.0,
    }
    if total > 0:
        weighted = {
            "precision": float(sum(c["precision"] * c["support"] for c in per_class) / total),
            "recall": float(sum(c["recall"] * c["support"] for c in per_class) / total),
            "f1": float(sum(c["f1"] * c["support"] for c in per_class) / total),
        }
    else:
        weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # 整体准确率 = 对角线和 / 总数，与 macro recall 不同（后者是各类等权）
    accuracy = float(tp.sum() / total) if total > 0 else 0.0

    return {
        "num_classes": n,
        "total": int(total),
        "accuracy": accuracy,
        "per_class": per_class,
        "macro": macro,
        "weighted": weighted,
    }


def topk_accuracy(probs, y_true, k: int = 3):
    """top-k 准确率：真值落在概率最高的 k 个类别里就算对。"""
    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int).ravel()
    if probs.ndim != 2 or probs.shape[0] != y_true.shape[0]:
        raise ValueError("概率矩阵与标签数量不匹配：%s vs %s"
                         % (probs.shape, y_true.shape))
    k = max(1, min(k, probs.shape[1]))
    top = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------

def save_confusion_csv(cm, path: Path, class_names=None):
    names = list(class_names) if class_names else ["class%d" % i for i in range(len(cm))]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["真值\\预测"] + names)
        for i, row in enumerate(cm):
            w.writerow([names[i]] + [int(v) for v in row])


def save_confusion_png(cm, path: Path, class_names=None, title="混淆矩阵"):
    """画混淆矩阵热力图。行归一化后显示，便于看每类的漏检/误判比例。"""
    # 用 config 里的统一设置——里面已经配好中文字体。
    # 早期版本这里直接 `import matplotlib.pyplot as plt`，漏了字体设置，
    # 结果图上标题/坐标轴/色标的中文全部变成方框，而程序只在 stderr
    # 留一句 `Glyph xxxxx missing from current font` 警告，很容易漏掉。
    from config import setup_matplotlib
    plt = setup_matplotlib()

    names = list(class_names) if class_names else ["class%d" % i for i in range(len(cm))]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sum == 0, 1, row_sum)).astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            v = cm[i, j]
            if v:
                ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "#1f2937")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="行归一化比例")
    fig.savefig(path)
    plt.close(fig)


def metrics_to_md(metrics, sample_accuracy=None, top3=None, extra=None) -> str:
    lines = []
    lines.append("| 类别 | 真值数 | 预测数 | 正确数 | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in metrics["per_class"]:
        name = c["name"] + ("（%s）" % c["name_zh"] if c["name_zh"] else "")
        lines.append("| %s | %d | %d | %d | %.4f | %.4f | %.4f |"
                     % (name, c["support"], c["predicted"], c["tp"],
                        c["precision"], c["recall"], c["f1"]))
    m = metrics["macro"]
    w = metrics["weighted"]
    lines.append("| **macro 平均** | %d | %d | %d | %.4f | %.4f | %.4f |"
                 % (metrics["total"], metrics["total"],
                    sum(c["tp"] for c in metrics["per_class"]),
                    m["precision"], m["recall"], m["f1"]))
    lines.append("| **weighted 平均** | %d | %d | %d | %.4f | %.4f | %.4f |"
                 % (metrics["total"], metrics["total"],
                    sum(c["tp"] for c in metrics["per_class"]),
                    w["precision"], w["recall"], w["f1"]))
    lines.append("")
    lines.append("整体准确率（accuracy）: **%.4f**" % metrics["accuracy"])
    if sample_accuracy is not None:
        lines.append("")
        lines.append("> 注：上面逐类的 P/R/F1 是 macro/weighted 口径；"
                     "模型 `evaluate()` 报的 accuracy %.4f 与 "
                     "`metrics.accuracy` %.4f 应当一致（差异只可能来自浮点舍入）。"
                     % (sample_accuracy, metrics["accuracy"]))
    if top3 is not None:
        lines.append("")
        lines.append("top-3 准确率: **%.4f**" % top3)
    if extra:
        lines.append("")
        for k, v in extra.items():
            lines.append("- %s: %s" % (k, v))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def selftest() -> int:
    """用手工构造的混淆矩阵校验指标公式。"""
    fails = []

    def check(cond, msg):
        print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
        if not cond:
            fails.append(msg)

    print("evaluate.py 指标计算自检（纯 numpy，不需要 TensorFlow）")
    print("-" * 74)

    # 3 类的构造例子：
    #   class0: 10 个真值，8 个判对，1 个判成 1，1 个判成 2
    #   class1:  5 个真值，5 个判对；但收到 1 个来自 class0 的误判 -> 预测数 6
    #   class2:  6 个真值，4 个判对，2 个判成 0
    # 列和（每类被预测数）= [10, 6, 5]；行和（每类真值数）= [10, 5, 6]
    cm = np.array([[8, 1, 1],
                   [0, 5, 0],
                   [2, 0, 4]], dtype=np.int64)
    m = metrics_from_confusion(cm, ["a", "b", "c"])

    check(m["total"] == 21, "样本总数 21")
    check(abs(m["accuracy"] - 17 / 21) < 1e-9, "整体准确率 = 17/21")

    c0 = m["per_class"][0]
    check(c0["support"] == 10 and c0["predicted"] == 10, "class0 真值数=10 预测数=10")
    check(abs(c0["recall"] - 0.8) < 1e-9, "class0 recall = 8/10 = 0.8")
    check(abs(c0["precision"] - 0.8) < 1e-9, "class0 precision = 8/10 = 0.8")

    c1 = m["per_class"][1]
    # 关键点：class1 自己 5 个全判对，但被 class0 误判进来 1 个，
    # 所以 precision < 1（分母是"被预测为 class1 的总数"= 6，不是真值数 5）
    check(c1["support"] == 5 and c1["predicted"] == 6, "class1 真值=5 但预测=6")
    check(c1["tp"] == 5, "class1 正确数 = 5")
    check(abs(c1["precision"] - 5 / 6) < 1e-9,
          "class1 precision = 5/6（被误判进来 1 个，所以不是 1.0）")
    check(abs(c1["recall"] - 1.0) < 1e-9, "class1 recall = 5/5 = 1.0")
    check(abs(c1["f1"] - 2 * (5 / 6) * 1.0 / ((5 / 6) + 1.0)) < 1e-9, "class1 f1 正确")

    c2 = m["per_class"][2]
    check(abs(c2["recall"] - 4 / 6) < 1e-9, "class2 recall = 4/6")
    check(abs(c2["precision"] - 4 / 5) < 1e-9, "class2 precision = 4/5")
    expect_f1 = 2 * (4 / 5) * (4 / 6) / ((4 / 5) + (4 / 6))
    check(abs(c2["f1"] - expect_f1) < 1e-9, "class2 f1 公式正确")

    check(abs(m["macro"]["recall"] - (0.8 + 1.0 + 4 / 6) / 3) < 1e-9, "macro recall 正确")
    w_prec = (0.8 * 10 + (5 / 6) * 5 + (4 / 5) * 6) / 21
    check(abs(m["weighted"]["precision"] - w_prec) < 1e-9, "weighted precision 正确")

    # 完美预测
    perfect = metrics_from_confusion(np.diag([5, 5, 5]))
    check(abs(perfect["accuracy"] - 1.0) < 1e-9, "完美预测 accuracy = 1")
    check(abs(perfect["macro"]["f1"] - 1.0) < 1e-9, "完美预测 macro f1 = 1")

    # 全错
    wrong = metrics_from_confusion(np.array([[0, 5], [5, 0]]))
    check(abs(wrong["accuracy"]) < 1e-9, "全错 accuracy = 0")

    # 某类完全没出现：不计入 macro，且不除零崩溃
    empty = metrics_from_confusion(np.array([[4, 0], [0, 0]]))
    check(empty["per_class"][1]["recall"] == 0.0, "空类 recall 记为 0（未崩溃）")
    check(abs(empty["macro"]["recall"] - 1.0) < 1e-9,
          "空类不计入 macro（macro recall = 1.0）")

    # confusion_matrix 的朝向：行=真值，列=预测
    yt = [0, 0, 1, 1, 2]
    yp = [0, 1, 1, 1, 0]
    cm2 = confusion_matrix(yt, yp, num_classes=3)
    check(cm2[0, 0] == 1 and cm2[0, 1] == 1, "混淆矩阵行=真值、列=预测")
    check(cm2[1, 1] == 2, "类别 1 两个都判对")
    check(cm2[2, 0] == 1, "类别 2 误判为 0")
    check(int(cm2.sum()) == len(yt), "混淆矩阵元素之和 = 样本数")

    # top-k
    probs = np.array([[0.1, 0.7, 0.2],
                      [0.6, 0.3, 0.1],
                      [0.2, 0.3, 0.5]])
    check(abs(topk_accuracy(probs, [1, 0, 2], k=1) - 1.0) < 1e-9, "top-1 全对")
    check(abs(topk_accuracy(probs, [2, 1, 0], k=1) - 0.0) < 1e-9, "top-1 全错")
    # 逐行看 top-2（概率最高的前 2 个下标）：
    #   第 0 行 [0.1, 0.7, 0.2] -> {1, 2}，真值 2 命中
    #   第 1 行 [0.6, 0.3, 0.1] -> {0, 1}，真值 1 命中
    #   第 2 行 [0.2, 0.3, 0.5] -> {2, 1}，真值 0 未命中
    check(abs(topk_accuracy(probs, [2, 1, 0], k=2) - 2 / 3) < 1e-9, "top-2 命中 2/3")
    check(abs(topk_accuracy(probs, [2, 1, 0], k=3) - 1.0) < 1e-9, "top-3 必然全对")
    check(abs(topk_accuracy(probs, [1, 0, 2], k=1) - 1.0) < 1e-9,
          "top-1 取的是概率最大项（不是最小项）")

    # 长度不一致要报错
    try:
        confusion_matrix([1, 2], [1], num_classes=3)
        check(False, "长度不一致应当报错")
    except ValueError:
        check(True, "长度不一致时抛 ValueError")

    try:
        metrics_from_confusion(np.zeros((2, 3)))
        check(False, "非方阵应当报错")
    except ValueError:
        check(True, "非方阵混淆矩阵抛 ValueError")

    print("-" * 74)
    if fails:
        print("失败 %d 项：" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("全部通过。")
    return 0


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="在 CIFAR-10 测试集上评测 AlexNet")
    ap.add_argument("--model", help="待评测的 .h5 权重")
    ap.add_argument("--act", default="relu", help="与训练时一致的激活函数")
    ap.add_argument("--norm", default=DEFAULT_NORM, help="与训练时一致的规范化方式")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--val-size", type=int, default=VAL_SIZE)
    ap.add_argument("--run-name", default=None, help="结果文件名前缀，默认取权重文件名")
    ap.add_argument("--no-figures", action="store_true", help="不画混淆矩阵图")
    ap.add_argument("--report", action="store_true",
                    help="额外把混淆矩阵与指标表写到 reports/")
    ap.add_argument("--selftest", action="store_true", help="只跑指标自检")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.selftest:
        return selftest()
    if not args.model:
        ap.error("需要 --model（或使用 --selftest）")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    import data as data_mod
    from alexnet import build_alexnet

    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise SystemExit("找不到权重文件：%s" % model_path)
    run = args.run_name or model_path.stem.replace("_best", "")

    print("=" * 74)
    print("实验一 · AlexNet 测试集评测")
    print("=" * 74)
    print("权重       : %s" % model_path)
    print("激活函数   : %s" % args.act)
    print("规范化     : %s" % args.norm)
    print("")

    print("正在加载数据 ...")
    splits = data_mod.load_splits(val_size=args.val_size, seed=args.seed,
                                  use_cache=not args.no_cache, quiet=True)
    x_te, y_te = splits["x_test"], splits["y_test"]
    mean, std = splits["mean"], splits["std"]
    print("  测试集 %d 张" % len(x_te))

    test_ds = data_mod.make_dataset(x_te, y_te, mean, std, args.batch_size,
                                    training=False, seed=args.seed)

    print("正在构建模型并加载权重 ...")
    model = build_alexnet(act=args.act, norm=args.norm)
    model.load_weights(str(model_path))
    print("  参数量 %.2f M" % (model.count_params() / 1e6))

    # 必须先 compile 才能调用 model.evaluate()，否则 Keras 会抛
    # "You must compile your model before training/testing"。
    # 这里只是评测，optimizer 取什么都行（不会用到），
    # 但 loss 与 metrics 要与训练时一致，交叉核对才有意义。
    model.compile(
        optimizer="sgd",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )

    print("正在评估（这也会给出 Keras 自己的 accuracy，用于交叉核对）...")
    t0 = time.time()
    keras_metrics = model.evaluate(test_ds, verbose=0, return_dict=True)
    elapsed = time.time() - t0
    print("  用时 %.1f 秒" % elapsed)

    print("正在逐样本取预测（用于算混淆矩阵与逐类指标）...")
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1).astype(int)
    # predict 按批次输出，顺序与数据集一致（本管线不做 shuffle）
    y_true = np.asarray(y_te).astype(int)
    if len(y_pred) != len(y_true):
        raise SystemExit("预测数量 %d 与测试集 %d 不一致" % (len(y_pred), len(y_true)))

    names = splits.get("class_names") or CLASS_NAMES
    cm = confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
    metrics = metrics_from_confusion(cm, names)
    acc3 = topk_accuracy(probs, y_true, k=3)

    kacc = float(keras_metrics.get("accuracy", float("nan")))
    ktop3 = float(keras_metrics.get("top3", float("nan")))

    print("")
    print("=" * 74)
    print("结果")
    print("=" * 74)
    print("整体准确率        : %.4f" % metrics["accuracy"])
    print("top-3 准确率      : %.4f" % acc3)
    print("macro    P/R/F1   : %.4f / %.4f / %.4f"
          % (metrics["macro"]["precision"], metrics["macro"]["recall"],
             metrics["macro"]["f1"]))
    print("weighted P/R/F1   : %.4f / %.4f / %.4f"
          % (metrics["weighted"]["precision"], metrics["weighted"]["recall"],
             metrics["weighted"]["f1"]))
    print("")
    print("逐类明细：")
    print("  %-12s %6s %7s %6s %10s %8s %8s"
          % ("类别", "真值", "预测", "正确", "Precision", "Recall", "F1"))
    for c in metrics["per_class"]:
        print("  %-12s %6d %7d %6d %10.4f %8.4f %8.4f"
              % (c["name"], c["support"], c["predicted"], c["tp"],
                 c["precision"], c["recall"], c["f1"]))

    # 交叉核对：自己算的准确率应当与 Keras 报的一致
    if abs(kacc - metrics["accuracy"]) > 1e-6:
        print("")
        print("[警告] Keras 报的 accuracy（%.6f）与自行统计的（%.6f）不一致，"
              % (kacc, metrics["accuracy"]))
        print("       可能是 predict/evaluate 的样本顺序或批次处理不同，请检查。")
    else:
        print("")
        print("交叉核对：Keras accuracy %.6f 与自行统计一致。" % kacc)

    # ---------------- 落盘 ----------------
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    cm_csv = EVAL_DIR / ("%s_confusion.csv" % run)
    save_confusion_csv(cm, cm_csv, names)
    print("")
    print("混淆矩阵 CSV : %s" % cm_csv)

    if not args.no_figures:
        try:
            cm_png = EVAL_DIR / ("%s_confusion.png" % run)
            save_confusion_png(cm, cm_png, names,
                               title="混淆矩阵（%s，测试集准确率 %.4f）"
                                     % (run, metrics["accuracy"]))
            print("混淆矩阵图   : %s" % cm_png)
        except Exception as e:
            print("[提示] 画混淆矩阵失败（不影响指标）：%s: %s" % (type(e).__name__, e))

    payload = {
        "run_name": run,
        "model": str(model_path),
        "act": args.act,
        "norm": args.norm,
        "elapsed_sec": elapsed,
        "accuracy": metrics["accuracy"],
        "top3_accuracy": acc3,
        "keras_accuracy": kacc,
        "keras_top3": ktop3,
        "macro": metrics["macro"],
        "weighted": metrics["weighted"],
        "per_class": metrics["per_class"],
        "confusion_matrix": cm.tolist(),
        "class_names": names,
    }
    mjson = EVAL_DIR / ("%s_metrics.json" % run)
    mjson.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("指标 JSON    : %s" % mjson)

    mmd = EVAL_DIR / ("%s_metrics.md" % run)
    mmd.write_text(
        "# %s 测试集评测\n\n" % run
        + metrics_to_md(metrics, sample_accuracy=kacc, top3=acc3,
                        extra={"权重": "`%s`" % model_path.name,
                               "激活函数": args.act, "规范化": args.norm})
        + "\n混淆矩阵：\n\n"
        + _cm_md_table(cm, names),
        encoding="utf-8")
    print("指标 Markdown: %s" % mmd)

    if args.report:
        REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_TAB_DIR.mkdir(parents=True, exist_ok=True)
        if not args.no_figures:
            try:
                save_confusion_png(cm, REPORT_FIG_DIR / ("01b_confusion_%s.png" % run),
                                   names, title="混淆矩阵（%s）" % run)
            except Exception:
                pass
        (REPORT_TAB_DIR / ("01b_eval_%s.md" % run)).write_text(
            mmd.read_text(encoding="utf-8"), encoding="utf-8")
        print("已写入 reports/：01b_confusion_%s.png、01b_eval_%s.md" % (run, run))

    return 0


def _cm_md_table(cm, names) -> str:
    lines = ["| 真值\\预测 | " + " | ".join(names) + " |",
             "|---" + "|---:" * len(names) + "|"]
    for i, row in enumerate(cm):
        lines.append("| %s | " % names[i]
                     + " | ".join(str(int(v)) for v in row) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
