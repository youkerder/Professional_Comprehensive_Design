#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用**合成数据**跑通 train.py → evaluate.py → compare.py 的完整命令行链路。

为什么需要它
------------
`test_pipeline.py` 验证的是各个模块的函数级正确性，但**不覆盖命令行入口**：
argparse 参数名、回调链、checkpoint 命名、history.json 的字段、
evaluate 读取权重、compare 汇总……这些只有在真正调用 `main()` 时才会暴露。

真正训练要 3.5 小时、下载要几小时，等那么久才发现"某个参数名写错了"代价太大。
本脚本把 `data.load_splits` 换成合成数据，几秒钟就能把整条链路跑一遍。

它**不会碰 datasets/**，只写 `models/`、`logs/`、`outputs/` 下名字以
`smoketest-` 开头的产物，跑完会自动清理（加 `--keep` 可保留查看）。

用法
----
    python test_cli.py              # 跑完自动清理
    python test_cli.py --keep       # 保留产物以便检查
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import config  # noqa: E402

STATE = {"pass": 0, "fail": 0, "fails": []}
PREFIX = "smoketest"


def check(cond, msg, detail=None):
    if cond:
        STATE["pass"] += 1
        print("  [ OK ] %s" % msg)
    else:
        STATE["fail"] += 1
        STATE["fails"].append(msg)
        print("  [FAIL] %s" % msg)
        if detail:
            print("         %s" % detail)
    return bool(cond)


def make_fake_splits(n_train=256, n_val=64, n_test=64, seed=0):
    """造一份"可学"的合成数据：类别由图像平均亮度决定。

    这样模型几步内就能把损失降下来，便于确认训练链路真的在工作
    （如果损失完全不降，说明接线有问题）。
    """
    rng = np.random.RandomState(seed)
    c = config.NUM_CLASSES
    shape = config.IMAGE_SHAPE

    def gen(n):
        y = rng.randint(0, c, size=n).astype(np.uint8)
        x = np.zeros((n,) + shape, dtype=np.uint8)
        base = 20 + y * 20            # 每个类别一个基准亮度
        noise = rng.randint(-8, 9, size=(n,) + shape)
        x[:] = np.clip(base[:, None, None, None] + noise, 0, 255).astype(np.uint8)
        return x, y

    x_tr, y_tr = gen(n_train)
    x_va, y_va = gen(n_val)
    x_te, y_te = gen(n_test)

    # 复用真实实现算归一化统计，保证和线上一致
    import data as dm
    mean, std = dm.compute_norm_stats(x_tr)

    return {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_va, "y_val": y_va,
        "x_test": x_te, "y_test": y_te,
        "mean": mean, "std": std,
        "class_names": list(config.CLASS_NAMES),
    }


def cleanup(keep: bool):
    """删掉本次产生的 smoketest-* 产物。"""
    if keep:
        print("  已保留产物（--keep）")
        return
    removed = []
    for d in (config.MODELS_DIR, config.EVAL_DIR):
        for p in d.glob("%s*" % PREFIX):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            removed.append(p.name)
    for d in config.LOGS_DIR.glob("%s*" % PREFIX):
        shutil.rmtree(d, ignore_errors=True)
        removed.append(d.name)
    for name in ("summary.csv", "summary.md",
                 "curves_val_acc.png", "curves_loss.png", "test_accuracy.png"):
        p = config.COMPARE_DIR / name
        if p.is_file():
            p.unlink()
            removed.append(name)
    print("  已清理 %d 个临时产物" % len(removed))


def main():
    ap = argparse.ArgumentParser(description="用合成数据跑通命令行链路")
    ap.add_argument("--keep", action="store_true", help="保留产物以便检查")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 74)
    print("命令行链路冒烟测试（合成数据，不需要 CIFAR-10，不碰 datasets/）")
    print("=" * 74)

    # ---------------- 打桩：把真实数据换成合成数据 ----------------
    import data as dm
    real_load = dm.load_splits
    fake = make_fake_splits()
    dm.load_splits = lambda **kw: fake

    # compare.py 从 config 取四组；这里只留一组冒烟用的
    real_all_runs = config.all_runs
    smoke_run = ("%s-relu" % PREFIX, "relu", 0.01)
    config.all_runs = lambda: [smoke_run]

    import compare
    compare.all_runs = config.all_runs

    run = smoke_run[0]
    rc_all = 0
    try:
        # ================= 1. train.py =================
        print("")
        print("-" * 74)
        print("1. train.py（%d epoch，合成数据）" % args.epochs)
        print("-" * 74)
        import train
        argv = ["--act", "relu", "--lr", "0.01",
                "--epochs", "%d" % args.epochs,
                "--run-name", run, "--keep-every", "0",
                "--batch-size", "64", "--val-size", "64", "--seed", "0"]
        rc = train.main(argv)
        rc_all = rc_all or rc
        check(rc == 0, "train.main 返回 0（实际 %s）" % rc)

        weight = config.MODELS_DIR / ("%s.h5" % run)
        check(weight.is_file(), "生成了权重 %s" % weight.name)

        hist_path = config.EVAL_DIR / ("%s_history.json" % run)
        check(hist_path.is_file(), "生成了训练历史 JSON")
        if hist_path.is_file():
            h = json.loads(hist_path.read_text(encoding="utf-8"))
            check("loss" in h and h["loss"], "history.json 里有 loss 曲线")
            check("val_accuracy" in h, "history.json 里有 val_accuracy")
            check(h["loss"][-1] < h["loss"][0],
                  "训练损失下降（%.4f -> %.4f）" % (h["loss"][0], h["loss"][-1]))
            meta = h.get("_meta", {})
            check(meta.get("act") == "relu" and meta.get("lr") == 0.01,
                  "_meta 记录了激活函数与学习率")
            check(meta.get("train_size") == 256, "_meta 记录了训练集大小")

        ckpts = list((config.LOGS_DIR / run).glob("ep*-valacc*.h5"))
        check(len(ckpts) >= 1, "生成了验证集 checkpoint（%d 个）" % len(ckpts))
        check((config.LOGS_DIR / run / "history.csv").is_file(),
              "生成了 CSVLogger 的 history.csv")
        # 注意：Keras 的 TensorBoard 回调会把事件写到 log_dir/train/ 与
        # log_dir/validation/ 两个**子目录**里，不是直接放在 log_dir 下，
        # 所以这里必须递归找。
        tb_files = list((config.LOGS_DIR / run).rglob("events.out.tfevents.*"))
        check(len(tb_files) >= 1,
              "生成了 TensorBoard 事件文件（%d 个，位于子目录）" % len(tb_files))

        # ================= 2. evaluate.py =================
        print("")
        print("-" * 74)
        print("2. evaluate.py（用上一步的权重在合成测试集上评测）")
        print("-" * 74)
        import evaluate
        rc = evaluate.main(["--model", str(weight), "--act", "relu",
                            "--run-name", run, "--val-size", "64",
                            "--batch-size", "64", "--seed", "0",
                            "--no-figures"])
        rc_all = rc_all or rc
        check(rc == 0, "evaluate.main 返回 0（实际 %s）" % rc)

        mpath = config.EVAL_DIR / ("%s_metrics.json" % run)
        check(mpath.is_file(), "生成了指标 JSON")
        if mpath.is_file():
            m = json.loads(mpath.read_text(encoding="utf-8"))
            check(abs(m["accuracy"] - m["keras_accuracy"]) < 1e-6,
                  "自算准确率与 Keras 报的一致（%.4f）" % m["accuracy"])
            check(len(m["per_class"]) == config.NUM_CLASSES,
                  "逐类指标覆盖全部 %d 类" % config.NUM_CLASSES)
            check(len(m["confusion_matrix"]) == config.NUM_CLASSES,
                  "混淆矩阵为 %dx%d" % (config.NUM_CLASSES, config.NUM_CLASSES))
            cm = np.array(m["confusion_matrix"])
            check(int(cm.sum()) == 64, "混淆矩阵元素之和 = 测试集样本数（%d）" % cm.sum())
        check((config.EVAL_DIR / ("%s_metrics.md" % run)).is_file(),
              "生成了指标 Markdown")
        check((config.EVAL_DIR / ("%s_confusion.csv" % run)).is_file(),
              "生成了混淆矩阵 CSV")

        # ================= 3. compare.py =================
        print("")
        print("-" * 74)
        print("3. compare.py（汇总）")
        print("-" * 74)
        rc = compare.main(["--no-report"])
        check(rc == 0, "compare.main 返回 0（实际 %s，1 表示有组缺结果）" % rc)

        csv_path = config.COMPARE_DIR / "summary.csv"
        check(csv_path.is_file(), "生成了对比表 CSV")
        md_path = config.COMPARE_DIR / "summary.md"
        check(md_path.is_file(), "生成了对比表 Markdown")
        if md_path.is_file():
            txt = md_path.read_text(encoding="utf-8")
            check("ReLU" in txt, "对比表里有激活函数列")
            check("%d" % args.epochs in txt or "0.01" in txt, "对比表里填入了数据")
        for fig in ("curves_val_acc.png", "curves_loss.png", "test_accuracy.png"):
            check((config.COMPARE_DIR / fig).is_file(), "生成了对比图 %s" % fig)

        # ================= 4. predict.py =================
        print("")
        print("-" * 74)
        print("4. predict.py（用上一步的权重复核推理链路）")
        print("-" * 74)
        import predict
        # 造一张临时图片
        from PIL import Image
        tmp_img = config.PREDICT_DIR / "_smoketest_input.png"
        config.PREDICT_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(fake["x_test"][0]).resize((96, 96)).save(tmp_img)

        rc = predict.main(["--image", str(tmp_img), "--model", str(weight),
                           "--act", "relu", "--val-size", "64",
                           "--save-dir", str(config.PREDICT_DIR)])
        rc_all = rc_all or rc
        check(rc == 0, "predict.main 返回 0（实际 %s）" % rc)
        check((config.PREDICT_DIR / "_smoketest_input_pred.jpg").is_file(),
              "生成了带预测结果的图片")

        if not args.keep:
            tmp_img.unlink(missing_ok=True)
            (config.PREDICT_DIR / "_smoketest_input_pred.jpg").unlink(missing_ok=True)

    finally:
        # 恢复被打桩的函数，避免影响同进程内的其它逻辑
        dm.load_splits = real_load
        config.all_runs = real_all_runs
        compare.all_runs = real_all_runs

    print("")
    print("-" * 74)
    print("清理")
    print("-" * 74)
    cleanup(args.keep)

    print("")
    print("=" * 74)
    print("冒烟测试汇总：通过 %d 项，失败 %d 项" % (STATE["pass"], STATE["fail"]))
    if STATE["fails"]:
        print("失败清单：")
        for f in STATE["fails"]:
            print("  - %s" % f)
    print("=" * 74)
    return 1 if STATE["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
