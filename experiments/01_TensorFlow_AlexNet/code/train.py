#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练基础 AlexNet 对 CIFAR-10 分类（单组配置）。

本实验要对比两个自变量（见 config.py）：
    * 激活函数：ReLU / LeakyReLU
    * 学习率  ：0.01 / 0.001
共 4 组，由 `run_all.py` 依次调用本脚本完成。

公平性要求
----------
四组必须**除激活函数与学习率外完全一致**：同一随机种子、同一 train/val 划分、
同一 epoch 数、同一 batch size、同一数据增强，且**默认不启用 EarlyStopping**
（否则各组实际训练的轮数不同，对比就失去意义）。

用法
----
    # 实测速度，决定 epoch 数（不真正训练）
    python train.py --benchmark 5

    # 小样本冒烟验证（约 1 分钟）
    python train.py --epochs 1 --subset 2000

    # 正式训练一组
    python train.py --act relu --lr 0.01 --epochs 15

    # 断点续训
    python train.py --act relu --lr 0.01 --epochs 15 \
        --resume models/relu-lr0.01.h5 --initial-epoch 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

try:
    from config import (ACTIVATIONS, ACT_LABELS, DEFAULT_BATCH_SIZE,
                        DEFAULT_EPOCHS, DEFAULT_NORM, DEFAULT_SEED, EVAL_DIR,
                        IMAGE_SHAPE, KEEP_EVERY, LEARNING_RATES, LOGS_DIR,
                        MODELS_DIR, MOMENTUM, NUM_CLASSES, VAL_SIZE, run_name as mk_run_name)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (ACTIVATIONS, ACT_LABELS, DEFAULT_BATCH_SIZE,
                        DEFAULT_EPOCHS, DEFAULT_NORM, DEFAULT_SEED, EVAL_DIR,
                        IMAGE_SHAPE, KEEP_EVERY, LEARNING_RATES, LOGS_DIR,
                        MODELS_DIR, MOMENTUM, NUM_CLASSES, VAL_SIZE, run_name as mk_run_name)


def _human(sec: float) -> str:
    if sec < 60:
        return "%.1f 秒" % sec
    if sec < 3600:
        return "%.1f 分钟" % (sec / 60)
    return "%.2f 小时" % (sec / 3600)


def count_nonfinite(path: Path):
    """统计 h5 权重里的 NaN/Inf 个数。

    返回 int（NaN 个数，0 表示正常）或 None（**文件打不开/损坏**）。

    必须把"打不开"和"没有 NaN"区分开——实验一踩过这个坑：
    旧实现两种情况都返回 0，于是一个因中断而损坏的权重被判断成"正常"，
    直到拿去推理才报错。
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None

    total = 0

    def visit(_name, obj):
        nonlocal total
        if isinstance(obj, h5py.Dataset) and obj.size:
            a = obj[()]
            total += int(np.isnan(a).sum() + np.isinf(a).sum())

    try:
        with h5py.File(str(path), "r") as f:
            _ = list(f.keys())          # 触发内部结构校验
            f.visititems(visit)
    except Exception:
        return None
    return total


def make_keep_every_n(log_dir: Path, every: int):
    """每 N 个 epoch 无条件保存一次权重的回调。

    理由：`ModelCheckpoint(save_best_only=True)` 只在验证指标改善时保存，
    若指标长期不改善就可能几十轮不落盘，一旦中断这些轮的成果全部丢失。
    实验一就吃过这个亏（电脑重启时正在写入的 checkpoint 损坏）。
    """
    from tensorflow.keras.callbacks import Callback

    class KeepEveryN(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            n = epoch + 1
            if every <= 0 or n % every != 0:
                return
            va = logs.get("val_accuracy")
            name = "ep%03d-val_acc%s.h5" % (n, ("%.4f" % va) if va is not None else "na")
            try:
                self.model.save_weights(str(log_dir / name))
                print("\n[定期保存] %s" % name)
            except Exception as e:
                print("\n[警告] 定期保存失败：%s: %s" % (type(e).__name__, e))

    return KeepEveryN()


def build_argparser(ap=None):
    ap = ap or argparse.ArgumentParser(
        description="训练基础 AlexNet 分类 CIFAR-10（单组）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--act", default="relu", choices=list(ACTIVATIONS),
                    help="激活函数，本实验的对比变量之一（默认 relu）")
    ap.add_argument("--lr", type=float, default=LEARNING_RATES[0],
                    help="学习率，本实验的对比变量之二（默认 %g）" % LEARNING_RATES[0])
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--norm", default=DEFAULT_NORM, choices=["bn", "lrn", "none"],
                    help="规范化方式：bn（默认）/ lrn（更贴近原论文）/ none")
    ap.add_argument("--run-name", default=None, help="默认由 act 与 lr 生成，如 relu-lr0.01")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--val-size", type=int, default=VAL_SIZE)
    ap.add_argument("--subset", type=int, default=0,
                    help="只用前 N 张训练图（0 表示全部）。用于冒烟验证")
    ap.add_argument("--patience", type=int, default=0,
                    help="EarlyStopping 的 patience，0 表示**不启用**（默认）。"
                         "四组对比必须都不启用，否则训练轮数不同、对比不公平")
    ap.add_argument("--keep-every", type=int, default=KEEP_EVERY,
                    help="每 N 轮无条件保存一次权重（默认 %d，0 关闭）" % KEEP_EVERY)
    ap.add_argument("--clipnorm", type=float, default=0.0,
                    help="梯度裁剪范数上限，0 表示不裁剪（默认）")
    ap.add_argument("--benchmark", type=int, default=0, metavar="N",
                    help="只测 N 个 step 的耗时并推算总时间，不真正训练")
    ap.add_argument("--resume", default=None, help="从这个 .h5 权重继续训练")
    ap.add_argument("--initial-epoch", type=int, default=0,
                    help="续训时已完成的轮数")
    ap.add_argument("--no-cache", action="store_true", help="不使用 npz 数据缓存")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    run = args.run_name or mk_run_name(args.act, args.lr)
    log_dir = LOGS_DIR / run
    log_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("实验一 · AlexNet 训练 CIFAR-10")
    print("=" * 74)
    print("run_name     : %s" % run)
    print("激活函数     : %s" % ACT_LABELS.get(args.act, args.act))
    print("学习率       : %g" % args.lr)
    print("规范化       : %s" % args.norm)
    print("epochs       : %d   batch %d   seed %d" % (args.epochs, args.batch_size, args.seed))
    print("早停         : %s" % ("启用 patience=%d" % args.patience if args.patience else "不启用（四组一致）"))
    print("日志目录     : %s" % log_dir)
    print("")

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import (CSVLogger, ModelCheckpoint,
                                            TensorBoard)

    import data as data_mod
    from alexnet import build_alexnet, describe_model

    print("TensorFlow   : %s" % tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("可见 GPU     : %s" % (", ".join(g.name for g in gpus) if gpus else "无（CPU 训练）"))
    print("")

    # ---------------- 数据 ----------------
    print("正在准备数据 ...")
    t0 = time.time()
    splits = data_mod.load_splits(val_size=args.val_size, seed=args.seed,
                                  use_cache=not args.no_cache, quiet=True)
    x_tr, y_tr = splits["x_train"], splits["y_train"]
    x_va, y_va = splits["x_val"], splits["y_val"]
    x_te, y_te = splits["x_test"], splits["y_test"]
    mean, std = splits["mean"], splits["std"]

    if args.subset and args.subset > 0:
        n = min(args.subset, len(x_tr))
        # 用固定 seed 抽样，而不是直接取前 N 张——CIFAR-10 的批次是按类别打散的，
        # 但取前 N 仍可能让某些类别偏少，随机抽样更稳。
        rng = np.random.RandomState(args.seed)
        sel = np.sort(rng.choice(len(x_tr), size=n, replace=False))
        x_tr, y_tr = x_tr[sel], y_tr[sel]
        print("  [小样本模式] 训练集缩减为 %d 张" % n)

    print("  训练 %d 张 / 验证 %d 张 / 测试 %d 张（用时 %.1f 秒）"
          % (len(x_tr), len(x_va), len(x_te), time.time() - t0))

    train_ds = data_mod.make_dataset(x_tr, y_tr, mean, std, args.batch_size,
                                     training=True, seed=args.seed)
    val_ds = data_mod.make_dataset(x_va, y_va, mean, std, args.batch_size,
                                   training=False, seed=args.seed)
    test_ds = data_mod.make_dataset(x_te, y_te, mean, std, args.batch_size,
                                    training=False, seed=args.seed)

    # ---------------- 模型 ----------------
    print("")
    print("正在构建模型 ...")
    model = build_alexnet(act=args.act, norm=args.norm)
    print("  %s" % describe_model(model))

    steps_per_epoch = int(np.ceil(len(x_tr) / args.batch_size))
    val_steps = int(np.ceil(len(x_va) / args.batch_size))

    opt_kwargs = {"learning_rate": args.lr, "momentum": MOMENTUM}
    if args.clipnorm and args.clipnorm > 0:
        opt_kwargs["clipnorm"] = args.clipnorm
    # 注意：这里**不用** SGD 的 decay 参数。Keras 的 decay 会按
    # lr/(1+decay*iter) 让学习率随迭代下降，那样"恒定学习率"的前提就不成立，
    # 会污染本实验对学习率的对比。weight decay 已用 L2 正则实现（见 alexnet.py）。
    optimizer = tf.keras.optimizers.SGD(**opt_kwargs)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )
    print("优化器       : SGD(lr=%g, momentum=%g%s)"
          % (args.lr, MOMENTUM,
             ", clipnorm=%g" % args.clipnorm if args.clipnorm and args.clipnorm > 0 else ""))
    print("weight decay : L2 正则（见 alexnet.py，不影响学习率）")
    print("steps/epoch  : %d   val_steps %d" % (steps_per_epoch, val_steps))

    # ---------------- 续训 ----------------
    if args.resume:
        rp = Path(args.resume)
        if not rp.is_file():
            raise SystemExit("找不到续训权重：%s" % rp)
        bad = count_nonfinite(rp)
        if bad is None:
            raise SystemExit(
                "续训权重文件损坏、无法打开：%s\n"
                "  （常见于写入过程中进程被中断，请在 logs/ 里另挑一个 checkpoint）" % rp)
        if bad:
            raise SystemExit("续训权重里有 %d 个 NaN/Inf，不能用于续训：%s" % (bad, rp))
        model.load_weights(str(rp))
        print("续训权重     : %s（校验通过）" % rp.name)
        print("起始 epoch   : %d -> %d" % (args.initial_epoch, args.epochs))

    # ---------------- 测速模式 ----------------
    if args.benchmark and args.benchmark > 0:
        print("")
        print("=" * 74)
        print("测速模式：跑 %d 个 step，推算总时间" % args.benchmark)
        print("=" * 74)
        t = time.time()
        model.fit(train_ds, steps_per_epoch=args.benchmark, epochs=1, verbose=0)
        dt = (time.time() - t) / args.benchmark
        per_epoch = dt * steps_per_epoch
        print("  单 step        : %.3f 秒" % dt)
        print("  单 epoch       : %.1f 秒（%.1f 分钟，%d steps）"
              % (per_epoch, per_epoch / 60, steps_per_epoch))
        print("  每组 %d epoch  : %s" % (args.epochs, _human(per_epoch * args.epochs)))
        print("  4 组合计       : %s" % _human(per_epoch * args.epochs * 4))
        if args.subset:
            full = per_epoch * 50000 / max(1, len(x_tr))
            print("  （按全量 %d 张推算：单 epoch %.1f 分钟，4 组 %s）"
                  % (50000 - args.val_size, full / 60,
                     _human(full * args.epochs * 4)))
        return 0

    # ---------------- 回调 ----------------
    ckpt_path = str(log_dir / "ep{epoch:03d}-valacc{val_accuracy:.4f}.h5")
    callbacks = [
        TensorBoard(log_dir=str(log_dir), histogram_freq=0, write_graph=False),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_accuracy",
                        mode="max", save_weights_only=True,
                        save_best_only=True, save_freq="epoch",
                        verbose=1),
        CSVLogger(str(log_dir / "history.csv"), append=args.initial_epoch > 0),
    ]
    if args.keep_every and args.keep_every > 0:
        callbacks.append(make_keep_every_n(log_dir, args.keep_every))
    if args.patience and args.patience > 0:
        from tensorflow.keras.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(monitor="val_accuracy", mode="max",
                                       patience=args.patience,
                                       restore_best_weights=False))
        print("")
        print("[注意] 已启用 EarlyStopping。四组对比时应当都关闭，否则训练轮数不同。")

    # ---------------- 训练 ----------------
    print("")
    print("开始训练 ...（Ctrl+C 可中断，已保存的 checkpoint 不会丢）")
    t_start = time.time()
    try:
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps,
            epochs=args.epochs,
            initial_epoch=args.initial_epoch,
            callbacks=callbacks,
            verbose=1,
        )
    except KeyboardInterrupt:
        print("")
        print("[中断] 收到 Ctrl+C，保存当前权重 ...")
        history = None

    elapsed = time.time() - t_start
    print("")
    print("=" * 74)
    print("训练结束，用时 %s" % _human(elapsed))

    # ---------------- 保存 ----------------
    final_path = MODELS_DIR / ("%s.h5" % run)
    model.save_weights(str(final_path))
    bad = count_nonfinite(final_path)
    if bad is None:
        print("[错误] 保存的权重文件损坏、无法打开：%s" % final_path)
        print("       可用权重仍在：%s" % log_dir)
    elif bad:
        print("[错误] 保存的权重里有 %d 个 NaN/Inf 参数，这个模型不可用。" % bad)
        print("       学习率 %g 下训练发散——这本身是学习率对比的一个有效结果，" % args.lr)
        print("       请在报告里如实记录。可用 checkpoint 仍在：%s" % log_dir)
    else:
        print("权重检查 : 无 NaN/Inf，正常")
    print("最终权重 : %s" % final_path)

    # 归档验证集最优的 checkpoint（用于评测时优先选它）
    best = None
    best_acc = -1.0
    for p in log_dir.glob("ep*-valacc*.h5"):
        try:
            acc = float(p.stem.split("valacc")[-1])
        except ValueError:
            continue
        if acc > best_acc:
            best_acc, best = acc, p
    if best is not None:
        print("验证集最优 checkpoint : %s（val_acc=%.4f）" % (best.name, best_acc))
        shutil.copy2(best, MODELS_DIR / ("%s_best.h5" % run))
        print("  已归档 : %s" % (MODELS_DIR / ("%s_best.h5" % run)))

    # ---------------- 保存训练历史 ----------------
    hist_path = EVAL_DIR / ("%s_history.json" % run)
    if history is not None:
        h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        h["_meta"] = {
            "run_name": run, "act": args.act, "lr": args.lr, "norm": args.norm,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "seed": args.seed, "subset": args.subset,
            "val_size": args.val_size, "initial_epoch": args.initial_epoch,
            "train_size": int(len(x_tr)), "val_size_actual": int(len(x_va)),
            "elapsed_sec": float(elapsed),
            "best_val_accuracy": best_acc if best_acc >= 0 else None,
        }
        hist_path.write_text(json.dumps(h, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print("训练历史 : %s" % hist_path)

    # 快速用测试集报一个准确率，便于立刻判断这一组是否正常
    print("")
    print("正在测试集上快速评估 ...")
    te = model.evaluate(test_ds, verbose=0, return_dict=True)
    print("  测试集 accuracy : %.4f" % te.get("accuracy", float("nan")))
    print("  测试集 top3     : %.4f" % te.get("top3", float("nan")))
    print("  （完整指标请运行 evaluate.py）")
    print("")
    print("下一步：")
    print("  python evaluate.py --model \"%s\"" % final_path)
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
