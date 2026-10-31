#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一（TensorFlow / AlexNet 图像分类）一键流程。

按顺序执行：
    1. 数据准备（下载 CIFAR-10、解析、缓存）
    2. 训练四组对比（2 激活函数 × 2 学习率）
    3. 每组在测试集上评测
    4. 汇总出对比表与对比图

用法
----
    # 只检查环境与数据，不训练
    python run_all.py --dry-run

    # 正式跑全部（约 3.5 小时，CPU 实测）
    python run_all.py --epochs 15

    # 小样本冒烟验证（几分钟，确认链路通）
    python run_all.py --epochs 1 --subset 2000

    # 只重跑汇总（训练与评测都已完成时）
    python run_all.py --skip-train --skip-eval

    # 只跑某一组
    python run_all.py --only relu-lr0.01
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# 本脚本在实验根目录（不在 code/ 下），所以要显式把 code/ 加进 sys.path，
# 否则 `from config import ...` 会失败。
CODE_DIR = Path(__file__).resolve().parent / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import (ACT_LABELS, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS,  # noqa: E402
                    DEFAULT_NORM, DEFAULT_SEED, EXP_DIR, MODELS_DIR,
                    all_runs)


def human(sec: float) -> str:
    if sec < 60:
        return "%.1f 秒" % sec
    if sec < 3600:
        return "%.1f 分钟" % (sec / 60)
    return "%.2f 小时" % (sec / 3600)


def run_step(title: str, cmd, cwd=CODE_DIR):
    """执行一步并打印分隔标题。返回 (退出码, 用时秒)。"""
    print("")
    print("=" * 74)
    print(title)
    print("=" * 74)
    print("$ %s" % " ".join(str(c) for c in cmd))
    print("")
    t0 = time.time()
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd))
    dt = time.time() - t0
    print("")
    print("-> 退出码 %d，用时 %s" % (rc, human(dt)))
    return rc, dt


def main():
    ap = argparse.ArgumentParser(
        description="实验一 AlexNet 图像分类 · 一键流程",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--norm", default=DEFAULT_NORM, choices=["bn", "lrn", "none"])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--subset", type=int, default=0,
                    help="只用前 N 张训练图（冒烟验证用）")
    ap.add_argument("--only", default=None,
                    help="只跑指定 run_name（如 relu-lr0.01）")
    ap.add_argument("--skip-data", action="store_true", help="跳过数据准备")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练")
    ap.add_argument("--skip-eval", action="store_true", help="跳过评测")
    ap.add_argument("--dry-run", action="store_true",
                    help="只检查环境与数据，不做任何训练/下载")
    ap.add_argument("--tf-python", default=None,
                    help="跑训练/推理用的 python（需要 TensorFlow）；默认用当前解释器")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    tf_py = Path(args.tf_python) if args.tf_python else Path(sys.executable)

    print("=" * 74)
    print("实验一（TensorFlow 编程实验）· AlexNet 图像分类 · 全流程")
    print("=" * 74)
    print("实验目录   : %s" % EXP_DIR)
    print("TF python  : %s  %s" % (tf_py, "存在" if tf_py.is_file() else "!! 不存在"))
    print("计划       : %s" % ("仅检查（--dry-run）" if args.dry_run else
                              "数据 -> 训练 4 组 -> 评测 -> 汇总"))
    print("epochs     : %d   batch %d   norm %s   seed %d"
          % (args.epochs, args.batch_size, args.norm, args.seed))
    if args.subset:
        print("小样本模式 : 训练集只用 %d 张" % args.subset)

    runs = all_runs()
    if args.only:
        runs = [r for r in runs if r[0] == args.only]
        if not runs:
            raise SystemExit("--only 指定的 run_name 不存在。可选：%s"
                             % ", ".join(r[0] for r in all_runs()))
    print("")
    print("四组配置：")
    for name, act, lr in runs:
        print("  %-20s %-11s lr=%g" % (name, ACT_LABELS.get(act, act), lr))

    if not tf_py.is_file():
        raise SystemExit("指定的 python 不存在：%s\n"
                         "  请用 --tf-python 指向装了 TensorFlow 的解释器，例如\n"
                         "  envs\\pcd01_tf\\Scripts\\python.exe" % tf_py)

    t_all = time.time()

    # ---------------- 1. 数据 ----------------
    if args.dry_run:
        rc, _ = run_step("步骤 1/4：数据状态检查", [tf_py, CODE_DIR / "data.py"])
        print("")
        print("=" * 74)
        print("--dry-run 完成。要正式开始请去掉 --dry-run")
        print("=" * 74)
        return rc

    if not args.skip_data:
        rc, _ = run_step("步骤 1/4：准备数据（首次需下载 CIFAR-10）",
                         [tf_py, CODE_DIR / "data.py", "--download"])
        if rc != 0:
            print("")
            print("[终止] 数据准备失败。若是网络问题，重新运行本命令会**从断点续传**。")
            return rc

    # ---------------- 2. 训练四组 ----------------
    if not args.skip_train:
        for i, (name, act, lr) in enumerate(runs, 1):
            cmd = [tf_py, CODE_DIR / "train.py",
                   "--act", act, "--lr", "%g" % lr,
                   "--epochs", "%d" % args.epochs,
                   "--batch-size", "%d" % args.batch_size,
                   "--norm", args.norm, "--seed", "%d" % args.seed]
            if args.subset:
                cmd += ["--subset", "%d" % args.subset]
            rc, _ = run_step("步骤 2/4：训练第 %d/%d 组 —— %s（%s, lr=%g）"
                             % (i, len(runs), name, ACT_LABELS.get(act, act), lr), cmd)
            if rc != 0:
                print("")
                print("[警告] 第 %d 组训练返回非 0（%d）。" % (i, rc))
                print("       若是因为学习率过大导致 NaN 发散，这本身是学习率对比的")
                print("       有效结果，可以继续跑下一组；其余情况请检查上面的报错。")

    # ---------------- 3. 评测 ----------------
    if not args.skip_eval:
        for i, (name, act, lr) in enumerate(runs, 1):
            # 优先用验证集最优的权重；没有就退回最终权重
            best = MODELS_DIR / ("%s_best.h5" % name)
            final = MODELS_DIR / ("%s.h5" % name)
            weight = best if best.is_file() else final
            if not weight.is_file():
                print("")
                print("[跳过] %s 没有权重文件，无法评测" % name)
                continue
            cmd = [tf_py, CODE_DIR / "evaluate.py",
                   "--model", weight, "--act", act, "--norm", args.norm,
                   "--run-name", name, "--seed", "%d" % args.seed, "--report"]
            rc, _ = run_step("步骤 3/4：评测第 %d/%d 组 —— %s" % (i, len(runs), name), cmd)
            if rc != 0:
                print("[警告] %s 评测返回非 0（%d）" % (name, rc))

    # ---------------- 4. 汇总 ----------------
    rc, _ = run_step("步骤 4/4：汇总对比", [tf_py, CODE_DIR / "compare.py"])

    print("")
    print("=" * 74)
    print("全部完成，总用时 %s" % human(time.time() - t_all))
    print("=" * 74)
    print("模型     : %s" % MODELS_DIR)
    print("对比结果 : %s" % (EXP_DIR / "outputs" / "compare"))
    print("报告表格 : reports/tables/01b_compare.md")
    print("报告插图 : reports/figures/01b_*.png")
    print("")
    print("接下来：把 outputs/compare/summary.md 的数字填进")
    print("        reports/01b_AlexNet图像分类_实验报告.md 的第六节。")
    return rc


if __name__ == "__main__":
    sys.exit(main())
