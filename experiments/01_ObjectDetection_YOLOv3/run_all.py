#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 01 全流程一键执行：数据检查 → 训练 → 检测 → 评测 → 生成报告图表。

你只需要记住这一条命令（在仓库根目录下运行）：

    python experiments/01_ObjectDetection_YOLOv3/run_all.py

常用变体：

    # 先看看数据和环境有没有问题，不真的训练
    python experiments/01_ObjectDetection_YOLOv3/run_all.py --dry-run

    # 正式训练：200 epoch（约 7 小时，建议睡前跑）
    python experiments/01_ObjectDetection_YOLOv3/run_all.py --epochs 200

    # 已经有训练好的模型了，只重跑检测+评测+出图
    python experiments/01_ObjectDetection_YOLOv3/run_all.py --skip-train

    # 用指定的权重
    python experiments/01_ObjectDetection_YOLOv3/run_all.py --skip-train --weights ../models/xxx.h5

    # 不重新生成索引/标注（数据没变时可以跳过，省时间）
    python experiments/01_ObjectDetection_YOLOv3/run_all.py --skip-data

说明：
  * 训练与检测需要 TensorFlow 环境（`envs/pcd01_tf`），本脚本会自动用那个 venv 的
    python 来跑，不需要你手动激活环境。
  * 其余步骤（生成索引、评测、画图）不依赖 TensorFlow，用当前 python 即可。
  * 每步的命令都会打印出来，失败会立刻停下并说明是哪一步。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/01_ObjectDetection_YOLOv3
REPO_ROOT = HERE.parents[1]                     # 仓库根目录
CODE_DIR = HERE / "code"
MODELS_DIR = HERE / "models"
OUTPUTS_DIR = HERE / "outputs"
LOGS_DIR = HERE / "logs"

# 训练/检测需要 TensorFlow，用这个 venv
TF_VENV_PY = REPO_ROOT / "envs" / "pcd01_tf" / "Scripts" / "python.exe"
# 画图用（matplotlib）；优先 TensorFlow 环境，它也有 matplotlib
FIG_PY_CANDIDATES = [
    TF_VENV_PY,
    Path(sys.executable),
]

NEW_ANCHORS = CODE_DIR / "model_data" / "yolo_anchors_catface.txt"


# --------------------------------------------------------------------------

def hr(title=""):
    print("")
    print("=" * 74)
    if title:
        print("  " + title)
        print("=" * 74)


def step(n, total, title):
    hr("步骤 %d/%d：%s" % (n, total, title))


def run(cmd, cwd=None, desc=""):
    """执行命令，stdio 继承（不用管道，避免受限环境下的 EPERM）。"""
    print("  $ %s" % " ".join(str(c) for c in cmd))
    if desc:
        print("    %s" % desc)
    t0 = time.time()
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)
    dt = time.time() - t0
    print("    -> 退出码 %d，用时 %s" % (rc, human(dt)))
    return rc


def human(sec):
    if sec < 60:
        return "%.0f 秒" % sec
    if sec < 3600:
        return "%.1f 分钟" % (sec / 60)
    return "%.1f 小时" % (sec / 3600)


def pick_python(candidates):
    for p in candidates:
        if Path(p).is_file():
            return Path(p)
    return Path(sys.executable)


def count_annotations():
    """返回 (图片数, 标注文件数, 自动预标注数)。"""
    jpeg = REPO_ROOT / "datasets" / "cat_face" / "JPEGImages"
    anno = REPO_ROOT / "datasets" / "cat_face" / "Annotations"
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    n_img = sum(1 for p in jpeg.iterdir() if p.is_file() and p.suffix.lower() in exts) \
        if jpeg.is_dir() else 0
    xmls = list(anno.glob("*.xml")) if anno.is_dir() else []
    n_auto = 0
    for x in xmls:
        try:
            if "auto-fasterrcnn" in x.read_text(encoding="utf-8", errors="replace")[:400]:
                n_auto += 1
        except OSError:
            pass
    return n_img, len(xmls), n_auto


def find_latest_weights():
    """在 models/ 里找最新的 .h5 权重。"""
    if not MODELS_DIR.is_dir():
        return None
    cands = [p for p in MODELS_DIR.glob("*.h5")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def count_lines(p: Path):
    if not p.is_file():
        return 0
    return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="实验 01 全流程一键执行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--epochs", type=int, default=200,
                    help="训练轮数（默认 200，CPU 上约 7 小时）")
    ap.add_argument("--backbone", default="full", choices=["full", "tiny"])
    ap.add_argument("--input-shape", type=int, default=416)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--run-name", default=None, help="训练运行的名称（默认自动按时间生成）")
    ap.add_argument("--anchors", default=None,
                    help="anchor 文件。默认用原始的 COCO anchors（yolo_anchors.txt）；"
                         "重新聚类的那份（yolo_anchors_catface.txt）在 10 epoch 的对照里"
                         "反而更差，属于尚未验证的改进，要用请显式指定")
    ap.add_argument("--weights", default=None, help="检测用的权重；默认取 models/ 下最新的")
    ap.add_argument("--score", type=float, default=0.3, help="检测置信度阈值")
    ap.add_argument("--skip-data", action="store_true",
                    help="跳过 make_imagesets / voc_annotation（数据没变时用）")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练")
    ap.add_argument("--skip-detect", action="store_true", help="跳过检测")
    ap.add_argument("--dry-run", action="store_true", help="只检查前置条件，不执行")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    tf_py = TF_VENV_PY
    fig_py = pick_python(FIG_PY_CANDIDATES)
    # 默认用原始 COCO anchors。
    # 重新聚类的那份虽然 anchor 拟合度更高（IoU 67%→80%），但 10 epoch 的对照实验里
    # Recall 反而从 0.52 掉到 0.32，属于**尚未验证**的改进，不作为默认。
    anchors = Path(args.anchors) if args.anchors else (CODE_DIR / "model_data" / "yolo_anchors.txt")

    hr("实验 01 · 全流程执行")
    print("  仓库根目录   : %s" % REPO_ROOT)
    print("  TensorFlow   : %s  %s" % (tf_py, "存在" if tf_py.is_file() else "!! 不存在"))
    print("  画图用 python: %s" % fig_py)
    print("  anchors      : %s" % anchors.name)
    print("  计划         : %s" % ("（dry-run，只检查）" if args.dry_run else
                                   "训练 %d epoch -> 检测 -> 评测 -> 出图" % args.epochs
                                   if not args.skip_train else "（跳过训练）检测 -> 评测 -> 出图"))

    # ---------------- 前置检查 ----------------
    hr("前置检查")
    n_img, n_xml, n_auto = count_annotations()
    print("  图片总数          : %d" % n_img)
    print("  标注文件数        : %d" % n_xml)
    print("  其中机器预标注    : %d" % n_auto)
    print("  标注覆盖率        : %.1f%%" % (n_xml / n_img * 100 if n_img else 0))

    problems = []
    if not tf_py.is_file():
        problems.append("找不到 TensorFlow 环境：%s\n"
                        "       先运行：python envs/setup_tf_env.py" % tf_py)
    if n_xml < 10:
        problems.append("标注文件只有 %d 个，太少，训练没有意义。\n"
                        "       先用 LabelImg 标注，步骤见 datasets/cat_face/README.md" % n_xml)
    if n_auto > 0:
        print("")
        print("  [提醒] 有 %d 个 XML 是**机器预标注**的，框是「整只猫」而不是「猫脸」。" % n_auto)
        print("         直接训练只能得到整只猫的检测器。建议先用 LabelImg 收紧到猫脸。")
        print("         查看还剩多少：python %s --report"
              % (CODE_DIR / "auto_annotate.py").relative_to(REPO_ROOT))

    if problems:
        print("")
        for p in problems:
            print("  [错误] %s" % p)
        return 1

    if args.dry_run:
        hr("dry-run 结束（未执行任何训练/检测）")
        print("  一切就绪。去掉 --dry-run 即可正式运行。")
        return 0

    total = 3 + (0 if args.skip_train else 1) + (0 if args.skip_detect else 1) + 1
    n = 0
    t_all = time.time()

    # ---------------- 1. 数据 ----------------
    n += 1
    if args.skip_data:
        print("")
        print("跳过数据准备（--skip-data）")
    else:
        step(n, total, "生成索引与训练标注")
        if run([sys.executable, CODE_DIR / "make_imagesets.py"], cwd=CODE_DIR) != 0:
            print("[失败] make_imagesets.py 出错，已停止。")
            return 1
        if run([sys.executable, CODE_DIR / "voc_annotation.py"], cwd=CODE_DIR) != 0:
            print("[失败] voc_annotation.py 出错，已停止。")
            return 1

    n_train = count_lines(CODE_DIR / "train.txt")
    n_val = count_lines(CODE_DIR / "val.txt")
    print("")
    print("  train.txt %d 行，val.txt %d 行" % (n_train, n_val))
    if n_train < 10:
        print("  [错误] 训练样本太少（%d），请先补标注。" % n_train)
        return 1

    # ---------------- 2. 训练 ----------------
    weights = None
    if args.skip_train:
        print("")
        print("跳过训练（--skip-train）")
    else:
        n += 1
        run_name = args.run_name or ("%s_%s_%d" % (args.backbone, args.input_shape, int(time.time())))
        step(n, total, "训练 %d epoch（%s backbone，CPU 上预计 %s）"
             % (args.epochs, args.backbone,
                human(8.83 * (n_train // args.batch_size) * args.epochs)))
        cmd = [tf_py, CODE_DIR / "train.py",
               "--epochs", str(args.epochs),
               "--backbone", args.backbone,
               "--input-shape", str(args.input_shape),
               "--batch-size", str(args.batch_size),
               "--anchors", anchors,
               "--run-name", run_name]
        if run(cmd, cwd=CODE_DIR) != 0:
            print("[失败] 训练出错，已停止。已保存的 checkpoint 不会丢，可重跑续训。")
            return 1
        weights = MODELS_DIR / ("%s_trained_weights.h5" % run_name)

    # ---------------- 3. 检测 ----------------
    if not args.skip_detect:
        n += 1
        if args.weights:
            weights = Path(args.weights)
            if not weights.is_absolute():
                weights = (HERE / weights).resolve()
        if not weights or not Path(weights).is_file():
            weights = find_latest_weights()
        if not weights:
            print("[错误] 找不到任何 .h5 权重，无法检测。")
            print("       请先训练，或用 --weights 指定权重路径。")
            return 1

        step(n, total, "批量检测（测试划分，有标注，可直接评测）")
        out_dir = OUTPUTS_DIR / "detect" / Path(weights).stem
        if run([tf_py, CODE_DIR / "detect_images.py",
                "--weights", weights,
                "--split", "test",
                "--score", str(args.score),
                "--output", out_dir], cwd=CODE_DIR) != 0:
            print("[失败] 检测出错，已停止。")
            return 1
        pred_csv = out_dir / "detections.csv"

    # ---------------- 4. 评测 ----------------
    n += 1
    metrics_dir = OUTPUTS_DIR / "metrics" / Path(weights).stem if weights else OUTPUTS_DIR / "metrics"
    if 'pred_csv' in dir() and Path(pred_csv).is_file():
        step(n, total, "评测（Precision / Recall / F1 / AP / mAP）")
        if run([sys.executable, CODE_DIR / "evaluate_detection.py",
                "--pred", pred_csv, "--out", metrics_dir], cwd=CODE_DIR) != 0:
            print("[提醒] 评测未通过（常见原因：检测的图没有对应标注）。继续出图。")
    else:
        print("")
        print("跳过评测（没有检测结果）")

    # ---------------- 5. 出图 ----------------
    n += 1
    step(n, total, "生成报告图表")
    fig_cmd = [fig_py, CODE_DIR / "make_report_figures.py",
               "--out", REPO_ROOT / "reports" / "figures"]
    run_name_for_fig = Path(weights).stem.replace("_trained_weights", "") if weights else None
    if run_name_for_fig and (LOGS_DIR / run_name_for_fig).is_dir():
        fig_cmd += ["--logs", LOGS_DIR / run_name_for_fig,
                    "--run-label", run_name_for_fig]
    if metrics_dir.is_dir():
        fig_cmd += ["--metrics", metrics_dir]
    if weights and (OUTPUTS_DIR / "detect" / Path(weights).stem).is_dir():
        fig_cmd += ["--detect", OUTPUTS_DIR / "detect" / Path(weights).stem, "--montage", "8"]
    run(fig_cmd, cwd=CODE_DIR)

    # ---------------- 收尾 ----------------
    hr("全部完成，总用时 %s" % human(time.time() - t_all))
    print("  权重     : %s" % weights)
    print("  检测结果 : %s" % (OUTPUTS_DIR / "detect" / Path(weights).stem if weights else "-"))
    print("  指标     : %s" % metrics_dir)
    print("  报告图表 : %s" % (REPO_ROOT / "reports" / "figures"))
    print("")
    print("  接下来：把指标填进 reports/01_目标检测_实验报告.md 第六节。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
