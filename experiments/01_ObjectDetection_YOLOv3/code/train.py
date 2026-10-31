#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 01 训练脚本：在猫咪/猫脸数据集上**从头训练** YOLOv3。

对应《目标检测实验指导书》2.1 节。指导书直接给了一段代码要求"替换 train.py 中的代码"，
本脚本在保持其训练逻辑（Adam 优化、yolo_loss、data_generator、val_split）的前提下做了这些事：

  1. 路径全部走 config.py，不再把 'logs/000/' 之类的相对路径写死；
  2. 加了 `--benchmark`：先实测每个 step 的耗时，用来决定 epoch 数
     （本机 TF 只能用 CPU，见下），不要凭猜；
  3. 支持 `--backbone tiny` 切换到 Tiny-YOLOv3，CPU 上快很多；
  4. 数据不足时明确报错，而不是训出一个废模型。

======================================================================
本机重要限制：只能 CPU 训练
======================================================================
RTX 4060 Laptop 是 sm_89（Ada），需要 CUDA 11.8+。
  * tensorflow 2.10.0（本机实际使用）绑定 CUDA 11.2，最高支持 sm_86
  * tensorflow 2.3.0（指导书环境一）绑定 CUDA 10.1，最高 sm_75
  * TensorFlow 自 2.11 起取消 Windows 原生 GPU 支持
所以本机 TF 训练一定走 CPU。实测（full YOLOv3、416×416、batch 10）：
每 step 约 8.8 秒，14 step/epoch，200 epoch 约 6.9 小时。
先用 --benchmark 自己测一遍再决定 epoch 数。

用法：
    # 1) 先测速（强烈建议）
    python train.py --benchmark 10

    # 2) 正式训练（CPU 上 full 版 200 epoch 约 7 小时）
    python train.py --epochs 200

    # 3) 想快很多就用 Tiny-YOLOv3 + 更小输入
    python train.py --backbone tiny --input-shape 320 --epochs 100
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

# 必须在 import tensorflow 之前设置，压掉一堆 INFO 级日志
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from config import (
        ANCHORS_PATH, CLASSES_PATH, CODE_DIR, LOGS_DIR, MODELS_DIR,
        NUM_CLASSES, ensure_dirs,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        ANCHORS_PATH, CLASSES_PATH, CODE_DIR, LOGS_DIR, MODELS_DIR,
        NUM_CLASSES, ensure_dirs,
    )


# --------------------------------------------------------------------------
# 工具函数（与 keras-yolo3 / 指导书一致）
# --------------------------------------------------------------------------

def get_classes(classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = [c.strip() for c in f.readlines() if c.strip()]
    return class_names


def get_anchors(anchors_path):
    import numpy as np
    with open(anchors_path, "r", encoding="utf-8") as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",") if x.strip()]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """指导书给出的生成器：逐 batch 产出 (image_data, *y_true), zeros。"""
    import numpy as np
    from yolo3.model import preprocess_true_boxes
    from yolo3.utils import get_random_data

    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for _b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    if len(annotation_lines) == 0 or batch_size <= 0:
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def create_model(input_shape, anchors, num_classes, backbone="full",
                 load_pretrained=False, freeze_body=False,
                 weights_path=None, ignore_thresh=0.5):
    """建模型。

    load_pretrained=False 即**从头训练**（指导书要求）。
    backbone='tiny' 使用 tiny_yolo_body，CPU 上快得多。
    """
    import keras.backend as K
    from keras.layers import Input, Lambda
    from keras.models import Model
    from yolo3.model import yolo_body, tiny_yolo_body, yolo_loss

    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(shape=(h // {0: 32, 1: 16, 2: 8}[l],
                     w // {0: 32, 1: 16, 2: 8}[l],
                     num_anchors // 3, num_classes + 5))
        for l in range(3)
    ]

    if backbone == "tiny":
        model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
        print("Created Tiny-YOLOv3 model with %d anchors and %d classes."
              % (num_anchors, num_classes))
    else:
        model_body = yolo_body(image_input, num_anchors // 3, num_classes)
        print("Created YOLOv3 model with %d anchors and %d classes."
              % (num_anchors, num_classes))

    if load_pretrained:
        if not weights_path or not Path(weights_path).is_file():
            print("[警告] 指定了 load_pretrained 但权重文件不存在：%s" % weights_path)
            print("       继续从头训练。")
        else:
            model_body.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
            print("Loaded weights %s." % weights_path)
            if freeze_body:
                num = len(model_body.layers) - 7
                for i in range(num):
                    model_body.layers[i].trainable = False
                print("Froze the first %d of %d layers."
                      % (num, len(model_body.layers)))

    model_loss = Lambda(
        yolo_loss, output_shape=(1,), name="yolo_loss",
        arguments={"anchors": anchors, "num_classes": num_classes,
                   "ignore_thresh": ignore_thresh},
    )([*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)
    return model


def load_annotation_lines(path: Path):
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def rebase_annotation_lines(lines, base: Path = CODE_DIR):
    """把每行开头的图片路径转成绝对路径。

    为什么需要：`voc_annotation.py` 默认写的是**相对 code/ 目录**的路径，
    这样标注文件能随仓库整体移动。但 keras-yolo3 的 `get_random_data` 是
    拿字符串直接 `Image.open(line[0])`，相对路径会按**当前工作目录**解析。
    只要 train.py 不是从 code/ 下启动，就会 FileNotFoundError。

    这里统一重建成绝对路径，于是 train.py 从任何目录运行都没问题。
    返回 (新行列表, 被改写的行数)。
    """
    out = []
    changed = 0
    for ln in lines:
        groups = ln.split()
        if not groups:
            continue
        p = groups[0]
        if not os.path.isabs(p):
            groups[0] = os.path.normpath(os.path.join(str(base), p))
            changed += 1
        out.append(" ".join(groups))
    return out, changed


CKPT_RE = re.compile(r"ep(\d+)-loss([\d.]+)-val_loss([\d.]+)\.h5$", re.IGNORECASE)


def best_checkpoint(log_dir: Path, prefer: str = "val_loss"):
    """从 checkpoint 文件名里找出 loss / val_loss 最小的那个。

    ModelCheckpoint 的文件名形如 `ep075-loss26.423-val_loss20.335.h5`，
    本身就带 loss 与 val_loss，所以不用解析 h5 就能挑出最优的。
    返回 Path 或 None。
    """
    best = None
    best_val = None
    for p in log_dir.glob("*.h5"):
        m = CKPT_RE.search(p.name)
        if not m:
            continue
        val = float(m.group(3) if prefer == "val_loss" else m.group(2))
        if best_val is None or val < best_val:
            best_val, best = val, p
    return best


def count_nonfinite(path: Path):
    """统计 h5 权重里 NaN / Inf 的个数。

    返回：
      * int >= 0 —— 能正常打开，这是 NaN/Inf 的个数（0 表示权重正常）
      * None     —— **文件打不开**（损坏 / 只写了一半）

    ⚠️ 必须把「打不开」和「没有 NaN」区分开。
    最初的实现两件事都返回 0，于是一个因强制重启而损坏的 checkpoint
    被判断成"正常"，直到拿它去推理才暴露出来。
    """
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None

    total_bad = 0

    def visit(name, obj):
        nonlocal total_bad
        if isinstance(obj, h5py.Dataset) and obj.size:
            a = obj[()]
            total_bad += int(np.isnan(a).sum() + np.isinf(a).sum())

    try:
        with h5py.File(str(path), "r") as f:
            _ = list(f.keys())      # 触发内部结构校验，损坏文件会在这里抛异常
            f.visititems(visit)
    except Exception:
        return None
    return total_bad


def make_keep_every_n(log_dir: Path, every: int):
    """每 N 个 epoch 保存一次权重的回调，与 val_loss 是否改善无关。

    为什么需要：`ModelCheckpoint(save_best_only=True)` 只在 val_loss 改善时保存，
    而本实验的验证集只有 18 张、`validation_steps` 又只取 1 个 batch（10 张），
    **val_loss 的方差极大**——实测两次同配置训练的 val_loss 相差上百倍。
    后果是：
      * 可能连续很多轮不写 checkpoint，训练成果根本没落盘；
      * "val_loss 最优"的那个 checkpoint 未必是真最优。
    所以除了"最优"之外，再定期保存一份，事后用检测指标（AP）挑真正最好的那个。
    """
    from keras.callbacks import Callback

    class KeepEveryN(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            n = epoch + 1
            if every <= 0 or n % every != 0:
                return
            loss = logs.get("loss")
            vl = logs.get("val_loss", loss)
            name = "ep%03d-loss%.3f-val_loss%.3f.h5" % (
                n, loss if loss is not None else 0.0,
                vl if vl is not None else 0.0)
            try:
                self.model.save_weights(str(log_dir / name))
                print("\n[定期保存] %s" % name)
            except Exception as e:
                print("\n[警告] 定期保存失败：%s: %s" % (type(e).__name__, e))

    return KeepEveryN()


def describe_env():
    """打印 TF/Keras 版本与可见设备，便于确认是不是在 CPU 上跑。"""
    import tensorflow as tf
    import keras
    print("tensorflow   : %s" % tf.__version__)
    print("keras        : %s" % keras.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("可见 GPU     : %s" % (gpus if gpus else "无（本机 TF 只能 CPU，见脚本头部说明）"))


# --------------------------------------------------------------------------
# 测速
# --------------------------------------------------------------------------

def run_benchmark(model, lines, args, anchors, num_classes):
    """实测每个训练 step 的耗时，据此估算整轮训练时间。

    步数取 --benchmark 的值（例如 `--benchmark 5` 就是跑 5 个 step）。
    """
    import numpy as np

    n_steps = args.benchmark
    input_shape = (args.input_shape, args.input_shape)
    num_val = int(len(lines) * args.val_split)
    num_train = len(lines) - num_val
    if num_train < args.batch_size:
        print("[错误] 训练样本 %d 个，少于 batch_size=%d，无法测速。"
              % (num_train, args.batch_size))
        return 1

    model.compile(optimizer="adam",
                  loss={"yolo_loss": lambda y_true, y_pred: y_pred})
    gen = data_generator_wrap(lines[:num_train], args.batch_size,
                              input_shape, anchors, num_classes)

    print("=" * 70)
    print("测速（%d 个 step）" % n_steps)
    print("=" * 70)
    print("backbone     : %s" % args.backbone)
    print("input_shape  : %s" % (input_shape,))
    print("batch_size   : %d" % args.batch_size)

    # 头几个 step 作为预热（CPU 上首次计算图构建很慢），不计入
    warmup = min(2, max(1, n_steps // 3))
    print("预热 %d 个 step（不计时）..." % warmup)
    for _ in range(warmup):
        x, y = next(gen)
        model.train_on_batch(x, y)

    print("开始计时...")
    times = []
    t_all = time.time()
    for i in range(n_steps):
        x, y = next(gen)
        t0 = time.time()
        loss = model.train_on_batch(x, y)
        dt = time.time() - t0
        times.append(dt)
        print("  step %3d/%d  %.2f s  loss=%.4f" % (i + 1, n_steps, dt, float(loss)))
    total = time.time() - t_all

    import statistics
    mean = statistics.mean(times)
    med = statistics.median(times)
    steps_per_epoch = max(1, num_train // args.batch_size)

    print("")
    print("-" * 70)
    print("每个 step  : 平均 %.2f s，中位数 %.2f s" % (mean, med))
    print("训练样本   : %d 个（val %d 个）" % (num_train, num_val))
    print("每个 epoch : %d 个 step" % steps_per_epoch)
    print("")
    print("按平均耗时估算整轮训练时间：")
    for ep in (50, 100, 200, 500):
        sec = mean * steps_per_epoch * ep
        print("  %3d epoch -> %s" % (ep, _human(sec)))
    print("")
    print("建议：")
    if mean * steps_per_epoch * 100 > 3600 * 4:
        print("  CPU 上比较慢。可以先跑：")
        print("    python train.py --backbone tiny --input-shape 320 --epochs 100")
        print("  或者减少 epoch 数，先确认流程跑通。")
    else:
        print("  这个速度可以接受，直接跑 --epochs 200 即可。")
    return 0


def _human(sec: float) -> str:
    if sec < 60:
        return "%.0f 秒" % sec
    if sec < 3600:
        return "%.1f 分钟" % (sec / 60)
    return "%.1f 小时" % (sec / 3600)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="实验 01：从头训练 YOLOv3 做猫脸检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--annotation", default=str(CODE_DIR / "train.txt"),
                    help="训练标注文件（voc_annotation.py 生成）")
    ap.add_argument("--val-annotation", default=None,
                    help="验证标注文件（默认尝试 code/val.txt）")
    ap.add_argument("--classes", default=str(CLASSES_PATH), help="类别文件")
    ap.add_argument("--anchors", default=str(ANCHORS_PATH), help="anchor 文件")
    ap.add_argument("--input-shape", type=int, default=416, help="输入边长（32 的倍数）")
    ap.add_argument("--batch-size", type=int, default=10, help="batch size（指导书用 10）")
    ap.add_argument("--epochs", type=int, default=200,
                    help="训练轮数（指导书结论：200 epoch 效果最好）")
    ap.add_argument("--backbone", default="full", choices=["full", "tiny"],
                    help="full=YOLOv3，tiny=Tiny-YOLOv3（CPU 上快很多）")
    ap.add_argument("--val-split", type=float, default=0.1, help="从训练集切出的验证比例")
    ap.add_argument("--patience", type=int, default=20,
                    help="验证 loss 连续多少轮不下降就早停（默认 20）")
    ap.add_argument("--lr-plateau", action="store_true", default=True,
                    help="启用 ReduceLROnPlateau（默认启用；上游 keras-yolo3 本来就有）")
    ap.add_argument("--no-lr-plateau", dest="lr_plateau", action="store_false",
                    help="关闭 ReduceLROnPlateau")
    ap.add_argument("--lr-patience", type=int, default=8,
                    help="ReduceLROnPlateau 的 patience（默认 8）")
    ap.add_argument("--clipnorm", type=float, default=5.0,
                    help="梯度裁剪的范数上限（0 表示不裁剪，默认 5.0）。"
                         "实测不裁剪时训练后期会出现 NaN 发散")
    ap.add_argument("--ignore-thresh", type=float, default=0.5,
                    help="yolo_loss 的 ignore_thresh（默认 0.5，与指导书一致）。"
                         "与真值 IoU 超过该值的背景预测不再被当作背景惩罚")
    ap.add_argument("--keep-every", type=int, default=10,
                    help="每 N 个 epoch 无条件保存一次权重（默认 10，0 表示关闭）。"
                         "因为 val_loss 方差大，save_best_only 可能长期不写 checkpoint")
    ap.add_argument("--resume", default=None,
                    help="断点续训：从这个 .h5 权重继续训练（配合 --initial-epoch 使用）")
    ap.add_argument("--initial-epoch", type=int, default=0,
                    help="从第几个 epoch 接着算（断点续训时填已完成的轮数）")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="学习率（默认 1e-3，与指导书一致）。"
                         "断点续训建议调小，例如 2e-4，避免扰动已收敛的模型")
    ap.add_argument("--pretrained", default=None,
                    help="预训练权重路径；不给则从头训练（指导书要求从头训练）")
    ap.add_argument("--freeze-body", action="store_true", help="冻结主干（配合 --pretrained）")
    ap.add_argument("--run-name", default=None, help="日志与权重的子目录名")
    ap.add_argument("--benchmark", type=int, default=0, metavar="N",
                    help="只测速：跑 N 个 step 后退出，并估算整轮训练时间")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ensure_dirs()
    print("=" * 70)
    print("实验 01 · 训练 YOLOv3 猫脸检测")
    print("=" * 70)
    describe_env()
    print("")

    # ---------------- 检查数据 ----------------
    anno_path = Path(args.annotation)
    lines = load_annotation_lines(anno_path)
    if lines is None:
        print("[错误] 找不到训练标注文件：%s" % anno_path)
        print("       请先依次运行：")
        print("         python auto_annotate.py     # 自动预标注（可选）")
        print("         labelImg                    # 人工把框收紧到猫脸")
        print("         python make_imagesets.py    # 生成索引")
        print("         python voc_annotation.py    # 生成 train.txt")
        return 1
    print("训练标注     : %s（%d 行）" % (anno_path, len(lines)))
    lines, n_rebased = rebase_annotation_lines(lines)
    if n_rebased:
        print("               已把 %d 行里的相对路径重建成绝对路径（相对 %s）"
              % (n_rebased, CODE_DIR))

    val_path = Path(args.val_annotation) if args.val_annotation else (CODE_DIR / "val.txt")
    val_lines = load_annotation_lines(val_path)
    if val_lines:
        val_lines, _ = rebase_annotation_lines(val_lines)
        print("验证标注     : %s（%d 行）" % (val_path, len(val_lines)))
    else:
        print("验证标注     : 无（将从训练集按 val_split=%.2f 切分）" % args.val_split)
    print("")

    if len(lines) < 10:
        print("[错误] 训练样本只有 %d 条，太少了。" % len(lines))
        print("       YOLOv3 从头训练需要每张图都有标注，建议至少 100 张。")
        print("       当前数据集里未标注的图片会被当成纯背景，训不出可用模型。")
        print("       补标注步骤见 datasets/cat_face/README.md。")
        return 1
    if len(lines) < 100:
        print("[警告] 训练样本只有 %d 条。指导书的实验环境用了 230 张，"
              % len(lines))
        print("       规模偏小，检测效果会明显偏弱。")

    # ---------------- 读类别与 anchors ----------------
    class_names = get_classes(args.classes)
    anchors = get_anchors(args.anchors)
    num_classes = len(class_names)
    print("类别         : %s（%d 类）" % (class_names, num_classes))
    print("anchors      : %d 组" % len(anchors))
    print("backbone     : %s" % args.backbone)
    print("input_shape  : %d" % args.input_shape)
    print("batch_size   : %d" % args.batch_size)
    print("epochs       : %d" % args.epochs)

    expected_filters = 3 * (5 + num_classes)
    print("")
    print("提醒：yolov3.cfg 里 filters 应为 3*(5+%d)=%d（从头训练其实不读 cfg，"
          % (num_classes, expected_filters))
    print("      但若之后要转官网权重就必须一致，可用 patch_yolov3_cfg.py --check）")
    print("")

    # ---------------- 建模型 ----------------
    input_shape = (args.input_shape, args.input_shape)
    if args.input_shape % 32 != 0:
        print("[错误] --input-shape 必须是 32 的倍数。")
        return 1

    print("正在构建模型 ...")
    t0 = time.time()
    model = create_model(input_shape, anchors, num_classes,
                         backbone=args.backbone,
                         load_pretrained=bool(args.pretrained),
                         freeze_body=args.freeze_body,
                         weights_path=args.pretrained,
                         ignore_thresh=args.ignore_thresh)
    print("模型构建完成，用时 %.1f 秒" % (time.time() - t0))
    print("")

    # ---------------- 测速模式 ----------------
    if args.benchmark > 0:
        return run_benchmark(model, lines, args, anchors, num_classes)

    # ---------------- 正式训练 ----------------
    import numpy as np
    from keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                 ReduceLROnPlateau, TensorBoard)

    run_name = args.run_name or ("%s_%s_%d" % (args.backbone, args.input_shape, int(time.time())))
    log_dir = LOGS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print("日志目录     : %s" % log_dir)

    # 梯度裁剪：指导书用的是裸 "adam"，实测训练到第 90 epoch 会 NaN 发散，
    # 裁剪后可以稳住。--clipnorm 0 可关掉，回到与指导书完全一致的行为。
    from keras.optimizers import Adam
    if args.clipnorm and args.clipnorm > 0:
        opt = Adam(learning_rate=args.lr, clipnorm=args.clipnorm)
        print("优化器       : Adam(lr=%g, clipnorm=%.1f)" % (args.lr, args.clipnorm))
    else:
        opt = Adam(learning_rate=args.lr)
        print("优化器       : Adam(lr=%g)" % args.lr)

    # 断点续训：把已有权重加载进来再继续
    if args.resume:
        rp = Path(args.resume)
        if not rp.is_file():
            raise SystemExit("找不到续训权重：%s" % rp)
        bad = count_nonfinite(rp)
        if bad is None:
            raise SystemExit(
                "续训权重文件损坏，无法打开：%s\n"
                "（常见于写入过程中进程被杀。请在 logs/ 里另挑一个校验通过的 checkpoint，\n"
                "  可用 tools 目录下的校验方式或直接看 train.py 里的 count_nonfinite）" % rp)
        if bad:
            raise SystemExit(
                "续训权重里有 %d 个 NaN/Inf，不能用来续训：%s" % (bad, rp))
        model.load_weights(str(rp))
        print("续训权重     : %s（校验通过，无 NaN）" % rp.name)
        print("起始 epoch   : %d，将训练到 %d" % (args.initial_epoch, args.epochs))

    model.compile(optimizer=opt,
                  loss={"yolo_loss": lambda y_true, y_pred: y_pred})

    if val_lines:
        train_lines = lines
        validation_lines = val_lines
    else:
        np.random.shuffle(lines)
        num_val = int(len(lines) * args.val_split)
        train_lines = lines[num_val:]
        validation_lines = lines[:num_val]
        print("已按 val_split=%.2f 切分：train %d，val %d"
              % (args.val_split, len(train_lines), len(validation_lines)))

    if not validation_lines:
        print("[警告] 验证集为空，无法监控 val_loss，将只按训练 loss 保存权重。")

    batch_size = args.batch_size
    steps_per_epoch = max(1, len(train_lines) // batch_size)
    print("steps/epoch  : %d" % steps_per_epoch)
    print("")

    callbacks = [
        TensorBoard(log_dir=str(log_dir)),
        ModelCheckpoint(
            filepath=str(log_dir / "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"),
            monitor="val_loss" if validation_lines else "loss",
            save_weights_only=True, save_best_only=True,
            # 指导书用的是 period=1；TF 2.10 已把它换成 save_freq，
            # 'epoch' 表示每个 epoch 结束保存一次（行为与 period=1 一致）
            save_freq="epoch"),
    ]
    # 与 val_loss 是否改善无关的定期保存（见 make_keep_every_n 的说明）
    if args.keep_every and args.keep_every > 0:
        callbacks.append(make_keep_every_n(log_dir, args.keep_every))
    if validation_lines:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=args.patience,
                                      restore_best_weights=False))
        # 上游 keras-yolo3 的 train.py 本来就有 ReduceLROnPlateau，指导书给的版本去掉了。
        # 实测去掉之后 val_loss 长期卡在 20 附近不再下降，第 90 epoch 还出现 NaN 发散。
        if args.lr_plateau:
            callbacks.append(ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=args.lr_patience,
                min_lr=1e-6, verbose=1))

    print("开始训练 ...（Ctrl+C 可中断，已保存的 checkpoint 不会丢）")
    t_start = time.time()
    try:
        # 指导书用的是 fit_generator；TF 2.10 里它已被弃用，
        # Model.fit 本身就能直接吃生成器，参数完全一致，换成 fit 可去掉弃用警告。
        model.fit(
            data_generator_wrap(train_lines, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=steps_per_epoch,
            validation_data=(data_generator_wrap(validation_lines, batch_size, input_shape,
                                                 anchors, num_classes)
                             if validation_lines else None),
            # 用 ceil 而不是 floor：本实验只有 18 张验证图、batch 10，
            # floor 会得到 1 个 step，也就是**只验证 10 张**，
            # 剩下 8 张从不参与，val_loss 方差极大且不可比。
            # 改成 ceil 后 2 个 step 覆盖全部 18 张。
            validation_steps=(max(1, -(-len(validation_lines) // batch_size))
                              if validation_lines else None),
            epochs=args.epochs,
            initial_epoch=args.initial_epoch,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("")
        print("[中断] 收到 Ctrl+C，正在保存当前权重 ...")

    elapsed = time.time() - t_start
    print("")
    print("=" * 70)
    print("训练结束，用时 %s" % _human(elapsed))

    # ------------------------------------------------------------------
    # 保存权重：**优先用验证集最优的 checkpoint，而不是"当前"权重**
    # ------------------------------------------------------------------
    # 这里踩过一次坑：训练到后期损失变成 NaN，而 model.save_weights 保存的是
    # **当前**权重，于是 trained_weights.h5 里 6157 万个参数全是 NaN，
    # 拿去检测一个框都出不来，白跑 3 小时。
    # 所以：
    #   1) 先从 checkpoint 文件名里找出 val_loss 最小的那个（它就是最优权重）；
    #   2) 把它加载回模型，再保存成 trained_weights.h5；
    #   3) 如果确实没有可用 checkpoint，才退回保存当前权重，并检查是否含 NaN。
    best_ckpt = best_checkpoint(log_dir, prefer="val_loss" if validation_lines else "loss")
    if best_ckpt is not None:
        print("最佳 checkpoint : %s" % best_ckpt.name)
        try:
            model.load_weights(str(best_ckpt))
            print("已把最优权重加载回模型（而不是保存 NaN 之后的当前权重）")
        except Exception as e:
            print("[警告] 加载最优 checkpoint 失败：%s: %s" % (type(e).__name__, e))
            print("       将保存当前权重。")

    final = log_dir / "trained_weights.h5"
    model.save_weights(str(final))

    n_bad = count_nonfinite(Path(str(final)))
    if n_bad is None:
        print("")
        print("[错误] 保存的权重**文件损坏，无法打开**！")
        print("       常见原因：写入过程中进程被杀（断电 / 强制重启）。")
        print("       可用的历史 checkpoint 仍在：%s" % log_dir)
    elif n_bad:
        print("")
        print("[错误] 保存的权重里有 %d 个 NaN/Inf 参数，这个模型不可用！" % n_bad)
        print("       原因通常是训练后期数值发散（学习率过大 / 出现坏 batch）。")
        print("       可用的历史 checkpoint 仍在：%s" % log_dir)
        print("       建议改用最优 checkpoint 做检测：")
        if best_ckpt is not None:
            print("         python detect_images.py --weights \"%s\"" % best_ckpt)
    else:
        print("权重检查 : 无 NaN/Inf，正常")

    print("最终权重 : %s" % final)

    # 归档一份到 models/，方便检测脚本直接引用
    # 注意 n_bad 可能是 None（文件打不开），它也是"不能用"，所以用 `is not None and ...`
    usable = (n_bad == 0)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archived = MODELS_DIR / ("%s_trained_weights.h5" % run_name)
    shutil.copy2(final, archived)
    print("已归档   : %s" % archived)
    if not usable and best_ckpt is not None:
        good_bad = count_nonfinite(best_ckpt)
        if good_bad == 0:
            good = MODELS_DIR / ("%s_best_%s" % (run_name, best_ckpt.name))
            shutil.copy2(best_ckpt, good)
            print("最终权重不可用，已把最优 checkpoint 也归档（推荐用这个）：")
            print("  %s" % good)
        else:
            print("[警告] 最优 checkpoint 也不可用（NaN=%s），请在 logs/ 里另挑一个。")
    print("")
    print("下一步：批量检测")
    if usable:
        print("  python detect_images.py --weights \"%s\"" % archived)
    else:
        print("  ↑ 上面归档的 trained_weights.h5 不可用，请改用 logs/ 里校验通过的 checkpoint。")
    return 0 if usable else 2


if __name__ == "__main__":
    sys.exit(main())
