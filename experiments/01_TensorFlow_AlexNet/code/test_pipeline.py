#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验一完整管线自检。

覆盖范围
--------
1. 配置一致性：四组对比矩阵、类别映射、超参合法性
2. 数据管线：形状/类型/标签范围、划分不重叠、归一化只来自训练集、增强可复现
3. 模型：两种激活函数都能构图、输出形状、概率归一、参数量、激活函数确实不同
4. 指标：混淆矩阵方向、P/R/F1 公式、macro/weighted、top-k
5. 推理输出格式：top-k 排序、中文名、任意尺寸输入
6. 端到端：用合成数据跑一次真实的训练 step，确认损失能下降
7. 目录约定：models/logs/outputs 存在

设计原则
--------
* **不需要真的下载 CIFAR-10**：所有数据检查都用合成数组，随时可跑。
* 需要 TensorFlow 的检查（第 3、6 项）在没有 TF 时会**跳过**并明确说明，
  而不是判为失败——这样在只装了 numpy 的环境里也能跑前面几项。

用法
----
    python test_pipeline.py
    python test_pipeline.py --no-tf      # 跳过所有需要 TensorFlow 的检查
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

# 模块级的检查计数与失败收集
STATE = {"pass": 0, "fail": 0, "skip": 0, "fails": []}


def head(title: str):
    print("")
    print("-" * 74)
    print(title)
    print("-" * 74)


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


def skip(msg):
    STATE["skip"] += 1
    print("  [跳过] %s" % msg)


# --------------------------------------------------------------------------
# 1. 配置
# --------------------------------------------------------------------------

def test_config():
    head("1. 配置一致性")

    check(config.NUM_CLASSES == 10, "类别数为 10")
    check(len(config.CLASS_NAMES) == config.NUM_CLASSES,
          "CLASS_NAMES 长度与 NUM_CLASSES 一致")
    check(len(set(config.CLASS_NAMES)) == config.NUM_CLASSES, "类别名无重复")
    check("cat" in config.CLASS_NAMES, "类别里有 cat（与实验一的猫脸检测呼应）")
    check(all(c in config.CLASS_NAMES_ZH for c in config.CLASS_NAMES),
          "每个英文类别都有中文名")

    runs = config.all_runs()
    check(len(runs) == 4, "四组对比配置（2 激活 × 2 学习率）")
    names = [r[0] for r in runs]
    check(len(set(names)) == 4, "四个 run_name 互不重复：%s" % ", ".join(names))
    acts = {r[1] for r in runs}
    lrs = {r[2] for r in runs}
    check(acts == set(config.ACTIVATIONS), "覆盖全部激活函数")
    check(lrs == set(config.LEARNING_RATES), "覆盖全部学习率")
    check("relu" in acts and len(acts) == 2, "包含 ReLU 与另一种激活函数")

    check(config.run_name("relu", 0.01) == "relu-lr0.01",
          "run_name 生成规则正确（%s）" % config.run_name("relu", 0.01))
    check(config.DEFAULT_EPOCHS > 0 and config.DEFAULT_BATCH_SIZE > 0,
          "默认 epoch 与 batch 为正数")
    check(0 < config.VAL_SIZE < 50000, "验证集大小合理")
    check(config.IMAGE_SHAPE == (32, 32, 3), "输入尺寸为 CIFAR-10 原生 32x32x3")
    check(len(config.CONV_CHANNELS) == 5, "卷积层数为 5（AlexNet 的结构特征）")
    check(len(config.FC_UNITS) + 1 == 3,
          "全连接层数为 3（两个隐藏 FC + 一个分类头）")


# --------------------------------------------------------------------------
# 2. 数据
# --------------------------------------------------------------------------

def test_data():
    head("2. 数据管线")

    import data as dm

    # 复用 data.py 自身的自检，但把它纳入总计数
    before = STATE["pass"], STATE["fail"]
    rc = dm.selftest()
    check(rc == 0, "data.selftest() 全部通过")
    # data.selftest 内部已经打印过明细，这里不重复计数它的每一项
    _ = before

    rng = np.random.RandomState(0)
    x = rng.randint(0, 256, size=(300, 32, 32, 3), dtype=np.uint8)
    y = rng.randint(0, 10, size=(300,)).astype(np.uint8)

    tr, va = dm.split_indices(y, val_size=50, seed=42)
    check(len(tr) + len(va) == 300, "划分后总数不变")
    check(not set(tr.tolist()) & set(va.tolist()), "训练/验证索引不重叠")

    mean, std = dm.compute_norm_stats(x[tr])
    check(np.all(std > 0), "标准差全为正")

    xn = dm.normalize(x[tr], mean, std)
    check(xn.shape == x[tr].shape, "归一化不改变形状")
    check(abs(float(xn.mean())) < 0.05, "归一化后均值接近 0")
    check(abs(float(xn.std()) - 1.0) < 0.05, "归一化后标准差接近 1")

    # 归一化的顺序：先 /255 再减均值，如果漏了 /255，数值会大 255 倍
    check(float(np.abs(xn).max()) < 10, "归一化数值量级正常（没有漏掉 /255）")

    # 类别名读取（数据没下载时应返回空列表，不能抛异常）
    try:
        names = dm.class_names()
        check(isinstance(names, list), "class_names() 返回列表（数据未下载时为空）")
    except Exception as e:
        check(False, "class_names() 不应抛异常：%s: %s" % (type(e).__name__, e))

    check(hasattr(dm, "CIFAR_MIRRORS") and len(dm.CIFAR_MIRRORS) >= 2,
          "配置了多个下载镜像（网络不稳定时的兜底）")
    check(hasattr(dm, "_fetch_resumable"), "存在可续传下载函数")
    check(dm.STALL_TIMEOUT > 0, "停滞超时为正数")


# --------------------------------------------------------------------------
# 3. 模型（需要 TensorFlow）
# --------------------------------------------------------------------------

def test_model():
    head("3. 模型结构（需要 TensorFlow）")

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        skip("未安装 TensorFlow，跳过模型检查")
        return

    from alexnet import build_alexnet, describe_model, make_activation

    for act in config.ACTIVATIONS:
        m = build_alexnet(act=act)
        check(m.output_shape == (None, config.NUM_CLASSES),
              "%s：输出形状为 (None, %d)" % (act, config.NUM_CLASSES))
        check(m.count_params() > 0, "%s：%s" % (act, describe_model(m)))

    # 两种激活函数必须真的不同（否则对比就没意义了）
    m_relu = build_alexnet(act="relu")
    m_leaky = build_alexnet(act="leaky_relu")
    types_relu = {type(l).__name__ for l in m_relu.layers}
    types_leaky = {type(l).__name__ for l in m_leaky.layers}
    check("ReLU" in types_relu, "relu 模型里确实有 ReLU 层")
    check("LeakyReLU" in types_leaky, "leaky_relu 模型里确实有 LeakyReLU 层")
    check(types_relu != types_leaky, "两种激活函数构造出的层类型不同")
    check(m_relu.count_params() == m_leaky.count_params(),
          "两种激活函数参数量相同（激活函数本身无参数，对比才公平）")

    # LeakyReLU 的负区间斜率
    lr_layer = next(l for l in m_leaky.layers if type(l).__name__ == "LeakyReLU")
    got_alpha = float(getattr(lr_layer, "alpha", getattr(lr_layer, "negative_slope", -1)))
    check(abs(got_alpha - config.LEAKY_ALPHA) < 1e-6,
          "LeakyReLU 的 alpha = %g" % config.LEAKY_ALPHA)

    # 前向：输出应当是概率分布（和为 1，非负）
    x = np.random.rand(4, 32, 32, 3).astype("float32")
    p = m_relu.predict(x, verbose=0)
    check(p.shape == (4, config.NUM_CLASSES), "前向输出形状 (4, 10)")
    check(np.all(p >= 0), "概率非负")
    check(np.allclose(p.sum(axis=1), 1.0, atol=1e-4), "每行概率和为 1")

    # 激活函数异常名要报错
    try:
        make_activation("nonexistent")
        check(False, "非法激活函数名应当报错")
    except ValueError:
        check(True, "非法激活函数名抛 ValueError")

    # LRN 变体也要能构图（原论文用的就是它）
    try:
        m_lrn = build_alexnet(act="relu", norm="lrn")
        check(m_lrn.output_shape == (None, config.NUM_CLASSES), "norm='lrn' 也能正常构图")
    except Exception as e:
        check(False, "norm='lrn' 构图失败：%s: %s" % (type(e).__name__, e))


# --------------------------------------------------------------------------
# 4. 指标
# --------------------------------------------------------------------------

def test_metrics():
    head("4. 评测指标")

    from evaluate import (confusion_matrix, metrics_from_confusion,
                          topk_accuracy)

    cm = np.array([[5, 1], [2, 8]], dtype=np.int64)
    m = metrics_from_confusion(cm, ["a", "b"])
    check(m["total"] == 16, "样本总数 16")
    check(abs(m["accuracy"] - 13 / 16) < 1e-9, "准确率 = (5+8)/16")

    ca = m["per_class"][0]
    check(ca["support"] == 6 and ca["predicted"] == 7, "class0 真值 6 预测 7")
    check(abs(ca["precision"] - 5 / 7) < 1e-9, "class0 precision = 5/7")
    check(abs(ca["recall"] - 5 / 6) < 1e-9, "class0 recall = 5/6")

    check(confusion_matrix([0, 1, 1], [0, 1, 0], num_classes=2).tolist()
          == [[1, 0], [1, 1]], "混淆矩阵朝向：行=真值、列=预测")

    check(abs(topk_accuracy(np.eye(3), [0, 1, 2], k=1) - 1.0) < 1e-9,
          "单位矩阵下 top-1 全对")

    # 把 evaluate 的自检也跑一遍，确认没有回归
    import evaluate as ev
    rc = ev.selftest()
    check(rc == 0, "evaluate.selftest() 全部通过")


def test_predict_format():
    head("5. 推理输出格式")

    import predict as pd

    probs = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    probs[3] = 0.9      # cat
    probs[5] = 0.07     # dog
    probs[0] = 0.03     # airplane
    txt = pd.format_prediction(probs, topk=3)
    check("cat" in txt.splitlines()[0], "第一名是 cat")
    check("dog" in txt.splitlines()[1], "第二名是 dog")
    check("airplane" in txt.splitlines()[2], "第三名是 airplane")
    check("（猫）" in txt, "输出里有中文类别名")

    img = np.random.randint(0, 256, size=(100, 77, 3), dtype=np.uint8)
    mean = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    x = pd.preprocess(img, mean, std)
    check(x.shape == (1,) + config.IMAGE_SHAPE, "非方形输入也能缩放成模型输入尺寸")

    import predict as pd2
    check(pd2.selftest() == 0, "predict.selftest() 全部通过")


# --------------------------------------------------------------------------
# 6. 端到端训练 step（需要 TensorFlow）
# --------------------------------------------------------------------------

def test_end_to_end():
    head("6. 端到端：合成数据跑真实训练 step（需要 TensorFlow）")

    try:
        import tensorflow as tf
    except ImportError:
        skip("未安装 TensorFlow，跳过端到端检查")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data as dm
    from alexnet import build_alexnet

    rng = np.random.RandomState(0)
    # 造一个"可学"的小任务：把图片均值映射到类别，保证几步内损失能降
    n, c = 64, config.NUM_CLASSES
    x = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    y = rng.randint(0, c, size=n).astype(np.uint8)
    for i in range(n):
        x[i, :, :, :] = 40 + y[i] * 18          # 每个类别一个亮度

    mean, std = dm.compute_norm_stats(x)
    check(np.all(std > 0), "合成数据的标准差为正")

    model = build_alexnet(act="relu")
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    ds = dm.make_dataset(x, y, mean, std, batch_size=32, training=True, seed=0)
    check(ds is not None, "tf.data 数据集构造成功")

    # 取一个批次检查形状与数值
    xb, yb = next(iter(ds))
    check(tuple(xb.shape) == (32, 32, 32, 3), "批次形状为 (32, 32, 32, 3)")
    check(tuple(yb.shape) == (32,), "标签批次形状为 (32,)")
    check(float(tf.reduce_max(tf.abs(xb))) < 20, "批次已归一化（数值量级正常）")

    # 训练几步，损失应当下降
    h1 = model.fit(ds, epochs=1, verbose=0)
    l1 = float(h1.history["loss"][0])
    for _ in range(4):
        h = model.fit(ds, epochs=1, verbose=0)
    l2 = float(h.history["loss"][0])
    check(l2 < l1, "训练 5 个 epoch 后损失下降（%.4f -> %.4f）" % (l1, l2),
          detail="若损失不降，说明数据管线或模型接线有问题")

    acc = float(h.history["accuracy"][0])
    check(acc > 0.3, "在可学的合成任务上准确率提升（%.4f）" % acc)

    # 增强应当只发生在训练模式：验证模式不 shuffle、不裁剪
    ds_val = dm.make_dataset(x, y, mean, std, batch_size=16, training=False, seed=0)
    xb1, _ = next(iter(ds_val))
    xb2, _ = next(iter(ds_val))
    check(np.allclose(xb1.numpy(), xb2.numpy()),
          "非训练模式下两次取批次一致（不随机增强、不打乱）")


# --------------------------------------------------------------------------
# 7. 目录约定
# --------------------------------------------------------------------------

def test_layout():
    head("7. 目录约定")

    for d, name in ((config.MODELS_DIR, "models/"),
                    (config.LOGS_DIR, "logs/"),
                    (config.OUTPUTS_DIR, "outputs/"),
                    (config.PREDICT_DIR, "outputs/predict/"),
                    (config.EVAL_DIR, "outputs/eval/"),
                    (config.COMPARE_DIR, "outputs/compare/")):
        check(d.is_dir(), "%s 存在" % name)

    check(config.REPO_ROOT.is_dir(), "仓库根目录可定位")
    check(config.EXP_DIR.name == "01_TensorFlow_AlexNet",
          "实验目录名正确（%s）" % config.EXP_DIR.name)
    check((config.EXP_DIR / "code").is_dir(), "code/ 存在")

    # 不能误改实验一（目标检测）的目录
    yolo = config.EXPERIMENTS_DIR / "01_ObjectDetection_YOLOv3"
    check(yolo.is_dir(), "实验一（目标检测）目录仍在，未被本次改动影响")


def main():
    ap = argparse.ArgumentParser(description="实验一 AlexNet 完整管线自检")
    ap.add_argument("--no-tf", action="store_true", help="跳过所有需要 TensorFlow 的检查")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 74)
    print("实验一（TensorFlow / AlexNet 图像分类）· 管线自检")
    print("=" * 74)
    print("仓库根目录 : %s" % config.REPO_ROOT)
    print("实验目录   : %s" % config.EXP_DIR)

    test_config()
    test_data()
    if args.no_tf:
        skip("指定了 --no-tf，跳过模型与端到端检查")
    else:
        test_model()
        test_end_to_end()
    test_metrics()
    test_predict_format()
    test_layout()

    print("")
    print("=" * 74)
    print("自检汇总：通过 %d 项，失败 %d 项，跳过 %d 项"
          % (STATE["pass"], STATE["fail"], STATE["skip"]))
    if STATE["fails"]:
        print("")
        print("失败清单：")
        for f in STATE["fails"]:
            print("  - %s" % f)
    print("=" * 74)
    return 1 if STATE["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
