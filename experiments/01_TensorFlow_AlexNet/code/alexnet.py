#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础 AlexNet 模型定义（激活函数可切换）。

结构
----
保留 AlexNet 的**定义性拓扑**：5 个卷积 + 3 个全连接，卷积/全连接后接激活函数，
最后接 softmax 分类头。原论文的配置是：

    输入 224x224x3
    conv1  96 @ 11x11/4  -> LRN -> MaxPool 3x3/2
    conv2 256 @  5x5/1   -> LRN -> MaxPool 3x3/2
    conv3 384 @  3x3/1
    conv4 384 @  3x3/1
    conv5 256 @  3x3/1   -> MaxPool 3x3/2
    FC 4096 -> FC 4096 -> FC 1000 -> softmax

本实验为 CIFAR-10（32x32）与 CPU 训练做的三处调整，报告里需明确说明：

    1. 输入 224 -> 32，故首层卷积由 11x11/4 改为 3x3/1；
       conv2 由 5x5 改为 3x3（32x32 上 5x5 感受野占比过大）。
    2. 全连接 4096 -> 512。原始尺寸下 FC 层占绝大部分参数，
       在 CPU 上是主要瓶颈；缩到 512 后仍保留"三层全连接"的结构。
    3. 局部响应归一化（LRN）默认换成 BatchNorm。BN 能让模型在 epoch 数
       有限时更快收敛，对本次"15 个 epoch 内看出差异"的目标更有利；
       需要更贴近原论文时可用 norm="lrn"。

通道数与全连接宽度见 config.py（实测该配置 3.70 M 参数、4.2 ms/图）。
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from config import (CONV_CHANNELS, DEFAULT_NORM, DROPOUT, FC_UNITS,
                        IMAGE_SHAPE, LEAKY_ALPHA, NUM_CLASSES, WEIGHT_DECAY)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (CONV_CHANNELS, DEFAULT_NORM, DROPOUT, FC_UNITS,
                        IMAGE_SHAPE, LEAKY_ALPHA, NUM_CLASSES, WEIGHT_DECAY)


def make_activation(name: str):
    """按名字返回激活层。这是本实验要对比的自变量之一。"""
    from tensorflow.keras import layers

    key = name.lower().replace("-", "_")
    if key == "relu":
        return layers.ReLU()
    if key in ("leaky_relu", "leakyrelu"):
        return layers.LeakyReLU(alpha=LEAKY_ALPHA)
    if key == "elu":
        return layers.ELU()
    if key == "tanh":
        return layers.Activation("tanh")
    if key in ("sigmoid", "logistic"):
        return layers.Activation("sigmoid")
    raise ValueError("不支持的激活函数：%r（可选 relu / leaky_relu / elu / tanh / sigmoid）" % name)


def _norm_layer(norm: str):
    """返回规范化层工厂；norm='none' 时返回 None。"""
    from tensorflow.keras import layers

    key = (norm or "none").lower()
    if key == "bn":
        return layers.BatchNormalization
    if key == "none":
        return None
    if key == "lrn":
        # Keras 没有 LRN 层，用 Lambda 包一层 tf.nn.local_response_normalization。
        # 原论文参数：depth_radius=2, bias=1.0, alpha=1e-4, beta=0.75
        def _lrn(x):
            import tensorflow as tf
            return tf.nn.local_response_normalization(
                x, depth_radius=2, bias=1.0, alpha=1e-4, beta=0.75)
        return lambda: layers.Lambda(_lrn)
    raise ValueError("不支持的规范化方式：%r（可选 bn / lrn / none）" % norm)


def build_alexnet(input_shape=IMAGE_SHAPE, num_classes: int = NUM_CLASSES,
                  act: str = "relu", norm: str = DEFAULT_NORM,
                  conv_channels=CONV_CHANNELS, fc_units=FC_UNITS,
                  dropout: float = DROPOUT, weight_decay: float = WEIGHT_DECAY,
                  name: str = "alexnet"):
    """构造基础 AlexNet。

    参数
    ----
    act   : 激活函数名（'relu' / 'leaky_relu' / ...），本实验的对比变量之一
    norm  : 'bn'（默认）/ 'lrn'（更贴近原论文）/ 'none'
    weight_decay : 原论文的 weight decay = 5e-4。

        这里用 **L2 正则加到卷积核与全连接核上**实现，而不是用
        `SGD(decay=...)`。原因：Keras 的 `decay` 参数是按
        `lr / (1 + decay * iterations)` 让学习率随迭代下降的，那样
        "恒定学习率"这个前提就不成立了，会污染本实验对学习率的对比。
        L2 正则才是 weight decay 的正确等价形式，且不影响学习率取值。
    """
    from tensorflow.keras import layers, models, regularizers

    norm_factory = _norm_layer(norm)
    # LRN 只对 4 维张量有意义（原论文也只接在 conv1/conv2 之后）。
    # 全连接层是 2 维张量，套 LRN 会直接报错，所以 FC 层只允许用 BN。
    fc_norm_factory = norm_factory if (norm or "none").lower() == "bn" else None
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None

    def conv_block(x, filters, kernel, strides):
        x = layers.Conv2D(filters, kernel, strides=strides, padding="same",
                          use_bias=(norm_factory is None),
                          kernel_regularizer=reg)(x)
        if norm_factory is not None:
            x = norm_factory()(x)
        return make_activation(act)(x)

    inp = layers.Input(shape=input_shape, name="image")

    # --- 5 个卷积层 ---
    x = conv_block(inp, conv_channels[0], 3, 1)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = conv_block(x, conv_channels[1], 3, 1)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = conv_block(x, conv_channels[2], 3, 1)
    x = conv_block(x, conv_channels[3], 3, 1)
    x = conv_block(x, conv_channels[4], 3, 1)
    x = layers.MaxPooling2D(pool_size=2, strides=2, padding="same")(x)

    # --- 3 个全连接层（前两层接 dropout，最后一层是分类头）---
    x = layers.Flatten()(x)
    for i, units in enumerate(fc_units):
        x = layers.Dense(units, name="fc%d" % (i + 1),
                         kernel_regularizer=reg)(x)
        if fc_norm_factory is not None:
            x = fc_norm_factory()(x)
        x = make_activation(act)(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)

    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return models.Model(inp, out, name=name)


def describe_model(model) -> str:
    """返回一行模型摘要：参数量与层数。"""
    return "参数量 %.2f M，层数 %d" % (model.count_params() / 1e6, len(model.layers))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="AlexNet 结构预览")
    ap.add_argument("--act", default="relu")
    ap.add_argument("--norm", default=DEFAULT_NORM)
    ap.add_argument("--summary", action="store_true", help="打印逐层结构")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    m = build_alexnet(act=args.act, norm=args.norm)
    print("激活函数 : %s" % args.act)
    print("规范化   : %s" % args.norm)
    print("输入     : %s" % (IMAGE_SHAPE,))
    print("输出     : %s" % (m.output_shape,))
    print(describe_model(m))
    if args.summary:
        print("")
        m.summary()
