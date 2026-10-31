#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容层：让 keras-yolo3 能在较新的 Keras 上跑起来。

keras-yolo3 写于 Keras 2.2~2.4 时代，之后 Keras 把不少层**从子模块提升到了
`keras.layers` 顶层**，旧的子模块路径要么被删、要么还在但不再导出这些名字。
本仓库不改上游源码（`yolo3/model.py`、`yolo.py` 保持原样，便于对照与还原），
而是在**导入之前**把缺失的名字补进对应的子模块。

用法：在 import yolo3 / yolo 之前先 `import keras_compat`。
`config.py` 已经帮所有脚本做了这件事（用 try/except 包住，
所以不依赖 Keras 的纯标准库脚本照样能跑）。

已知的两处（`yolo3/model.py` 第 9、10 行）：
    from keras.layers.advanced_activations import LeakyReLU
    from keras.layers.normalization import BatchNormalization
"""

from __future__ import annotations

import importlib
import sys
import types

# (子模块路径, 需要保证可导入的名字列表)
# 这些名字在 Keras 2.10 都位于 keras.layers 顶层。
SHIMS = [
    ("keras.layers.advanced_activations", [
        "LeakyReLU", "PReLU", "ELU", "Softmax", "ReLU", "ThresholdedReLU",
    ]),
    ("keras.layers.normalization", [
        "BatchNormalization", "LayerNormalization",
    ]),
    ("keras.layers.merge", [
        "Add", "Subtract", "Multiply", "Average", "Maximum", "Minimum",
        "Concatenate", "Dot",
    ]),
    ("keras.layers.pooling", [
        "MaxPooling2D", "AveragePooling2D", "GlobalMaxPooling2D",
        "GlobalAveragePooling2D",
    ]),
]

_applied = []


def _patch(submodule: str, names) -> str:
    """把 names 里缺失的层补进 submodule，返回本次补了什么（空串表示没动）。"""
    from keras import layers as L

    # 子模块可能已被删（ImportError），也可能还在但缺名字
    try:
        mod = importlib.import_module(submodule)
    except Exception:
        mod = types.ModuleType(submodule)
        sys.modules[submodule] = mod
        # 挂到父包上，让 `from keras.layers import <子模块>` 也能用
        parent_path, _, leaf = submodule.rpartition(".")
        try:
            parent = importlib.import_module(parent_path)
            setattr(parent, leaf, mod)
        except Exception:
            pass

    added = []
    for name in names:
        if hasattr(mod, name):
            continue
        obj = getattr(L, name, None)
        if obj is None:
            continue
        # 不覆盖已有属性，只补缺的
        try:
            setattr(mod, name, obj)
            added.append(name)
        except Exception:
            pass

    if added:
        return "%s += %s" % (submodule, ",".join(added))
    return ""


def apply():
    """应用所有兼容补丁，返回已应用项的说明列表。

    若当前环境根本没装 Keras，说明调用方是纯标准库脚本
    （make_imagesets / voc_annotation / evaluate_detection 等），
    此时**静默返回**，不要刷一堆报错出来干扰正常输出。
    """
    try:
        import keras  # noqa: F401
    except ImportError:
        return []

    for submodule, names in SHIMS:
        try:
            note = _patch(submodule, names)
            if note:
                _applied.append(note)
        except Exception as e:
            print("[keras_compat] 处理 %s 失败：%s: %s"
                  % (submodule, type(e).__name__, e))

    for fn in (_patch_backend, _patch_keras_utils):
        try:
            note = fn()
            if note:
                _applied.append(note)
        except Exception as e:
            print("[keras_compat] %s 失败：%s: %s" % (fn.__name__, type(e).__name__, e))

    return list(_applied)


def _patch_backend() -> str:
    """补 keras.backend 上被移除的属性。

    `yolo3/model.py` 第 394 行用了
        K.control_flow_ops.while_loop(cond, body, loop_vars)
    这是 TF1 时代的写法；Keras 2.10 的 backend 不再转发 `control_flow_ops`。
    TF2 的 `tf.while_loop` 签名兼容（cond, body, loop_vars），直接把它挂上去即可。
    """
    import keras.backend as K
    import tensorflow as tf

    notes = []

    if not hasattr(K, "control_flow_ops"):
        K.control_flow_ops = tf
        notes.append("keras.backend.control_flow_ops -> tensorflow")

    if not hasattr(K, "learning_phase"):
        lp = getattr(getattr(tf, "keras", None), "backend", None)
        lp = getattr(lp, "learning_phase", None) or \
            getattr(getattr(tf, "compat", None), "v1", None)
        if lp is not None and hasattr(lp, "keras"):
            lp = lp.keras.backend.learning_phase
        if callable(lp):
            K.learning_phase = lp
            notes.append("keras.backend.learning_phase -> tf")

    return "; ".join(notes)


def _patch_keras_utils() -> str:
    """补 keras.utils.multi_gpu_model。

    `yolo.py` 第 19 行是模块级导入：
        from keras.utils import multi_gpu_model
    这个函数在较新的 Keras 已被移除，缺了它 `import yolo` 直接失败。
    它只在 gpu_num>=2 时被调用，而本机没有可用 GPU，所以放一个会明确报错的替身。
    """
    from keras import utils

    if hasattr(utils, "multi_gpu_model"):
        return ""

    def multi_gpu_model(model, gpus=None):
        raise NotImplementedError(
            "当前 Keras 已移除 multi_gpu_model；而且本机 TensorFlow 用不了 GPU"
            "（RTX 4060 是 sm_89，需要 CUDA 11.8+）。请把 gpu_num 设为 1。")

    utils.multi_gpu_model = multi_gpu_model
    return "keras.utils.multi_gpu_model -> 占位实现（gpu_num>=2 时才调用）"


# 导入本模块即生效
apply()
