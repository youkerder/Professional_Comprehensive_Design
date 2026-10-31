#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 keras-yolo3 的 YOLO 之上，提供"能拿到原始检测框"的接口。

======================================================================
为什么需要这一层
======================================================================
上游 `yolo.py` 的 `detect_image()` 把前处理、推理、绘图揉在一起，
**只返回画好框的图片，不返回框坐标**：

    out_boxes, out_scores, out_classes = self.sess.run(...)
    for i, c in reversed(list(enumerate(out_classes))):
        draw.rectangle(...)          # 框信息就地消费掉了
    return image                     # 只有图

要算 mAP / Precision / Recall 就必须拿到坐标，所以这里把三步拆开：
    predict(image)  -> (boxes, scores, classes)   只推理
    annotate(...)   -> 画好框的图                 只绘图
    detect(image)   -> (图, boxes, scores, classes) 两者都要

前处理与绘图逻辑**照抄上游**，保证结果图与直接跑 yolo.py 一致；
代价是有一段重复代码，上游一旦改动这里要跟着改。
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --------------------------------------------------------------------------
# 推理前必须关掉 TF2 的 eager 执行
# --------------------------------------------------------------------------
# 上游 yolo.py 是 **TF1 图模式**的写法：
#     self.sess = K.get_session()
#     self.input_image_shape = K.placeholder(shape=(2,))
#     out_boxes, out_scores, out_classes = self.sess.run([...], feed_dict={...})
# 在 TF2 默认的 eager 模式下，yolo_eval() 里对动态形状做切片/reshape 时会直接抛：
#     ValueError: Subshape must have computed start >= end since stride is negative,
#     but is 0 and 2 (computed from start 0 and end 9223372036854775807 over shape
#     with rank 2 and stride-1)
# 关掉 eager 回到图模式后即可正常工作（训练路径不需要，它走的是 Keras 的
# 符号图，实测在 eager 下也能正常训练）。
import tensorflow as tf  # noqa: E402

tf.compat.v1.disable_eager_execution()

import numpy as np  # noqa: E402
from keras import backend as K  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from yolo import YOLO  # noqa: E402
from yolo3.utils import letterbox_image  # noqa: E402

# 上游 yolo.py 里字体的相对路径是 'font/FiraMono-Medium.otf'，
# 依赖"在 code/ 目录下运行"。这里按本文件位置定位，避免受工作目录影响。
_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "font", "FiraMono-Medium.otf")


class BoxYOLO(YOLO):
    """带原始框输出的 YOLO 封装。"""

    def __init__(self, **kwargs):
        # 上游在 detect_image 里用相对路径取字体，这里不依赖 CWD
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # 推理：与上游 detect_image 的前处理完全一致
    # ------------------------------------------------------------------
    def predict(self, image: Image.Image):
        """返回 (boxes, scores, classes)。

        boxes 为 [[top, left, bottom, right], ...]（注意顺序，与 yolo_eval 一致），
        坐标已映射回**原图**像素坐标。
        """
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, "Multiples of 32 required"
            assert self.model_image_size[1] % 32 == 0, "Multiples of 32 required"
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)

        image_data = np.array(boxed_image, dtype="float32")
        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0,
            },
        )
        return out_boxes, out_scores, out_classes

    # ------------------------------------------------------------------
    # 绘图：与上游 detect_image 的绘制逻辑一致
    # ------------------------------------------------------------------
    def annotate(self, image: Image.Image, out_boxes, out_scores, out_classes):
        try:
            font = ImageFont.truetype(
                font=_FONT_PATH,
                size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
        except OSError:
            # 字体缺失时退化到默认字体，不要让整个检测流程挂掉
            font = ImageFont.load_default()
            print("[警告] 找不到字体 %s，改用默认字体（框仍会画出）" % _FONT_PATH)

        thickness = (image.size[0] + image.size[1]) // 300
        thickness = max(1, thickness)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = "{} {:.2f}".format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            try:
                label_size = draw.textbbox((0, 0), label, font=font)[2:]
            except AttributeError:
                # Pillow < 10
                label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype("int32"))
            left = max(0, np.floor(left + 0.5).astype("int32"))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
            right = min(image.size[0], np.floor(right + 0.5).astype("int32"))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 画多层矩形模拟线宽。
            #
            # 上游 yolo.py 直接写成 `range(thickness)` 无条件画，当框很小而线宽较大时
            # （例如框宽 3 px、thickness 4），内层会出现 left+t > right-t，
            # PIL 会抛 `ValueError: y1 must be greater than or equal to y0`，
            # 整张图的检测就失败了。这里在内层矩形退化前提前退出。
            for t in range(thickness):
                if left + t >= right - t or top + t >= bottom - t:
                    break
                draw.rectangle(
                    [left + t, top + t, right - t, bottom - t],
                    outline=self.colors[c])

            # 标签底色与文字同样要做边界保护
            lw, lh = int(label_size[0]), int(label_size[1])
            tx, ty = int(text_origin[0]), int(text_origin[1])
            if lw > 0 and lh > 0 and \
                    tx + lw <= image.size[0] and ty + lh <= image.size[1] \
                    and tx >= 0 and ty >= 0:
                draw.rectangle([tx, ty, tx + lw, ty + lh], fill=self.colors[c])
                draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
            del draw

        return image

    # ------------------------------------------------------------------
    def detect(self, image: Image.Image, copy: bool = True):
        """返回 (带框的图, boxes, scores, classes)。"""
        src = image.copy() if copy else image
        boxes, scores, classes = self.predict(src)
        annotated = self.annotate(src.copy() if copy else src, boxes, scores, classes)
        return annotated, boxes, scores, classes
