#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""输入一张图像，输出它的类别 —— 本实验题目要求的核心程序。

题目原文："编写程序实现输入一张图像，输出图像的类别"。

用法
----
    # 单张
    python predict.py --image cat.jpg

    # 一批（并保存带标注的结果图）
    python predict.py --dir ../../datasets/cat_face/JPEGImages --limit 5 \
        --save-dir ../outputs/predict

    # 指定权重（默认取 models/ 里验证集最优的那个）
    python predict.py --image x.png --model ../models/relu-lr0.01_best.h5 --act relu

输出内容（同时打印到终端）：
    排名 1: cat（猫）      置信度 0.9231
    排名 2: dog（狗）      置信度 0.0512
    排名 3: horse（马）    置信度 0.0103

注意事项
--------
* 预处理必须与训练时**完全一致**：缩放到 32x32、按训练集统计的均值/标准差归一化。
  这里复用 `data.load_splits()` 拿到同一套 mean/std，避免"训练用一套、推理用另一套"
  这种最常见的精度掉点原因。
* 支持任意尺寸与格式（jpg/png/bmp/webp），带 EXIF 旋转的照片也会正确处理。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from config import (CLASS_NAMES, CLASS_NAMES_ZH, DEFAULT_NORM, DEFAULT_SEED,
                        IMAGE_SHAPE, NUM_CLASSES, PREDICT_DIR, VAL_SIZE)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (CLASS_NAMES, CLASS_NAMES_ZH, DEFAULT_NORM, DEFAULT_SEED,
                        IMAGE_SHAPE, NUM_CLASSES, PREDICT_DIR, VAL_SIZE)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif")


# --------------------------------------------------------------------------
# 预处理（与训练一致）
# --------------------------------------------------------------------------

def load_image_array(path: Path):
    """读图 -> RGB 的 numpy 数组（uint8）。带 EXIF 旋转处理。"""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)     # 手机照片常见，不处理会上下颠倒
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def preprocess(img: np.ndarray, mean, std, size=None):
    """缩放到训练尺寸并归一化。返回 (1, H, W, 3) 的 float32 批次。"""
    from PIL import Image

    size = size or (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    im = Image.fromarray(img).resize(size, Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - mean) / std
    return x[None, ...].astype(np.float32)


def format_prediction(probs, class_names=None, topk: int = 3) -> str:
    """把概率向量格式化成多行文本。这是 predict.py 的输出格式，自检会校验它。"""
    names = list(class_names) if class_names else list(CLASS_NAMES)
    probs = np.asarray(probs).ravel()
    if probs.size != len(names):
        raise ValueError("概率长度 %d 与类别数 %d 不一致" % (probs.size, len(names)))
    k = max(1, min(topk, probs.size))
    order = np.argsort(-probs)[:k]
    lines = []
    for rank, idx in enumerate(order, 1):
        zh = CLASS_NAMES_ZH.get(names[idx], "")
        lines.append("  排名 %d: %-12s%s  置信度 %.4f"
                     % (rank, names[idx], ("（%s）" % zh) if zh else "", probs[idx]))
    return "\n".join(lines)


def _cjk_font(size: int = 14):
    """尽量找一个支持中文的字体。

    PIL 的**默认位图字体只支持 latin-1**，直接画中文会抛
    `UnicodeEncodeError: 'latin-1' codec can't encode character`。
    所以这里按优先级找系统中文字体，找不到就返回 None，
    调用方退回"只画英文类别名"。

    注意仓库里实验一用的 `FiraMono-Medium.otf` 没有中文字形，不能替代。
    """
    from PIL import ImageFont

    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",    # 黑体
        r"C:\Windows\Fonts\simsun.ttc",    # 宋体
        r"C:\Windows\Fonts\Deng.ttf",      # 等线
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return None


def draw_result(img: np.ndarray, probs, class_names=None, topk: int = 3):
    """在图片顶部画出预测结果，返回 PIL Image（用于保存）。

    有中文字体时同时画中英文；没有时只画英文，保证**任何环境都不会崩**。
    """
    from PIL import Image, ImageDraw

    names = list(class_names) if class_names else list(CLASS_NAMES)
    probs = np.asarray(probs).ravel()
    order = np.argsort(-probs)[:max(1, min(topk, probs.size))]

    font = _cjk_font(14)

    def label(i):
        en = names[i]
        zh = CLASS_NAMES_ZH.get(en, "")
        if font is not None and zh:
            return "%s/%s  %.4f" % (en, zh, probs[i])
        return "%s  %.4f" % (en, probs[i])

    im = Image.fromarray(img).convert("RGB")
    line_h = 20 if font is not None else 12
    lines = [label(i) for i in order]

    # 顶部加一条半透明的黑底，保证文字在任何图上都看得清。
    # 注意 Image.alpha_composite 要求两张图尺寸完全一致，
    # 所以叠加层必须建成整图大小，再把文字条 paste 到顶部。
    strip_h = min(line_h * len(lines) + 6, im.height)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    overlay.paste(Image.new("RGBA", (im.width, strip_h), (0, 0, 0, 160)), (0, 0))
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(im)
    for i, text in enumerate(lines):
        try:
            draw.text((6, 3 + i * line_h), text, font=font,
                      fill=(255, 235, 59) if i == 0 else (255, 255, 255))
        except Exception:
            # 字体出了问题也不能让整张图失败，退回英文
            ascii_text = names[order[i]] + "  %.4f" % probs[order[i]]
            draw.text((6, 3 + i * line_h), ascii_text,
                      fill=(255, 235, 59) if i == 0 else (255, 255, 255))
    return im


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------

def pick_default_model():
    """默认取 models/ 里 val_acc 最高的那个 _best.h5。"""
    from config import MODELS_DIR
    cands = sorted(MODELS_DIR.glob("*_best.h5"))
    if cands:
        return cands[0]
    cands = sorted(MODELS_DIR.glob("*.h5"))
    return cands[0] if cands else None


def load_model(model_path: Path, act: str, norm: str):
    import tensorflow as tf                      # noqa: F401

    from alexnet import build_alexnet

    model = build_alexnet(act=act, norm=norm)
    model.load_weights(str(model_path))
    return model


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def selftest() -> int:
    """校验输出格式、top-k 排序、预处理形状与数值范围。不需要 TensorFlow。"""
    fails = []

    def check(cond, msg):
        print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
        if not cond:
            fails.append(msg)

    print("predict.py 自检（不需要 TensorFlow）")
    print("-" * 74)

    probs = np.array([0.05, 0.02, 0.03, 0.80, 0.01, 0.04, 0.01, 0.02, 0.01, 0.01])
    text = format_prediction(probs, topk=3)
    lines = text.strip().splitlines()
    check(len(lines) == 3, "topk=3 时输出 3 行")
    check("cat" in lines[0], "概率最大的类别排在第一名（cat）")
    check("0.8000" in lines[0], "第一名置信度正确")
    check("（猫）" in lines[0], "中文名正确显示")
    # 概率降序：0.80(cat, idx3) > 0.05(airplane, idx0) > 0.04(dog, idx5)
    check("airplane" in lines[1], "第二名是 airplane（0.05）")
    check("dog" in lines[2], "第三名是 dog（0.04）")

    # 概率都是一样时不应崩溃，且长度仍然正确
    flat = np.ones(NUM_CLASSES) / NUM_CLASSES
    check(len(format_prediction(flat).strip().splitlines()) == 3, "等概率时输出行数仍为 3")

    # topk 超过类别数时应被截断
    check(len(format_prediction(flat, topk=99).strip().splitlines()) == NUM_CLASSES,
          "topk 超过类别数时截断为 %d 行" % NUM_CLASSES)

    # 长度不符要报错
    try:
        format_prediction(np.ones(3))
        check(False, "概率长度不符应当报错")
    except ValueError:
        check(True, "概率长度不符时抛 ValueError")

    # 预处理：形状、类型、数值范围
    rng = np.random.RandomState(0)
    img = rng.randint(0, 256, size=(64, 48, 3), dtype=np.uint8)   # 故意不是 32x32
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.25, 0.25, 0.25], dtype=np.float32)
    x = preprocess(img, mean, std)
    check(x.shape == (1, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3),
          "任意尺寸输入都能缩放成 %s" % (IMAGE_SHAPE,))
    check(x.dtype == np.float32, "预处理输出 float32")
    check(float(x.min()) >= -2.01 - 1e-6 and float(x.max()) <= 2.01 + 1e-6,
          "归一化后数值落在合理范围（约 -2 ~ 2）")

    # 类别名与中文名一一对应
    check(len(CLASS_NAMES) == NUM_CLASSES, "类别名数量与类别数一致")
    missing = [c for c in CLASS_NAMES if c not in CLASS_NAMES_ZH]
    check(not missing, "每个英文类别都有中文名（缺：%s）" % (missing or "无"))
    check(len(set(CLASS_NAMES)) == len(CLASS_NAMES), "类别名无重复")

    # draw_result 不崩且尺寸不变
    try:
        from PIL import Image
        out = draw_result(img, probs)
        check(isinstance(out, Image.Image) and out.size == (48, 64),
              "draw_result 返回同尺寸的图片")
    except Exception as e:
        check(False, "draw_result 失败：%s: %s" % (type(e).__name__, e))

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
    ap = argparse.ArgumentParser(
        description="输入一张图像，输出它的类别（实验一题目要求的核心程序）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image", help="单张图片路径")
    src.add_argument("--dir", help="图片目录（批量预测）")
    ap.add_argument("--limit", type=int, default=0, help="--dir 时最多处理多少张")
    ap.add_argument("--model", default=None, help="权重路径；默认取 models/ 里最优的")
    ap.add_argument("--act", default="relu", help="与训练时一致的激活函数")
    ap.add_argument("--norm", default=DEFAULT_NORM, help="与训练时一致的规范化方式")
    ap.add_argument("--topk", type=int, default=3, help="输出前 k 个类别（默认 3）")
    ap.add_argument("--save-dir", default=None,
                    help="把带预测结果的图保存到该目录")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--val-size", type=int, default=VAL_SIZE)
    ap.add_argument("--selftest", action="store_true", help="只跑输出格式自检")
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.selftest:
        return selftest()
    if not args.image and not args.dir:
        ap.error("需要 --image 或 --dir（或用 --selftest）")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # ---------------- 收集待预测图片 ----------------
    if args.image:
        paths = [Path(args.image)]
    else:
        d = Path(args.dir)
        if not d.is_dir():
            raise SystemExit("找不到目录：%s" % d)
        paths = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if args.limit and args.limit > 0:
            paths = paths[:args.limit]
    if not paths:
        raise SystemExit("没有找到任何图片")

    model_path = Path(args.model) if args.model else pick_default_model()
    if model_path is None or not Path(model_path).is_file():
        raise SystemExit(
            "找不到权重文件。请先训练（train.py），或用 --model 指定路径。\n"
            "  当前查找位置：experiments/01_TensorFlow_AlexNet/models/*.h5")

    print("=" * 74)
    print("实验一 · AlexNet 图像分类推理")
    print("=" * 74)
    print("权重     : %s" % model_path)
    print("激活函数 : %s" % args.act)
    print("待预测   : %d 张" % len(paths))
    print("")

    # ---------------- 预处理统计量（必须与训练一致）----------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import data as data_mod

    splits = data_mod.load_splits(val_size=args.val_size, seed=args.seed,
                                 use_cache=True, quiet=True)
    mean, std = splits["mean"], splits["std"]
    names = splits.get("class_names") or CLASS_NAMES

    print("正在加载模型 ...")
    model = load_model(model_path, args.act, args.norm)
    print("  参数量 %.2f M" % (model.count_params() / 1e6))
    print("")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    n_err = 0
    for p in paths:
        print("-" * 74)
        print("图像 : %s" % p)
        try:
            img = load_image_array(p)
        except Exception as e:
            n_err += 1
            print("  [错误] 无法读取该图片：%s: %s" % (type(e).__name__, e))
            continue

        x = preprocess(img, mean, std)
        probs = model.predict(x, verbose=0)[0]
        print("  原始尺寸 %dx%d" % (img.shape[1], img.shape[0]))
        print(format_prediction(probs, names, topk=args.topk))
        top = int(np.argmax(probs))
        print("  => 类别：%s（%s），置信度 %.4f"
              % (names[top], CLASS_NAMES_ZH.get(names[top], ""), probs[top]))

        if save_dir:
            try:
                out = draw_result(img, probs, names, topk=args.topk)
                dst = save_dir / ("%s_pred.jpg" % p.stem)
                out.save(dst, quality=92)
                print("  已保存：%s" % dst)
            except Exception as e:
                print("  [警告] 保存结果图失败：%s: %s" % (type(e).__name__, e))

    print("")
    print("=" * 74)
    print("完成：成功 %d 张，失败 %d 张" % (len(paths) - n_err, n_err))
    if save_dir:
        print("结果图目录：%s" % save_dir)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
