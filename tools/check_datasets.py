#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集自检工具 —— 检查 4 个实验的数据是否放好、命名是否合规。

只依赖 Python 标准库，所以**在还没装任何第三方库时就能跑**：

    python tools/check_datasets.py
    python tools/check_datasets.py --only cat_face
    python tools/check_datasets.py --root datasets

检查项：
  01 cat_face         图片/标注配对、命名是否含中文、ImageSets 索引、XML 标签名是否统一
  02 tracking_videos  视频与真值配对、真值文件格式（OTB / VOT / MOT）
  03 TVSum            视频数量、标注 tsv 是否可解析、特征文件
  04 retrieval        图像/视频/查询图数量、类别目录结构、相关性真值

退出码：0 = 全部通过（或仅有提示）；1 = 存在错误。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, OrderedDict

# --------------------------------------------------------------------------
# 输出编码：Windows 控制台默认是 GBK(cp936)，中文容易乱码。
# 统一强制 UTF-8 输出；若终端仍是乱码，先执行 `chcp 65001` 切换代码页。
# --------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --------------------------------------------------------------------------
# 输出小工具
# --------------------------------------------------------------------------

IS_TTY = sys.stdout.isatty()


def _c(text, code):
    if not IS_TTY:
        return text
    return "\033[%sm%s\033[0m" % (code, text)


def ok(msg):
    print("  " + _c("[ OK ]", "32") + " " + msg)


def warn(msg):
    print("  " + _c("[WARN]", "33") + " " + msg)
    STATE["warn"] += 1


def err(msg):
    print("  " + _c("[FAIL]", "31") + " " + msg)
    STATE["err"] += 1


def info(msg):
    print("  " + _c("[INFO]", "36") + " " + msg)


def head(title):
    print("")
    print(_c("=" * 74, "1"))
    print(_c("  " + title, "1"))
    print(_c("=" * 74, "1"))


STATE = {"warn": 0, "err": 0}

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v"}


def has_cjk(s):
    return bool(CJK.search(s))


def list_files(d, exts=None):
    """列出目录下（递归）符合扩展名的文件。"""
    out = []
    if not os.path.isdir(d):
        return out
    for root, _dirs, files in os.walk(d):
        for f in files:
            if f.startswith("."):
                continue
            if exts is None or os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    return sorted(out)


def require_dir(path, label):
    if os.path.isdir(path):
        ok("%s 目录存在：%s" % (label, path))
        return True
    err("%s 目录不存在：%s" % (label, path))
    return False


# --------------------------------------------------------------------------
# 01 cat_face
# --------------------------------------------------------------------------

def check_cat_face(root):
    base = os.path.join(root, "cat_face")
    head("01 目标检测 · datasets/cat_face")

    if not os.path.isdir(base):
        err("目录不存在：%s" % base)
        return

    jpg_dir = os.path.join(base, "JPEGImages")
    ann_dir = os.path.join(base, "Annotations")
    sets_dir = os.path.join(base, "ImageSets", "Main")
    raw_dir = os.path.join(base, "raw")

    for d, label in ((jpg_dir, "JPEGImages"), (ann_dir, "Annotations"), (sets_dir, "ImageSets/Main")):
        require_dir(d, label)

    images = list_files(jpg_dir, IMAGE_EXT)
    anns = list_files(ann_dir, {".xml"})

    if not images:
        err("JPEGImages/ 里没有图片。请把猫咪图片放进去。")
    else:
        ok("JPEGImages/ 共 %d 张图片" % len(images))

    if not anns:
        err("Annotations/ 里没有 .xml 标注。请用 LabelImg 标注后放进来。")
    else:
        ok("Annotations/ 共 %d 个标注文件" % len(anns))

    # --- 命名检查：不要含中文 ---
    bad = [os.path.basename(p) for p in images + anns if has_cjk(os.path.basename(p))]
    if bad:
        err("有 %d 个文件名含中文（指导书要求避免使用汉字），例如：%s"
            % (len(bad), ", ".join(bad[:5])))
    elif images or anns:
        ok("文件名均为非中文，符合指导书要求")

    # --- 图片 / 标注配对 ---
    if images and anns:
        img_stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
        ann_stems = {os.path.splitext(os.path.basename(p))[0] for p in anns}
        only_img = sorted(img_stems - ann_stems)
        only_ann = sorted(ann_stems - img_stems)
        paired = len(img_stems & ann_stems)
        ok("图片与标注同名配对成功 %d 对" % paired)
        if only_img:
            warn("%d 张图片没有对应标注，例如：%s" % (len(only_img), ", ".join(only_img[:5])))
        if only_ann:
            warn("%d 个标注没有对应图片，例如：%s" % (len(only_ann), ", ".join(only_ann[:5])))

    # --- 图片数量建议 ---
    if images and len(images) < 230:
        warn("当前 %d 张图片，指导书实验环境用 230 张；图片偏少可能训练效果不佳" % len(images))

    # --- XML 标签名统一性 ---
    if anns:
        labels = Counter()
        parsed = 0
        for p in anns[:500]:
            try:
                tree = ET.parse(p)
            except Exception as e:
                warn("XML 解析失败：%s（%s）" % (os.path.basename(p), e))
                continue
            parsed += 1
            for obj in tree.getroot().findall("object"):
                name = obj.findtext("name")
                if name:
                    labels[name.strip()] += 1
        if parsed:
            if not labels:
                err("解析了 %d 个 XML，但没有找到任何 <object><name>，标注可能是空的" % parsed)
            else:
                detail = ", ".join("%s×%d" % (k, v) for k, v in labels.most_common())
                ok("标注标签统计（前 500 个文件）：%s" % detail)
                if len(labels) > 1:
                    warn("检测到 %d 种不同标签名。本实验只做猫脸检测，建议统一为 cat"
                         % len(labels))
                elif "cat" not in labels:
                    warn("标签名不是 cat（当前为 %s）。指导书要求统一命名为 cat"
                         % list(labels)[0])
                else:
                    ok("标签名统一为 cat，正确")

    # --- ImageSets/Main ---
    if os.path.isdir(sets_dir):
        for fn in ("trainval.txt", "train.txt", "val.txt", "test.txt"):
            p = os.path.join(sets_dir, fn)
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    n = sum(1 for line in fh if line.strip())
                ok("ImageSets/Main/%s 存在，%d 行" % (fn, n))
            else:
                warn("ImageSets/Main/%s 缺失（跑 test.py 生成）" % fn)

    # --- 下游产物：code/ 下的 train/val/test.txt ---
    repo_root = os.path.dirname(os.path.abspath(root))
    code_dir = os.path.join(repo_root, "experiments", "01_ObjectDetection_YOLOv3", "code")
    for fn in ("train.txt", "val.txt", "test.txt"):
        p = os.path.join(code_dir, fn)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                n = sum(1 for line in fh if line.strip())
            ok("code/%s 存在，%d 行（voc_annotation.py 产物）" % (fn, n))
        else:
            info("尚未生成 code/%s（运行 voc_annotation.py）" % fn)

    if os.path.isdir(raw_dir):
        raw = list_files(raw_dir)
        if raw:
            info("raw/ 里还有 %d 个暂存文件（整理完可以清掉）" % len(raw))


# --------------------------------------------------------------------------
# 02 tracking_videos
# --------------------------------------------------------------------------

def _gt_kind(path):
    """读真值文件前几行，判断是 OTB / VOT / MOT 哪种格式。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = []
            for _ in range(5):
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    lines.append(line.strip())
    except Exception:
        return "unreadable", 0
    if not lines:
        return "empty", 0

    first = lines[0]
    # MOT: frame,id,x,y,w,h,conf,...（8 列以上，且第 2 列是整数 id）
    parts = re.split(r"[,\s]+", first)
    if len(parts) >= 7:
        return "MOT", len(parts)
    if len(parts) == 4:
        # OTB 与 VOT 都是 4 列，靠目录名/文件名区分
        low = path.replace("\\", "/").lower()
        if "groundtruth" in low:
            return "VOT?", 4
        return "OTB?", 4
    if len(parts) == 8:
        return "VOT(rotated)", 8
    return "unknown(%d列)" % len(parts), len(parts)


def check_tracking(root):
    base = os.path.join(root, "tracking_videos")
    head("02 目标跟踪 · datasets/tracking_videos")

    if not os.path.isdir(base):
        err("目录不存在：%s" % base)
        return

    vdir = os.path.join(base, "videos")
    adir = os.path.join(base, "annotations")
    require_dir(vdir, "videos")
    require_dir(adir, "annotations")

    videos = list_files(vdir, VIDEO_EXT)
    if not videos:
        err("videos/ 里没有视频。请放入 .mp4/.avi（含多个运动目标）")
    else:
        ok("videos/ 共 %d 段视频" % len(videos))
        for p in videos:
            size_mb = os.path.getsize(p) / 1024.0 / 1024.0
            info("  %s  (%.1f MB)" % (os.path.basename(p), size_mb))

    # 真值：可能是 <name>.txt，或 <name>/groundtruth.txt，或 <name>/gt/gt.txt
    gt_files = list_files(adir, {".txt"})
    gt_flat = [p for p in gt_files if os.path.dirname(p) == adir]

    if not gt_files:
        warn("annotations/ 下没有任何真值文件。")
        info("没有真值就**无法计算精度/准确率指标**，只能比较 FPS 与目视稳定性。")
        info("公开数据集（OTB/VOT/MOT）自带真值；自录视频需用 code/make_gt.py 自己标。")
    else:
        ok("annotations/ 共 %d 个真值文件" % len(gt_files))
        kinds = Counter()
        for p in gt_files:
            kind, ncol = _gt_kind(p)
            kinds[kind] += 1
        ok("真值格式统计：%s" % ", ".join("%s×%d" % (k, v) for k, v in kinds.most_common()))
        for p in gt_files[:5]:
            kind, ncol = _gt_kind(p)
            info("  %s → %s（%d 列）"
                 % (os.path.relpath(p, base).replace("\\", "/"), kind, ncol))

    # 视频 / 真值配对
    if videos and gt_files:
        vstems = {os.path.splitext(os.path.basename(p))[0] for p in videos}
        gstems = set()
        for p in gt_files:
            rel = os.path.relpath(p, adir).replace("\\", "/")
            gstems.add(rel.split("/")[0].rsplit(".", 1)[0])
        unpaired_v = sorted(vstems - gstems)
        unpaired_g = sorted(gstems - vstems)
        if unpaired_v:
            warn("这些视频没有对应真值：%s" % ", ".join(unpaired_v[:5]))
        if unpaired_g:
            warn("这些真值没有对应视频：%s" % ", ".join(unpaired_g[:5]))
        if not unpaired_v and not unpaired_g:
            ok("视频与真值全部同名配对成功（%d 组）" % len(vstems))

    info("提示：本实验需要 ≥3 种算法 + 多目标 + 画出运动轨迹 + ≥2 种评价指标")


# --------------------------------------------------------------------------
# 03 TVSum
# --------------------------------------------------------------------------

def check_tvsum(root):
    base = os.path.join(root, "TVSum")
    head("03 视频摘要 · datasets/TVSum")

    if not os.path.isdir(base):
        err("目录不存在：%s" % base)
        return

    vdir = os.path.join(base, "videos")
    adir = os.path.join(base, "annotations")
    fdir = os.path.join(base, "features")
    for d, label in ((vdir, "videos"), (adir, "annotations"), (fdir, "features")):
        require_dir(d, label)

    videos = list_files(vdir, VIDEO_EXT)
    if not videos:
        err("videos/ 里没有视频。TVSum 共 50 段视频。")
    else:
        ok("videos/ 共 %d 段视频" % len(videos))
        if len(videos) < 50:
            warn("TVSum 共 50 段视频，当前只有 %d 段" % len(videos))
        else:
            ok("视频数量达到 50 段")

    # 标注
    anno = os.path.join(adir, "ydata-tvsum50-anno.tsv")
    meta = os.path.join(adir, "ydata-tvsum50-info.tsv")
    for p, label in ((anno, "ydata-tvsum50-anno.tsv"), (meta, "ydata-tvsum50-info.tsv")):
        if os.path.isfile(p):
            ok("%s 存在（%.1f KB）" % (label, os.path.getsize(p) / 1024.0))
        else:
            err("%s 缺失" % label)

    if os.path.isfile(anno):
        ids = OrderedDict()
        bad_rows = 0
        with open(anno, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().rstrip("\n")
            cols = header.split("\t")
            ok("anno 表头：%s" % " | ".join(cols))
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    bad_rows += 1
                    continue
                vid, _aid, scores = parts[0], parts[1], parts[2]
                ids.setdefault(vid, 0)
                ids[vid] += 1
        if bad_rows:
            warn("anno 中有 %d 行列数不足 3" % bad_rows)
        if ids:
            ok("anno 覆盖 %d 段视频，共 %d 条标注" % (len(ids), sum(ids.values())))
            multi = [k for k, v in ids.items() if v > 1]
            if multi:
                ok("每段视频有多个标注者（例如 %s 有 %d 条）" % (multi[0], ids[multi[0]]))

        # 视频 id 与文件是否对得上
        if videos:
            vstems = {os.path.splitext(os.path.basename(p))[0] for p in videos}
            matched = vstems & set(ids)
            ok("视频文件名与标注 video_id 匹配上 %d / %d" % (len(matched), len(vstems)))
            miss = sorted(vstems - set(ids))
            if miss:
                warn("这些视频文件名在标注里找不到对应 video_id：%s" % ", ".join(miss[:5]))
                info("可用 ydata-tvsum50-info.tsv 里的 video_id 重命名，或维护 id_map.csv")

    # 特征
    feats = list_files(fdir, {".npy", ".npz", ".h5", ".pt", ".pth"})
    if not feats:
        warn("features/ 为空。FCSN 训练需要帧级特征，先跑 extract_features.py 提取。")
    else:
        ok("features/ 共 %d 个特征文件" % len(feats))
        info("期望形状 (帧数, 特征维度)，GoogLeNet pool5 为 1024 维")

    # 测试集 10 段
    cfg = os.path.join(root, os.pardir, "experiments", "03_VideoSummarization_FCSN",
                       "code", "configs", "test10.txt")
    if os.path.isfile(cfg):
        ok("已指定 10 段测试视频：%s" % cfg)
    else:
        info("尚未指定测试用的 10 段视频（建议写到 code/configs/test10.txt）")


# --------------------------------------------------------------------------
# 04 retrieval
# --------------------------------------------------------------------------

def check_retrieval(root):
    base = os.path.join(root, "retrieval")
    head("04 视频检索 · datasets/retrieval")

    if not os.path.isdir(base):
        err("目录不存在：%s" % base)
        return

    idir = os.path.join(base, "images")
    vdir = os.path.join(base, "videos")
    qdir = os.path.join(base, "queries")
    for d, label in ((idir, "images"), (vdir, "videos"), (qdir, "queries")):
        require_dir(d, label)

    images = list_files(idir, IMAGE_EXT)
    videos = list_files(vdir, VIDEO_EXT)
    queries = list_files(qdir, IMAGE_EXT)

    for items, label, minimum in ((images, "images/ 图片底库", 500),
                                  (videos, "videos/ 视频底库", 20),
                                  (queries, "queries/ 查询图", 20)):
        if not items:
            err("%s 为空" % label)
        elif len(items) < minimum:
            warn("%s 共 %d 个，建议 ≥%d" % (label, len(items), minimum))
        else:
            ok("%s 共 %d 个" % (label, len(items)))

    # 类别目录结构
    def classes_of(d):
        c = Counter()
        for p in list_files(d):
            rel = os.path.relpath(p, d).replace("\\", "/")
            parts = rel.split("/")
            c[parts[0] if len(parts) > 1 else "(未分类)"] += 1
        return c

    for d, label in ((idir, "images"), (vdir, "videos"), (qdir, "queries")):
        if not os.path.isdir(d):
            continue
        c = classes_of(d)
        if not c:
            continue
        if set(c) == {"(未分类)"}:
            warn("%s/ 下没有类别子目录。若要自动推导相关性真值，建议按类别分子目录"
                 % label)
        else:
            ok("%s/ 类别分布：%s"
               % (label, ", ".join("%s×%d" % (k, v) for k, v in c.most_common(10))))

    # 相关性真值（可选）
    rel = os.path.join(qdir, "relevance.csv")
    if os.path.isfile(rel):
        with open(rel, "r", encoding="utf-8", errors="replace") as fh:
            header = fh.readline().rstrip("\n")
            n = sum(1 for line in fh if line.strip())
        ok("queries/relevance.csv 存在，%d 条相关性标注（表头：%s）" % (n, header))
    else:
        info("未使用人工相关性标注 queries/relevance.csv（按类别目录推导即可）")

    # 自查：查询图是否与底库图重名/同尺寸（可能是同一张）
    if images and queries:
        img_names = {os.path.basename(p) for p in images}
        q_names = {os.path.basename(p) for p in queries}
        dup = sorted(img_names & q_names)
        if dup:
            warn("有 %d 个查询图与底库图**同名**，可能是同一张图（会导致指标虚高）：%s"
                 % (len(dup), ", ".join(dup[:5])))

    info("提示：要算准确率需要有相关性真值；按类别目录构造最省事")


# --------------------------------------------------------------------------
# 01 classification（实验一 TensorFlow / AlexNet 图像分类）
# --------------------------------------------------------------------------

CIFAR_FILES = ["data_batch_%d" % i for i in range(1, 6)] + ["test_batch", "batches.meta"]
CIFAR_EXPECTED_BYTES = 170498071      # 官方 tar.gz 大小，与 data.py 的 SHA256 对应
CIFAR_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def check_classification(root):
    """检查 CIFAR-10 是否就绪。

    只做"文件在不在了、大小对不对"这类轻量检查，**不算 SHA256**
    （162 MB 全量哈希要十几秒，日常检查没必要；data.py 下载时会校验）。
    """
    base = os.path.join(root, "classification")
    head("01 图像分类 · datasets/classification")

    if not os.path.isdir(base):
        warn("目录不存在：%s" % base)
        info("实验一（TensorFlow/AlexNet）需要它。创建方式：")
        info("  python experiments/01_TensorFlow_AlexNet/code/data.py --download")
        return

    ok("目录存在：%s" % base)

    cifar = os.path.join(base, "cifar10")
    if not os.path.isdir(cifar):
        warn("尚未下载 CIFAR-10（目录不存在：%s）" % cifar)
        info("下载命令：")
        info("  python experiments/01_TensorFlow_AlexNet/code/data.py --download")
        return

    batches = os.path.join(cifar, "cifar-10-batches-py")
    missing = [f for f in CIFAR_FILES if not os.path.isfile(os.path.join(batches, f))]

    if not missing:
        ok("CIFAR-10 已就绪，批次文件齐全：%s" % batches)
    else:
        # 区分"完全没下"和"下了一半"
        part = os.path.join(cifar, "cifar-10-python.tar.gz.part")
        tar = os.path.join(cifar, "cifar-10-python.tar.gz")
        if os.path.isfile(part):
            size = os.path.getsize(part)
            warn("CIFAR-10 正在下载（断点续传中）：%.1f / %.1f MB（%.1f%%）"
                 % (size / 1048576, CIFAR_EXPECTED_BYTES / 1048576,
                    size * 100.0 / CIFAR_EXPECTED_BYTES))
            info("重新运行下面命令会**从断点继续**，已下载的部分不会白费：")
            info("  python experiments/01_TensorFlow_AlexNet/code/data.py --download")
        elif os.path.isfile(tar):
            size = os.path.getsize(tar)
            if size == CIFAR_EXPECTED_BYTES:
                ok("压缩包大小正确（%.1f MB），但尚未解压" % (size / 1048576))
                info("跑一次 data.py --download 就会校验并解压")
            else:
                err("压缩包大小异常：%.1f MB，应为 %.1f MB（可能下载不完整）"
                    % (size / 1048576, CIFAR_EXPECTED_BYTES / 1048576))
        else:
            warn("CIFAR-10 尚未下载，缺少：%s" % ", ".join(missing[:3]))
            info("下载命令：")
            info("  python experiments/01_TensorFlow_AlexNet/code/data.py --download")

    cache = os.path.join(cifar, "cifar10.npz")
    if os.path.isfile(cache):
        ok("已有解析缓存 cifar10.npz（%.1f MB），加载会很快"
           % (os.path.getsize(cache) / 1048576))
    else:
        info("还没有 cifar10.npz 缓存，首次加载要逐个解析 pickle（约十几秒）")

    info("本机下载速度实测：Baidu BOS 约 0.02~0.07 MB/s，官方源 0.019 MB/s，")
    info("所以 CIFAR-10 全量可能需要 1.5~2.5 小时。详见 datasets/classification/README.md")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

CHECKS = OrderedDict([
    ("classification", check_classification),
    ("cat_face", check_cat_face),
    ("tracking_videos", check_tracking),
    ("TVSum", check_tvsum),
    ("retrieval", check_retrieval),
])


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.join(os.path.dirname(here), "datasets")

    ap = argparse.ArgumentParser(
        description="检查 4 个实验的数据集是否放置正确（仅用标准库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n  python tools/check_datasets.py\n"
               "  python tools/check_datasets.py --only cat_face\n"
               "  python tools/check_datasets.py --root datasets --only TVSum,retrieval",
    )
    ap.add_argument("--root", default=default_root,
                    help="datasets 目录（默认：仓库下的 datasets/）")
    ap.add_argument("--only", default="",
                    help="只检查指定数据集，逗号分隔。可选：%s" % ",".join(CHECKS))
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    print(_c("数据集自检", "1"))
    print("datasets 根目录：%s" % root)

    if not os.path.isdir(root):
        print("")
        err("datasets 根目录不存在：%s" % root)
        return 1

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        err("未知的数据集名：%s（可选：%s）" % (", ".join(unknown), ", ".join(CHECKS)))
        return 1

    for n in names:
        CHECKS[n](root)

    head("汇总")
    print("  错误 %d 个，警告 %d 个" % (STATE["err"], STATE["warn"]))
    if STATE["err"] == 0 and STATE["warn"] == 0:
        ok("全部检查通过")
    elif STATE["err"] == 0:
        info("没有错误，但有需要留意的警告")
    else:
        err("请按上面的 [FAIL] 项补齐数据")
    print("")
    return 1 if STATE["err"] else 0


if __name__ == "__main__":
    sys.exit(main())
