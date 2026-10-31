#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""获取实验 01 的基础代码：keras-yolo3（指导书指定的代码来源）。

指导书 1.1 节：
> 本实验以 yolo3 代码为基础
> 下载代码：https://github.com/qqwweee/keras-yolo3

为什么不直接用 `git clone`：本机到 github.com 的 git 协议不稳定，而
raw 文件下载是通的。本脚本按文件清单逐个取，并**记录 SHA256**，
便于确认拿到的是哪一份代码（上游仓库长期未更新，但记录哈希仍然有意义）。

上游仓库：https://github.com/qqwweee/keras-yolo3  （MIT License）
下载的文件版权归原作者，`LICENSE` 会一并取回放到 code/ 目录下。

用法：
    python fetch_base_code.py                  # 下载到 code/ 下
    python fetch_base_code.py --force          # 已存在也重新下载
    python fetch_base_code.py --list           # 只列出清单和当前状态
    python fetch_base_code.py --dest <目录>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    from config import CODE_DIR, MODEL_DATA_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import CODE_DIR, MODEL_DATA_DIR

REPO = "qqwweee/keras-yolo3"
REF = "master"

# 两个来源都试：优先 raw.githubusercontent，失败则用 jsDelivr 的 GitHub 镜像
MIRRORS = [
    "https://raw.githubusercontent.com/%s/%s/" % (REPO, REF),
    "https://cdn.jsdelivr.net/gh/%s@%s/" % (REPO, REF),
]

# (仓库内路径, 本地目标路径)  —— 目标默认相对 code/
# 目标路径写成 "model_data/xxx" 时会自动落到 code/model_data/ 下
FILES = [
    ("LICENSE",                                   "LICENSE"),
    ("README.md",                                 "UPSTREAM_README.md"),
    ("yolov3.cfg",                                "model_data/yolov3.cfg"),
    ("model_data/yolo_anchors.txt",               "model_data/yolo_anchors.txt"),
    ("model_data/tiny_yolo_anchors.txt",          "model_data/tiny_yolo_anchors.txt"),
    ("model_data/voc_classes.txt",                "model_data/voc_classes_upstream.txt"),
    ("model_data/coco_classes.txt",               "model_data/coco_classes.txt"),
    ("yolo3/__init__.py",                         "yolo3/__init__.py"),
    ("yolo3/model.py",                            "yolo3/model.py"),
    ("yolo3/utils.py",                            "yolo3/utils.py"),
    # yolo.py 的 detect_image 里写死了 font/FiraMono-Medium.otf，
    # 缺这个文件会直接 FileNotFoundError，必须一起取回
    ("font/FiraMono-Medium.otf",                  "font/FiraMono-Medium.otf"),
    ("yolo.py",                                   "yolo.py"),
    ("yolo_video.py",                             "yolo_video_upstream.py"),
    ("train.py",                                  "train_upstream.py"),
    ("voc_annotation.py",                         "voc_annotation_upstream.py"),
    ("kmeans.py",                                 "kmeans.py"),
]

# 这些是上游原文件，本仓库提供了自己的替代版本，所以改名保存，避免被覆盖
REPLACED_BY_US = {
    "UPSTREAM_README.md": "上游 README（参考用；本实验的说明见 ../README.md）",
    "voc_annotation_upstream.py": "上游版本会生成 2007_train.txt 并要求手动改名，本仓库用 voc_annotation.py 替代",
    "train_upstream.py": "上游训练脚本；指导书要求用其给出的版本替换，见 train.py",
    "yolo_video_upstream.py": "上游入口；检测图片时用的 detect_img 需按指导书 2.2 节改写，见 detect_images.py",
    "model_data/voc_classes_upstream.txt": "上游 VOC 类别表（20 类）；本实验只需 cat，见 voc_classes.txt",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fetch_base_code)"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        if resp.status != 200:
            raise urllib.error.HTTPError(url, resp.status, "unexpected status", resp.headers, None)
        return resp.read()


def fetch(rel_path: str) -> tuple:
    """按镜像顺序尝试，返回 (bytes, 实际使用的 URL)。"""
    errors = []
    for base in MIRRORS:
        url = base + rel_path
        try:
            return download(url), url
        except Exception as e:
            errors.append("%s -> %s: %s" % (base.split("/")[2], type(e).__name__, e))
    raise RuntimeError("; ".join(errors))


def target_path(local: str, dest: Path) -> Path:
    if local.startswith("model_data/"):
        return MODEL_DATA_DIR / local[len("model_data/"):]
    return dest / local


def main():
    ap = argparse.ArgumentParser(description="下载 keras-yolo3 基础代码")
    ap.add_argument("--dest", default=str(CODE_DIR), help="目标目录（默认 code/）")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--list", action="store_true", help="只列出清单与当前状态")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    dest = Path(args.dest).resolve()
    print("=" * 70)
    print("获取 keras-yolo3 基础代码")
    print("=" * 70)
    print("上游仓库 : https://github.com/%s  (MIT License)" % REPO)
    print("版本     : %s" % REF)
    print("目标目录 : %s" % dest)
    print("")

    if args.list:
        print("%-42s %s" % ("文件", "状态"))
        print("-" * 70)
        for rel, local in FILES:
            p = target_path(local, dest)
            if p.is_file():
                print("%-42s 已存在  %8d B" % (local, p.stat().st_size))
            else:
                print("%-42s 缺失" % local)
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ok = skipped = failed = 0
    manifest = []

    for rel, local in FILES:
        p = target_path(local, dest)
        p.parent.mkdir(parents=True, exist_ok=True)

        if p.is_file() and not args.force:
            size = p.stat().st_size
            print("  跳过  %-42s 已存在（%d B）" % (local, size))
            skipped += 1
            manifest.append({"file": local, "source": rel, "bytes": size,
                             "sha256": sha256_of(p), "action": "skipped"})
            continue

        try:
            data, used = fetch(rel)
        except Exception as e:
            print("  失败  %-42s %s" % (local, e))
            failed += 1
            manifest.append({"file": local, "source": rel, "action": "failed",
                             "error": str(e)})
            continue

        p.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        note = REPLACED_BY_US.get(local)
        print("  下载  %-42s %7d B  sha256=%s" % (local, len(data), digest[:12]))
        if note:
            print("        └ 说明：%s" % note)
        ok += 1
        manifest.append({"file": local, "source": rel, "bytes": len(data),
                         "sha256": digest, "url": used, "action": "downloaded"})

    # ---------------- 清单 ----------------
    out = dest / "model_data" / "upstream_manifest.json"
    payload = {"repo": "https://github.com/%s" % REPO, "ref": REF,
               "license": "MIT", "files": manifest}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("-" * 70)
    print("下载 %d 个，跳过 %d 个，失败 %d 个" % (ok, skipped, failed))
    print("哈希清单：%s" % out)

    if failed:
        print("")
        print("[警告] 有文件下载失败。检查网络后重试，或手动从")
        print("       https://github.com/%s 下载后放到对应位置。" % REPO)
        return 1

    print("")
    print("=" * 70)
    print("下一步")
    print("=" * 70)
    print("1) 修改 cfg 的类别数与 filters：")
    print("     python patch_yolov3_cfg.py --in-place --random 0")
    print("2) 检查数据是否就绪：")
    print("     python make_imagesets.py && python voc_annotation.py")
    print("3) 训练（需要 tensorflow + keras 环境）：")
    print("     python train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
