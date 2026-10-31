#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIFAR-10 数据集的下载、解析、缓存与预处理。

设计要点
--------
1. **下载与解析部分不依赖 TensorFlow**，只用标准库 + numpy。
   这样在装 TF 之前就能先把数据准备好、把形状检查完，出问题好定位。
   只有 `make_dataset()`（tf.data 流水线）需要 TF，是延迟导入的。

2. **下载走 `datasets/classification/cifar10/`**，不放用户主目录。
   仓库约定是"数据都放在 datasets/ 下"，而且这样备份/迁移工作区时数据跟着走。
   下载后用官方 SHA256 校验（见 `config.CIFAR_HASH`）——校验很有必要，
   网络中断后拿到半个包会解析出乱七八糟的形状，报错信息还很难懂。

3. **归一化统计只来自训练集**。用全量数据算均值/方差会把测试集信息
   泄漏进训练流程，属于典型的数据泄漏。

4. 测试集 10000 张**任何环节都不参与**训练与超参选择，只在最终评测用一次。

用法
----
    python data.py                # 报告数据集状态
    python data.py --download     # 需要时下载并解析，然后报告
    python data.py --selftest     # 数据管线自检
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import shutil
import sys
import tarfile
import time
import urllib.request
import zlib
from pathlib import Path

import numpy as np

try:
    from config import (CIFAR_DIR, CIFAR_DIRNAME, CIFAR_HASH, CIFAR_URL,
                        IMAGE_SHAPE, NUM_CLASSES, PAD_SIZE, VAL_SIZE)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (CIFAR_DIR, CIFAR_DIRNAME, CIFAR_HASH, CIFAR_URL,
                        IMAGE_SHAPE, NUM_CLASSES, PAD_SIZE, VAL_SIZE)

BATCH_FILES = ["data_batch_%d" % i for i in range(1, 6)]
TEST_FILE = "test_batch"
NPZ_NAME = "cifar10.npz"

# CIFAR-10 的镜像。实测本机网络对境外站点很不稳定：
#   官方 cs.toronto.edu 0.019 MB/s，且每 8~9 秒停滞一次 -> 全量要 2 小时以上
#   Baidu BOS           0.060 MB/s                            -> 约 45 分钟
#   两个源的文件大小完全一致（170498071 字节），下载后统一用 SHA256 校验。
CIFAR_MIRRORS = [
    ("Baidu BOS", "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"),
    ("官方 cs.toronto.edu", CIFAR_URL),
]

# 单次连接"多久收不到数据"就判定为停滞并重连续传
STALL_TIMEOUT = 20
MAX_ATTEMPTS = 60          # 每次只丢一点进度，多试几次总能下完

# 断点续传时的**重叠回退字节数**（关键）。
#
# 踩过的坑：`resp.read(n)` 在超时抛异常之前，内部可能已经消费了一部分数据，
# 这些字节既没返回给我们、也拿不回来了。于是"已写入的字节数"与实际流位置
# 之间出现空洞，续传拼出来的文件**大小虽然正确、内容却是错的**
# （实测：gzip 在压缩流偏移 41943040 处报 `invalid distance code`，
#   而前 40 MB 完全正常——损坏点恰好落在一次续传边界上）。
#
# 解决办法：每次续传都**从 `have - OVERLAP` 处重新下载**，并先把文件截断到
# 该位置。多下 1 MB 换来边界绝对对齐，非常划算。
OVERLAP = 1 << 20

# 连续多少秒完全没有新增字节就放弃当前镜像（换下一个源）。
# 这个"进度看门狗"是必需的：只靠 socket 超时不够——
# 实测遇到过"重试时服务端返回 200 而不是 206"，代码会把文件截断重下，
# 于是陷入「下到 3 MB -> 停滞 -> 重试 -> 截断 -> 又到 3 MB」的死循环，
# 文件大小永远不变、进程却一直在跑。
NO_PROGRESS_GIVEUP = 180


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------

def dataset_dir() -> Path:
    """解压后的 CIFAR-10 目录。"""
    return CIFAR_DIR / CIFAR_DIRNAME


def is_downloaded() -> bool:
    """判断数据是否已经就绪（目录与 6 个批次文件都在）。"""
    d = dataset_dir()
    if not d.is_dir():
        return False
    need = BATCH_FILES + [TEST_FILE, "batches.meta"]
    return all((d / f).is_file() for f in need)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_corruption_offset(path: Path):
    """逐块解压，返回**第一个解压失败的压缩字节偏移**；完全正常则返回 None。

    为什么不能只比 SHA256：镜像站可能重新打包（gzip 头里含 mtime，
    重新压缩就会改哈希），那样"哈希不一致"并不代表文件坏了。
    反过来，**哈希一致也不能保证内容对**——本实验就遇到过
    文件大小完全正确、哈希对不上、实际解压报
    `invalid distance code` 的情况。

    真正可靠的判据是"能不能解开"。这个函数还能顺带给出损坏位置，
    从而只重下损坏之后的部分，而不必整包重来。
    """
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)      # 16+ 表示 gzip 容器
    consumed = 0
    CH = 1 << 16
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(CH)
                if not chunk:
                    break
                try:
                    d.decompress(chunk)
                except zlib.error:
                    return consumed
                consumed += len(chunk)
                if d.eof:
                    return None
    except Exception:
        return consumed
    return None


def verify_archive(path: Path) -> bool:
    """能否作为 tar.gz 正常打开。这是比哈希更贴近"能不能用"的判据。"""
    try:
        with tarfile.open(str(path), "r:gz") as tf:
            names = tf.getnames()
        return len(names) > 0
    except Exception:
        return False


def _remote_size(url: str, timeout: int = 20):
    """用 HEAD 取远端文件大小；取不到返回 None。"""
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "python-urllib/3"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.headers.get("Content-Length", 0)) or None
    except Exception:
        return None


def _fetch_resumable(url: str, dest: Path, total, quiet: bool,
                     stall_timeout: int = STALL_TIMEOUT,
                     max_attempts: int = MAX_ATTEMPTS,
                     no_progress_giveup: int = NO_PROGRESS_GIVEUP,
                     overlap: int = OVERLAP) -> bool:
    """把 url 下载到 dest，**支持断点续传、停滞重试、重叠回退与无进度放弃**。

    这套网络（境外站点）会每隔几秒就停滞一次，单次 `urlopen().read()` 循环
    几乎不可能一次跑完。做法是：

      * 每次尝试前先看 dest 已有多少字节，用 HTTP Range 从断点继续；
      * `urlopen(timeout=stall_timeout)` 的 timeout 作用于每次 recv，
        超过这个时间收不到数据就抛 socket.timeout —— 正好当作停滞探测器；
      * 捕获任何异常后，带着新的断点重试，直到文件完整。

    这样即使每 9 秒断一次，也只是多几轮重试，进度不会丢。
    返回是否最终下完。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_progress = time.time()
    last_size = dest.stat().st_size if dest.is_file() else 0
    restarts = 0

    for attempt in range(1, max_attempts + 1):
        have = dest.stat().st_size if dest.is_file() else 0
        if total and have >= total:
            return True
        if not total and have > 0:
            return True

        # 进度看门狗：长时间没有任何新增字节就放弃这个源
        if have > last_size:
            last_size, last_progress = have, time.time()
        elif time.time() - last_progress > no_progress_giveup:
            if not quiet:
                print("    [放弃] 连续 %d 秒没有任何进度（停在第 %d 字节）"
                      % (no_progress_giveup, have))
            return False

        # 重叠回退：见 OVERLAP 的说明。不能从 have 精确续传，
        # 因为上一轮 read() 超时时可能已经吞掉了若干字节。
        if have and overlap > 0:
            rewind = min(overlap, have)
            have -= rewind
            try:
                with open(dest, "r+b") as fh:
                    fh.truncate(have)
                if not quiet:
                    print("    [续传] 回退 %.1f MB 后从 %d 字节继续（保证边界对齐）"
                          % (rewind / 1048576, have))
            except Exception as e:
                if not quiet:
                    print("    [警告] 回退截断失败：%s: %s" % (type(e).__name__, e))

        headers = {"User-Agent": "python-urllib/3"}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=stall_timeout) as resp:
                # 请求了 Range 却回 200，说明该源不支持断点续传。
                # 这时只能从头下，但不能无限重来——否则会陷入死循环。
                if have and resp.status == 200:
                    restarts += 1
                    if restarts > 2:
                        if not quiet:
                            print("    [放弃] 该源反复忽略 Range 请求（已从头重来 %d 次）"
                                  % restarts)
                        return False
                    if not quiet:
                        print("    [提示] 该源不支持断点续传，从头下载（第 %d 次）" % restarts)
                    have = 0
                    dest.unlink(missing_ok=True)
                mode = "ab" if have else "wb"
                last_print = [0.0]
                with open(dest, mode) as out:
                    while True:
                        try:
                            chunk = resp.read(1 << 20)
                        except Exception as e:
                            if not quiet:
                                print("    [重试] 读取中断（%s），已得 %.1f MB，将续传"
                                      % (type(e).__name__, have / 1048576))
                            break
                        if not chunk:
                            break
                        out.write(chunk)
                        have += len(chunk)
                        if have > last_size:
                            last_size, last_progress = have, time.time()
                        if not quiet and total:
                            pct = have * 100.0 / total
                            if pct - last_print[0] >= 10:
                                last_print[0] = pct
                                print("    %5.1f%%  %6.1f / %.1f MB  (第 %d 次尝试)"
                                      % (pct, have / 1048576, total / 1048576, attempt),
                                      flush=True)
        except Exception as e:
            if not quiet:
                print("    [重试] %s: %s（已得 %.1f MB，将续传）"
                      % (type(e).__name__, str(e)[:60], have / 1048576))
        if total and have >= total:
            return True
    return bool(total and dest.is_file() and dest.stat().st_size >= total)


def download(force: bool = False, quiet: bool = False, mirror: str = None) -> Path:
    """下载并解压 CIFAR-10，返回解压后的目录。已存在则直接返回。

    依次尝试 `CIFAR_MIRRORS` 里的源，每个源都用可续传方式下载；
    下载完成后用 SHA256 校验，不通过就换下一个源。
    """
    if is_downloaded() and not force:
        if not quiet:
            print("CIFAR-10 已就绪，跳过下载：%s" % dataset_dir())
        return dataset_dir()

    CIFAR_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = CIFAR_DIR / Path(CIFAR_URL).name
    part_path = tar_path.with_suffix(tar_path.suffix + ".part")

    if force:
        tar_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    mirrors = CIFAR_MIRRORS
    if mirror:
        mirrors = [(m, u) for m, u in CIFAR_MIRRORS if mirror.lower() in m.lower() or mirror == u]
        if not mirrors:
            mirrors = [(mirror, mirror)]

    # ------------------------------------------------------------------
    # 0) 已有 tar.gz：先做**内容级**校验（能不能解开），而不是只看哈希
    # ------------------------------------------------------------------
    # 为什么要这样：镜像站可能重新打包（gzip 头含 mtime，重压就改哈希），
    # "哈希不一致"不等于坏了；反过来本实验真遇到过**大小完全正确、
    # 哈希对不上、实际解压报 invalid distance code** 的坏包。
    # 所以判据是"能不能解开"，解不开就回退到最后一个完好位置续传。
    if tar_path.is_file() and not force:
        if not quiet:
            print("已存在压缩包，正在校验能否解压：%s" % tar_path)
        off = find_corruption_offset(tar_path)
        if off is None:
            if not quiet:
                print("  压缩包完好，跳过下载")
        else:
            # 回退一段重叠区，把坏包降级成"待续传的部分文件"
            keep = max(0, off - OVERLAP)
            if not quiet:
                print("  [损坏] 压缩流在 %d 字节（%.2f MB）处损坏" % (off, off / 1048576))
                print("         回退到 %d 字节（%.2f MB）后继续下载剩余部分"
                      % (keep, keep / 1048576))
            if part_path.is_file():
                part_path.unlink()
            try:
                with open(tar_path, "r+b") as fh:
                    fh.truncate(keep)
                tar_path.replace(part_path)
            except Exception as e:
                raise SystemExit("回退截断失败：%s: %s\n请手动删除后重新下载：%s"
                                 % (type(e).__name__, e, tar_path))

    if not tar_path.is_file():
        t0 = time.time()
        ok = False
        # 还原现场：如果只有 .part，就直接从它续传
        for name, url in mirrors:
            if not quiet:
                print("")
                print("正在下载 CIFAR-10（源：%s）" % name)
                print("  地址 : %s" % url)
                print("  保存 : %s" % tar_path)
            total = _remote_size(url)
            if total and not quiet:
                print("  大小 : %.1f MB" % (total / 1048576))
            if _fetch_resumable(url, part_path, total, quiet):
                if not verify_archive(part_path):
                    # 下满了但解不开：多半还是有字节错位，降级重试
                    off = find_corruption_offset(part_path)
                    if off is None:
                        off = 0
                    keep = max(0, off - OVERLAP)
                    if not quiet:
                        print("  [警告] 下载完成但无法解压（损坏于 %d 字节），"
                              "回退到 %d 字节重试" % (off, keep))
                    try:
                        with open(part_path, "r+b") as fh:
                            fh.truncate(keep)
                    except Exception:
                        pass
                    continue
                part_path.replace(tar_path)
                ok = True
                if not quiet:
                    print("  下载完成，用时 %.1f 分钟" % ((time.time() - t0) / 60))
                break
            if not quiet:
                print("  [失败] 该源没能下完，换下一个源（已下载的部分保留，可续传）")

        if not ok:
            raise SystemExit(
                "所有镜像都没能下载完 CIFAR-10。\n"
                "  已下载的部分保存在：%s\n"
                "  重新运行本命令会**从断点继续**，不会白费。\n"
                "  也可以手动下载后放到：%s\n"
                "    官方：%s\n"
                "    镜像：%s"
                % (part_path, tar_path, CIFAR_URL, CIFAR_MIRRORS[0][1]))

    # ------------------------------------------------------------------
    # 校验：先看能不能解开（主判据），再比对 SHA256（参考信息）
    # ------------------------------------------------------------------
    if not quiet:
        print("")
        print("正在校验压缩包（解压测试 + SHA256，约需十几秒）...")
    if not verify_archive(tar_path):
        off = find_corruption_offset(tar_path)
        raise SystemExit(
            "CIFAR-10 压缩包无法解压！\n"
            "  损坏位置：%s 字节\n"
            "  重新运行本命令会**自动回退到最后一个完好位置并续传**：\n"
            "    python data.py --download\n"
            "  也可以直接删掉后重下：%s"
            % ("%d（%.2f MB）" % (off, off / 1048576) if off is not None else "未知",
               tar_path))

    got_hash = _sha256(tar_path)
    if CIFAR_HASH and got_hash != CIFAR_HASH:
        # 内容能解开就说明能用；哈希不同通常是镜像重新打包所致
        if not quiet:
            print("  解压测试通过")
            print("  [提示] SHA256 与官方值不同：")
            print("         官方：%s" % CIFAR_HASH)
            print("         实际：%s" % got_hash)
            print("         压缩包能正常解开，且下一步会逐批次校验内容，")
            print("         因此判定为「镜像重新打包」，可以继续使用。")
    elif not quiet:
        print("  解压测试通过，SHA256 与官方值一致")

    if not quiet:
        print("正在解压 ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        # Python 3.12 起需要显式声明 filter，用 data 最稳妥
        try:
            tf.extractall(CIFAR_DIR, filter="data")
        except TypeError:
            tf.extractall(CIFAR_DIR)

    if not is_downloaded():
        raise SystemExit("解压后仍找不到批次文件，目录：%s" % dataset_dir())

    # 内容级校验：形状、数量、标签范围。哈希一致不代表数据真的对，
    # 这一步才真正回答"能不能用来训练"。
    if not quiet:
        print("正在校验数据内容 ...")
    problems = validate_content(quiet=quiet)
    if problems:
        raise SystemExit(
            "CIFAR-10 内容校验失败：\n  - %s\n"
            "  数据可能已损坏，请删除后重新下载：%s"
            % ("\n  - ".join(problems), CIFAR_DIR))

    if not quiet:
        print("数据就绪：%s" % dataset_dir())
    return dataset_dir()


def validate_content(quiet: bool = True):
    """逐批次检查数据内容，返回问题列表（空列表表示没问题）。

    检查项：每批 10000 张、形状 (10000, 32, 32, 3) uint8、
    标签在 0~9 之间、`batches.meta` 的类别数是 10。
    只读每个批次的前若干张做快速验证，避免每次都把 6 个 pickle 全解一遍。
    """
    problems = []
    d = dataset_dir()
    if not d.is_dir():
        return ["解压目录不存在：%s" % d]

    expected = {"data_batch_%d" % i: 10000 for i in range(1, 6)}
    expected[TEST_FILE] = 10000

    for name, n in expected.items():
        p = d / name
        if not p.is_file():
            problems.append("缺少 %s" % name)
            continue
        try:
            with open(p, "rb") as fh:
                obj = pickle.load(fh, encoding="latin1")
            data = obj["data"]
            labels = np.asarray(obj["labels"])
            if data.shape != (n, 3072):
                problems.append("%s 的 data 形状为 %s，应为 (%d, 3072)"
                                % (name, data.shape, n))
            if labels.shape != (n,):
                problems.append("%s 的 labels 长度为 %d，应为 %d"
                                % (name, len(labels), n))
            elif labels.min() < 0 or labels.max() > 9:
                problems.append("%s 的标签范围 %d~%d 超出 0~9"
                                % (name, labels.min(), labels.max()))
        except Exception as e:
            problems.append("%s 解析失败：%s: %s" % (name, type(e).__name__, e))

    meta = d / "batches.meta"
    if not meta.is_file():
        problems.append("缺少 batches.meta")
    else:
        try:
            with open(meta, "rb") as fh:
                names = pickle.load(fh, encoding="latin1").get("label_names", [])
            if len(names) != NUM_CLASSES:
                problems.append("batches.meta 里类别数为 %d，应为 %d"
                                % (len(names), NUM_CLASSES))
            elif not quiet:
                print("  类别名：%s" % ", ".join(str(s) for s in names))
        except Exception as e:
            problems.append("batches.meta 解析失败：%s: %s" % (type(e).__name__, e))

    if not quiet and not problems:
        print("  批次文件 6 个，各 10000 张，标签范围正常")
    return problems


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------

def _read_batch(path: Path):
    """读一个 CIFAR-10 批次文件，返回 (x, y)。

    文件是 pickle 出来的 dict：`b'data'` 为 (10000, 3072) uint8（通道在前），
    `b'labels'` 为 10000 个 0-9 的标签。
    """
    with open(path, "rb") as fh:
        # CIFAR-10 是 Python 2 时代 pickle 的，必须用 latin1
        d = pickle.load(fh, encoding="latin1")
    x = d["data"].reshape(-1, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    x = x.transpose(0, 2, 3, 1)                     # -> (N, 32, 32, 3)
    y = np.asarray(d["labels"], dtype=np.uint8)
    return np.ascontiguousarray(x), y


def load_raw(use_cache: bool = True, quiet: bool = False):
    """加载原始 CIFAR-10，返回 (x_train, y_train, x_test, y_test)，均为 uint8。

    第一次会把 5 个训练批次 + 1 个测试批次合并后缓存成单个 `.npz`，
    之后加载只要一两秒（逐个 pickle 反序列化每次要十几秒）。
    """
    if not is_downloaded():
        raise SystemExit(
            "CIFAR-10 还没准备好。先运行：\n"
            "    python data.py --download")

    cache = CIFAR_DIR / NPZ_NAME
    if use_cache and cache.is_file():
        if not quiet:
            print("从缓存加载：%s" % cache)
        z = np.load(cache)
        return z["x_train"], z["y_train"], z["x_test"], z["y_test"]

    d = dataset_dir()
    if not quiet:
        print("正在解析批次文件（首次较慢）...")
    xs, ys = [], []
    for name in BATCH_FILES:
        x, y = _read_batch(d / name)
        xs.append(x)
        ys.append(y)
    x_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    x_test, y_test = _read_batch(d / TEST_FILE)

    if not quiet:
        print("  训练集 %s  测试集 %s" % (x_train.shape, x_test.shape))

    if use_cache:
        try:
            np.savez(cache, x_train=x_train, y_train=y_train,
                     x_test=x_test, y_test=y_test)
            if not quiet:
                print("  已缓存到 %s" % cache)
        except Exception as e:
            if not quiet:
                print("  [提示] 写缓存失败（不影响本次运行）：%s" % e)

    return x_train, y_train, x_test, y_test


def class_names() -> list:
    """从 `batches.meta` 读官方类别名，用于校验 config 里写死的顺序。"""
    meta = dataset_dir() / "batches.meta"
    if not meta.is_file():
        return []
    with open(meta, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    return [str(s) for s in d.get("label_names", [])]


# --------------------------------------------------------------------------
# 划分与归一化（纯 numpy，不依赖 TF）
# --------------------------------------------------------------------------

def split_indices(y_train, val_size: int = VAL_SIZE, seed: int = 42):
    """从训练集里固定地划出验证集，返回 (train_idx, val_idx)。

    用固定种子的置换，保证每次运行划分完全一致——四组对比必须用同一划分。
    """
    n = len(y_train)
    if not 0 < val_size < n:
        raise ValueError("val_size=%d 不合理（训练集 %d 张）" % (val_size, n))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:val_size])
    train_idx = np.sort(perm[val_size:])
    return train_idx, val_idx


def compute_norm_stats(x_train) -> tuple:
    """按通道统计训练集的均值与标准差（输入为 uint8，输出 float32 的 [0,1] 标度）。

    只允许传训练集进来：用全量数据统计会把测试集信息泄漏进训练流程。
    """
    x = x_train.astype("float32") / 255.0
    mean = x.mean(axis=(0, 1, 2))
    std = x.std(axis=(0, 1, 2))
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype("float32"), std.astype("float32")


def normalize(x, mean, std):
    """把 uint8 图像归一化到零均值单位方差。"""
    return (x.astype("float32") / 255.0 - mean) / std


# --------------------------------------------------------------------------
# tf.data 流水线（需要 TensorFlow）
# --------------------------------------------------------------------------

def make_dataset(x, y, mean, std, batch_size: int, training: bool,
                 seed: int = 42, pad: int = PAD_SIZE, shuffle_buffer: int = 10000):
    """构造 tf.data 数据集。

    training=True 时启用 CIFAR-10 的标准增强：4 像素零填充 + 随机裁剪 + 随机水平翻转。
    增强**只用于训练集**，验证/测试一律原图直送。
    """
    import tensorflow as tf

    def _prep(xb, yb):
        xb = normalize(xb.numpy(), mean, std)
        return xb.astype("float32"), yb

    def _augment(xb, yb):
        # 40x40 后再随机裁回 32x32，等价于先 pad 4 再 crop
        h, w = IMAGE_SHAPE[0], IMAGE_SHAPE[1]
        xb = tf.image.resize_with_crop_or_pad(xb, h + 2 * pad, w + 2 * pad)
        xb = tf.image.random_crop(xb, [h, w, IMAGE_SHAPE[2]], seed=seed)
        xb = tf.image.random_flip_left_right(xb, seed=seed)
        return xb, yb

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, len(x)),
                        seed=seed, reshuffle_each_iteration=True)

    def _map(xb, yb):
        xb = tf.py_function(_prep, [xb, yb], [tf.float32, yb.dtype])
        xb[0].set_shape(IMAGE_SHAPE)
        xb[1].set_shape([])
        return xb[0], xb[1]

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_splits(val_size: int = VAL_SIZE, seed: int = 42, use_cache: bool = True,
                quiet: bool = False):
    """一步到位：加载原始数据 → 划分 → 算归一化统计。

    返回 dict：x_train/y_train/x_val/y_val/x_test/y_test、mean/std、类别名。
    """
    x_train_all, y_train_all, x_test, y_test = load_raw(use_cache=use_cache, quiet=quiet)
    train_idx, val_idx = split_indices(y_train_all, val_size=val_size, seed=seed)
    x_tr, y_tr = x_train_all[train_idx], y_train_all[train_idx]
    x_va, y_va = x_train_all[val_idx], y_train_all[val_idx]
    mean, std = compute_norm_stats(x_tr)
    return {
        "x_train": x_tr, "y_train": y_tr,
        "x_val": x_va, "y_val": y_va,
        "x_test": x_test, "y_test": y_test,
        "mean": mean, "std": std,
        "class_names": class_names(),
    }


# --------------------------------------------------------------------------
# 报告与自检
# --------------------------------------------------------------------------

def report(quiet: bool = False):
    """打印数据集状态与划分规模。"""
    print("=" * 70)
    print("CIFAR-10 数据集状态")
    print("=" * 70)
    print("下载目录       : %s" % CIFAR_DIR)
    print("解压目录       : %s" % dataset_dir())
    print("是否就绪       : %s" % ("是" if is_downloaded() else "否（运行 --download）"))
    if is_downloaded():
        names = class_names()
        print("官方类别名     : %s" % (", ".join(names) if names else "（读不到 batches.meta）"))
        cache = CIFAR_DIR / NPZ_NAME
        print("npz 缓存       : %s" % ("存在" if cache.is_file() else "无"))
        if cache.is_file():
            print("  大小         : %.1f MB" % (cache.stat().st_size / 1048576))
    print("")
    print("划分（val_size=%d，seed=42）：" % VAL_SIZE)
    print("  训练 50000 - %d = %d 张" % (VAL_SIZE, 50000 - VAL_SIZE))
    print("  验证 %d 张" % VAL_SIZE)
    print("  测试 10000 张（训练与选型全程不参与）")
    print("")
    print("图像尺寸       : %s" % (IMAGE_SHAPE,))
    print("类别数         : %d" % NUM_CLASSES)
    _ = quiet
    return 0


def selftest() -> int:
    """数据管线自检：形状、类型、标签范围、划分不重叠、归一化只来自训练集。"""
    fails = []

    def check(cond, msg):
        print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
        if not cond:
            fails.append(msg)

    print("数据管线自检（用合成数据，不需要真的下载 CIFAR-10）")
    print("-" * 70)

    rng = np.random.RandomState(0)
    x = rng.randint(0, 256, size=(600, 32, 32, 3), dtype=np.uint8)
    y = rng.randint(0, 10, size=(600,)).astype(np.uint8)

    check(x.shape == (600, 32, 32, 3), "合成图像形状正确")
    check(y.dtype == np.uint8, "标签类型为 uint8")

    tr, va = split_indices(y, val_size=100, seed=42)
    check(len(tr) == 500 and len(va) == 100, "划分数量正确 (500/100)")
    check(len(set(tr.tolist()) & set(va.tolist())) == 0, "训练集与验证集无重叠")
    check(sorted(tr.tolist() + va.tolist()) == list(range(600)),
          "训练集与验证集并集覆盖全部样本")

    tr2, va2 = split_indices(y, val_size=100, seed=42)
    check(np.array_equal(tr, tr2) and np.array_equal(va, va2),
          "同一 seed 下划分可复现")

    tr3, _ = split_indices(y, val_size=100, seed=7)
    check(not np.array_equal(tr, tr3), "不同 seed 下划分不同")

    mean, std = compute_norm_stats(x[tr])
    check(mean.shape == (3,) and std.shape == (3,), "归一化统计量形状为 (3,)")
    check(bool(np.all(std > 0)), "标准差全为正（无除零）")

    xn = normalize(x[tr][:16], mean, std)
    check(xn.dtype == np.float32, "归一化结果为 float32")
    check(abs(float(xn.mean())) < 0.05, "训练集归一化后均值接近 0")

    # 归一化统计只取自训练集：用不同子集算出的统计量应当不同
    mean2, _ = compute_norm_stats(x[va])
    check(not np.allclose(mean, mean2, atol=1e-4),
          "只传训练集时统计量随子集变化（说明没有写死全量统计）")

    try:
        split_indices(y, val_size=0)
        check(False, "val_size=0 应当报错")
    except ValueError:
        check(True, "val_size 非法时抛 ValueError")

    print("-" * 70)
    if fails:
        print("失败 %d 项：" % len(fails))
        for f in fails:
            print("  - %s" % f)
        return 1
    print("全部通过。")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="CIFAR-10 数据准备与检查")
    ap.add_argument("--download", action="store_true", help="需要时下载并解析")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    ap.add_argument("--mirror", default=None,
                    help="只用一个镜像（名字里含该字符串即可，或直接给 URL）")
    ap.add_argument("--selftest", action="store_true", help="跑数据管线自检（不需要下载）")
    ap.add_argument("--no-cache", action="store_true", help="不使用 npz 缓存，直接解析 pickle")
    args = ap.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.selftest:
        return selftest()

    if args.download:
        download(force=args.force, mirror=args.mirror)
        x_tr, y_tr, x_te, y_te = load_raw(use_cache=not args.no_cache)
        print("")
        print("训练集 : %s  %s" % (x_tr.shape, x_tr.dtype))
        print("测试集 : %s  %s" % (x_te.shape, x_te.dtype))
        print("标签范围: %d ~ %d" % (int(y_tr.min()), int(y_tr.max())))
        tr, va = split_indices(y_tr)
        print("划分后 : train %d / val %d" % (len(tr), len(va)))
        mean, std = compute_norm_stats(x_tr[tr])
        print("均值   : %s" % np.round(mean, 4))
        print("标准差 : %s" % np.round(std, 4))
        print("")
    return report()


if __name__ == "__main__":
    sys.exit(main())
