#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""建立实验 01 的 TensorFlow 环境。

======================================================================
为什么用 Python 3.10 + TensorFlow 2.10.0（而不是指导书的 3.8 + 2.3.0）
======================================================================
指导书环境一是 python 3.7/3.8 + tensorflow 2.3.0 + keras 2.4.3。最初按这个装，
但在这台机器上装不下去：

  * venv 里 Python 3.8 的 OpenSSL 1.1.1 访问 pypi.org / 国内镜像时**间歇性 TLS 握手失败**，
    报 `SSLError(142, '[ASN1: NOT_ENOUGH_DATA] not enough data')`（证书数据被截断）。
    同一批 URL 用 Python 3.10 / 3.12 访问完全正常，说明不是网络问题，是 3.8 的 SSL 栈。
  * 换明文 HTTP 源也不行：镜像会把 http 重定向到 443，照样走 TLS。

于是改用 Python 3.10 + tensorflow 2.10.0：
  * tensorflow 2.10.0 提供 cp310 的 Windows 轮子，且**仍然是 Keras 2**
    （`fit_generator`、`keras.backend` 都还在），keras-yolo3 的改动最小；
  * Python 3.10 的 TLS 栈在本机工作正常。

仍然只能 CPU 训练：RTX 4060 是 sm_89（Ada），需要 CUDA 11.8+，
而 tensorflow 2.10.0 绑定 CUDA 11.2（最高 sm_86）；TensorFlow 自 2.11 起
取消了 Windows 原生 GPU 支持。

为什么不用 conda：conda 需要写 C:\\Users\\<用户>\\AppData\\Local\\conda-anaconda-tos
缓存，会报 CondaToSPermissionError。venv 更省事。

用法：
    python envs/setup_tf_env.py
    python envs/setup_tf_env.py --recreate
"""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = REPO_ROOT / "envs" / "pcd01_tf"
# 用 pytorch 环境的 Python 3.10（它的 pip 与 TLS 都正常）
# 用哪个解释器来建 venv：优先 --base 参数，其次环境变量 PCD_BASE_PYTHON，
# 再退回到 PATH 上的 python
DEFAULT_BASE = os.environ.get("PCD_BASE_PYTHON") or "python"

GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

PACKAGES = [
    # tensorflow 2.10.0 是最后一个带完整 Keras 2 API 的版本之一，
    # 且 Keras 2 的 fit_generator / keras.backend 都还在，keras-yolo3 改动最小
    "tensorflow==2.10.0",
    "keras==2.10.0",
    # 其余本实验需要的包
    "Pillow>=9.0.0",
    "matplotlib>=3.5.0",
    "lxml>=4.9.0",
    "tqdm>=4.64.0",
    "opencv-python>=4.5.5.64",
    "h5py>=3.6.0",
]

# 支持的 Python 版本：tensorflow 2.10 的 Windows 轮子是 cp37~cp310
SUPPORTED_PY = ("3.9", "3.10")

# pip 源。官方源即可 —— Python 3.10 的 TLS 栈在本机正常。
# 若哪天官方源慢，可换成国内镜像，例如：
#   --index-url http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
DEFAULT_INDEX_URL = "https://pypi.org/simple"
DEFAULT_TRUSTED_HOST = "pypi.org"


def run(cmd, **kw):
    """执行命令，stdio 继承（不用管道，避免受限环境下的 EPERM）。"""
    print("      $ %s" % " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], **kw)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def has_module(py: Path, module: str) -> bool:
    return subprocess.call([str(py), "-c", "import %s" % module],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


def download(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120,
                               context=ssl.create_default_context()) as r:
        data = r.read()
    dest.write_bytes(data)
    return len(data)


def main():
    ap = argparse.ArgumentParser(description="建立实验 01 的 TensorFlow 环境")
    ap.add_argument("--venv", default=str(DEFAULT_VENV), help="venv 目录")
    ap.add_argument("--base-python", default=DEFAULT_BASE,
                    help="基础解释器（需要 Python 3.8）")
    ap.add_argument("--recreate", action="store_true", help="删除已有 venv 后重建")
    ap.add_argument("--index-url", default=DEFAULT_INDEX_URL,
                    help="pip 源（默认清华 TUNA 镜像）")
    ap.add_argument("--trusted-host", default=DEFAULT_TRUSTED_HOST,
                    help="配合 --index-url 的 trusted-host")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    venv = Path(args.venv).resolve()
    base = Path(args.base_python).resolve()
    vpy = venv_python(venv)

    print("=" * 70)
    print("建立实验 01 的 TensorFlow 环境")
    print("=" * 70)
    print("  目标目录   : %s" % venv)
    print("  基础解释器 : %s" % base)
    print("  pip 源     : %s" % args.index_url)
    print("")

    # ---------- 0. 基础解释器检查 ----------
    if not base.is_file():
        print("[错误] 找不到基础 Python：%s" % base)
        print("       用 --base-python 指向一个 Python %s（tensorflow 2.10.0 "
              "的 Windows 轮子只到 cp310）。" % " 或 ".join(SUPPORTED_PY))
        return 1
    ver = subprocess.check_output(
        [str(base), "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
        encoding="utf-8").strip()
    print("  基础版本   : Python %s" % ver)
    if ver not in SUPPORTED_PY:
        print("[错误] tensorflow 2.10.0 的 Windows 轮子支持 %s，当前是 %s。"
              % (" / ".join(SUPPORTED_PY), ver))
        print("       请用 Python 3.10 解释器运行本脚本，例如：")
        print("       python setup_tf_env.py --base <Python3.10 的 python.exe>")
        return 1
    print("")

    # ---------- 1. 建 venv ----------
    if args.recreate and venv.exists():
        print("[1/4] 删除已有 venv ...")
        shutil.rmtree(venv, ignore_errors=True)
        if venv.exists():
            # ignore_errors=True 会吞掉删除失败，导致后面误判"venv 已存在"而跳过重建，
            # 于是继续用一个版本不对的环境。这里必须显式报错。
            print("[错误] 删除 %s 失败（可能残留了只读或受保护的权限项）。" % venv)
            print("       Windows 上可先执行：")
            print("         takeown /f \"%s\" /r /d y" % venv)
            print("         icacls \"%s\" /reset /t /c" % venv)
            print("       然后删掉该目录再重跑本脚本。")
            return 1

    if vpy.is_file():
        print("[1/4] venv 已存在，跳过创建")
    else:
        print("[1/4] 创建 venv ...")
        venv.mkdir(parents=True, exist_ok=True)
        if run([base, "-m", "venv", venv]) != 0:
            print("[错误] venv 创建失败。")
            return 1
    if run([vpy, "-c", "import sys;print('      venv python', sys.version.split()[0])"]) != 0:
        print("[错误] venv 里的 python 无法执行。")
        return 1

    # ---------- 2. 引导 pip ----------
    print("[2/4] 检查 pip ...")
    if has_module(vpy, "pip"):
        print("      pip 已就绪")
    else:
        print("      venv 里没有 pip，用 get-pip.py 引导")
        tmp = venv / ".tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        # 把临时目录限制在 venv 内，避免系统临时目录的权限问题
        os.environ["TEMP"] = str(tmp)
        os.environ["TMP"] = str(tmp)

        get_pip = tmp / "get-pip.py"
        try:
            n = download(GET_PIP_URL, get_pip)
            print("      get-pip.py 下载完成：%d 字节" % n)
        except Exception as e:
            print("[错误] get-pip.py 下载失败：%s: %s" % (type(e).__name__, e))
            return 1
        if run([vpy, get_pip, "--no-warn-script-location"]) != 0:
            print("[错误] get-pip.py 执行失败。")
            return 1

    if run([vpy, "-m", "pip", "--version"]) != 0:
        print("[错误] pip 不可用。")
        return 1

    # ---------- 3. 安装依赖 ----------
    print("[3/4] 安装依赖（首次约需下载 500MB，请耐心等待）...")
    os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    os.environ["PIP_DEFAULT_TIMEOUT"] = "120"
    cmd = [vpy, "-m", "pip", "install", "--no-warn-script-location",
           "-i", args.index_url,
           "--trusted-host", args.trusted_host,
           *PACKAGES]
    if run(cmd) != 0:
        print("")
        print("[错误] 依赖安装失败。常见原因：")
        print("  * 网络中断 —— 直接重跑本脚本，已装好的包会跳过")
        print("  * 找不到轮子 —— 确认基础解释器是 Python 3.8")
        print("  * 某个固定版本在 PyPI 上不存在 —— 按报错调整 PACKAGES 里的版本")
        return 1

    # ---------- 4. 验证 ----------
    print("")
    print("[4/4] 验证安装 ...")
    verify = (
        "import os,sys;"
        "os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');"
        "sys.stdout.reconfigure(encoding='utf-8',errors='replace');"
        "import tensorflow as tf, keras, numpy, PIL, cv2, matplotlib, h5py;"
        "print('  tensorflow  %s' % tf.__version__);"
        "print('  keras       %s' % keras.__version__);"
        "print('  numpy       %s' % numpy.__version__);"
        "print('  h5py        %s' % h5py.__version__);"
        "print('  Pillow      %s' % PIL.__version__);"
        "print('  opencv      %s' % cv2.__version__);"
        "print('  matplotlib  %s' % matplotlib.__version__);"
        "print('  GPU 可见设备: %s' % tf.config.list_physical_devices('GPU'));"
        "print('  (CPU 版 tensorflow，GPU 为空是正常的)')"
    )
    if run([vpy, "-c", verify]) != 0:
        print("[错误] 验证失败。")
        return 1

    # 导出依赖清单，便于复现
    freeze = venv.parent / "pcd01_tf_freeze.txt"
    with open(freeze, "w", encoding="utf-8") as fh:
        subprocess.call([str(vpy), "-m", "pip", "freeze"], stdout=fh)
    print("")
    print("  依赖清单已导出：%s" % freeze)

    print("")
    print("=" * 70)
    print("  环境就绪")
    print("=" * 70)
    print("  venv python : %s" % vpy)
    print("")
    print("  用法示例（注意用这个 python，不是系统的）：")
    print("    \"%s\" experiments\\01_ObjectDetection_YOLOv3\\code\\train.py --benchmark 10"
          % vpy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
