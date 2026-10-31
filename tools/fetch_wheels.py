# -*- coding: utf-8 -*-
"""直接从清华 PyPI 镜像下载 wheel 并解压到目标目录（绕过 pip 在沙箱中的权限问题）。

用法:
    python tools/fetch_wheels.py --target <dir> h5py==3.16.0 prettytable==3.16.0 ...

只选择与本机 Python 版本/平台匹配的 wheel，解压后不写 .dist-info 之外的钩子，纯解压安装。
"""
import argparse
import io
import os
import re
import sys
import urllib.parse
import urllib.request
import zipfile

MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
UA = {"User-Agent": "pip/24.0"}


def simple_links(name):
    url = f"{MIRROR}/{name}/"
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read().decode()
    out = []
    for href, fn in re.findall(r'href="([^"]+)"[^>]*>([^<]+)</a>', html):
        out.append((urllib.parse.urljoin(url, href.split('#')[0]), fn))
    return out


def py_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def pick(name, version, links):
    """在候选文件里挑出与当前解释器匹配的 wheel。"""
    want = f"{name.replace('-', '_')}-{version}"
    cands = []
    for url, fn in links:
        if not fn.endswith(".whl"):
            continue
        if not fn.replace('-', '_').lower().startswith(want.lower().replace('-', '_')):
            continue
        plat = "win_amd64" if sys.platform == "win32" else "manylinux"
        ok = ("none-any" in fn) or (py_tag() in fn and plat in fn) or ("abi3" in fn and plat in fn)
        if ok:
            cands.append((url, fn))
    if not cands:
        return None
    # 优先 pure-python，其次平台 wheel
    cands.sort(key=lambda x: ("none-any" not in x[1], len(x[1])))
    return cands[0]


def install(name, version, target, force=False):
    links = simple_links(name)
    hit = pick(name, version, links)
    if hit is None:
        print(f"  [skip] {name}=={version}: 没有匹配 {py_tag()}/win_amd64 的 wheel")
        return False
    url, fn = hit
    print(f"  [get ] {fn}")
    blob = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        if not force:
            # 已安装同名 dist-info 则跳过
            di = [n for n in names if n.endswith(".dist-info/METADATA")]
            if di:
                d = os.path.join(target, di[0].split("/")[0])
                if os.path.isdir(d):
                    print(f"  [have] {os.path.basename(d)} 已存在，跳过")
                    return True
        z.extractall(target)
    print(f"  [ok  ] {fn} -> {target}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("pkgs", nargs="+", help="name==version")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.target, exist_ok=True)
    ok = True
    for spec in a.pkgs:
        name, _, ver = spec.partition("==")
        if not ver:
            print("需要 name==version 形式:", spec)
            ok = False
            continue
        ok = install(name, ver, a.target, a.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
