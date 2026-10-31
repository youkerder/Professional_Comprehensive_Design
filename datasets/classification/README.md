# datasets/classification —— 01a 图像分类数据集

> 本目录服务于 01a「AlexNet 图像分类」，
> 与 `datasets/cat_face/`（目标检测）并列。

---

## 当前使用：CIFAR-10

| 项目 | 数值 |
|---|---|
| 图像 | 32×32 彩色，10 类 |
| 训练集 | 50000 张 |
| 测试集 | 10000 张 |
| 压缩包 | 约 162.6 MB（170498071 字节）|
| 官方 SHA256 | `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce` |
| 放置位置 | `datasets/classification/cifar10/` |

**类别（顺序即标签 0~9）**

| 标签 | 英文 | 中文 |
|---:|---|---|
| 0 | airplane | 飞机 |
| 1 | automobile | 汽车 |
| 2 | bird | 鸟 |
| 3 | **cat** | **猫** |
| 4 | deer | 鹿 |
| 5 | dog | 狗 |
| 6 | frog | 青蛙 |
| 7 | horse | 马 |
| 8 | ship | 船 |
| 9 | truck | 卡车 |

> 选用 CIFAR-10 的一个附带好处：它本身就有 `cat` 类，
> 与 01b（目标检测）的猫脸数据形成呼应。

---

## 目录结构（下载后）

```
datasets/classification/cifar10/
├─ cifar-10-python.tar.gz         # 下载的原始压缩包（保留，便于校验与重解压）
├─ cifar10.npz                    # 解析后的缓存，加载只要 1~2 秒（可删，会自动重建）
└─ cifar-10-batches-py/           # 解压出的官方批次
   ├─ data_batch_1 ~ data_batch_5 # 各 10000 张训练图（pickle，无扩展名）
   ├─ test_batch                  # 10000 张测试图
   └─ batches.meta                # 类别名
```

> `cifar-10-batches-py/` 下的文件**没有扩展名**（如 `data_batch_1`），
> 所以 `.gitignore` 里是对整个目录做忽略，不能靠 `*.bin` 之类的通配。

---

## 获取方式

**推荐：让脚本自动下载**（带断点续传、多镜像、SHA256 校验）

```powershell
$TF = ".\envs\pcd01_tf\Scripts\python.exe"
& $TF experiments\01_TensorFlow_AlexNet\code\data.py --download
```

**指定镜像**（名字里含该字符串即可，也可直接给 URL）：

```powershell
& $TF experiments\01_TensorFlow_AlexNet\code\data.py --download --mirror Baidu
& $TF experiments\01_TensorFlow_AlexNet\code\data.py --download --mirror "https://..."
```

**手动下载**：把 `cifar-10-python.tar.gz` 放到 `datasets/classification/cifar10/`，
再跑一次上面的 `--download` 即可（脚本会跳过下载，直接校验并解压）。

---

## 本机网络实测（重要）

这台机器到境外站点的速度很差，且会周期性停滞。实测吞吐：

| 源 | 实测速度 | 162.6 MB 预计耗时 |
|---|---:|---:|
| **PyPI**（仅作对照） | 1.378 MB/s | — |
| **GCS `storage.googleapis.com`**（仅作对照） | 0.317 MB/s | — |
| Baidu BOS（脚本默认首选）| 0.067 MB/s（突发）/ 约 0.017 MB/s（持续）| **1.5 ~ 2.5 小时** |
| 官方 `cs.toronto.edu` | 0.019 MB/s | 2 小时以上 |

> **只有 PyPI 和 GCS 快，但这两个源都不托管 CIFAR-10**——
> Keras 只把 MNIST / Fashion-MNIST / IMDB / Reuters 放在 GCS 上，
> CIFAR-10 至今只有 cs.toronto.edu 一个官方源（已逐一探测确认）。
> 因此 CIFAR-10 的下载注定慢，只能靠续传+重试扛过去。

**下载中断怎么办**：直接重跑同一条命令即可**从断点继续**，已下载的部分不会白费。

### 一个真实踩到的坑：续传拼出的包"大小对、内容是坏的"

第一次下载跑了 **75.6 分钟**，期间因网络停滞中断了 4 次，但续传机制把它们都扛过去了，
最终文件大小 **恰好等于官方的 170498071 字节**——看起来完全成功。

结果 SHA256 对不上，更要命的是**解压直接失败**：

```
期望 sha256: 6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce
实际 sha256: b9af559fb9eaf545d8f0998bd42cd1992eba157d57d5c7376f64213270f97e7b
解压报错   : Error -3 while decompressing data: invalid distance code
损坏位置   : 压缩流偏移 41943040 字节（正好 40.00 MB）
```

**原因**：`resp.read(n)` 在超时抛异常之前，内部可能**已经消费了一部分数据**。
这些字节既没返回给调用方、也拿不回来了，于是"已写入的字节数"与真实流位置之间
出现空洞。续传时按"已写入字节数"发 `Range` 请求，拼出来的文件长度虽然正确，
中间却缺了一段——压缩流一旦错位，后面全部失效。

**修法**（已实现在 `data.py`）：

| 措施 | 说明 |
|---|---|
| **重叠回退** | 每次续传都从 `have - 1 MB` 处重新下载，并先把文件截断到该位置。多下 1 MB 换来边界绝对对齐 |
| **内容级校验** | 判据从"比对 SHA256"改成"**能不能解开**"：`find_corruption_offset()` 用 `zlib` 增量解压，同时给出损坏位置 |
| **自动修复续传** | 发现坏包时，自动截断到 `损坏位置 - 1 MB` 并降级为 `.part` 继续下载，**不必从头重下** |
| **解压后逐批次校验** | 检查 6 个批次的形状、数量、标签范围，真正回答"能不能用来训练" |

> **为什么不只信 SHA256**：镜像站可能重新打包（gzip 头里含 mtime，重新压缩就会改哈希），
> 那样"哈希不一致"并不代表文件坏了；反过来"大小正确"也完全不保证内容正确。
> 唯一可靠的判据是**实际解压一遍**。

本次修复后的实际运行记录：

```
已存在压缩包，正在校验能否解压
  [损坏] 压缩流在 41943040 字节（40.00 MB）处损坏
         回退到 40894464 字节（39.00 MB）后继续下载剩余部分
    [续传] 回退 1.0 MB 后从 39845888 字节继续（保证边界对齐）
     24.0%    39.0 / 162.6 MB
```

只重下了损坏之后的部分，已下好的 39 MB 全部保留。

---

**为什么不用 Keras 自带的 `cifar10.load_data()`**：它把数据下到用户主目录
（`~/.keras/datasets`），不符合本仓库"数据统一放 `datasets/`"的约定，
而且无法指定镜像、没有断点续传，在这种网络下几乎必然失败。

---

## 划分方式

从 50000 张训练集中按固定种子（`seed=42`）划出 **5000 张作验证集**：

| 划分 | 数量 | 用途 |
|---|---:|---|
| 训练 | 45000 | 训练 |
| 验证 | 5000 | 早停/选 checkpoint |
| 测试 | 10000 | **只在最终评测用一次**，训练与选型全程不参与 |

归一化的均值/标准差**只从训练集统计**，避免测试集信息泄漏。

---

## 校验

```powershell
& $TF experiments\01_TensorFlow_AlexNet\code\data.py            # 报告数据状态
& $TF experiments\01_TensorFlow_AlexNet\code\data.py --selftest  # 数据管线自检（不需要下载）
python tools\check_datasets.py                                    # 全仓库数据集检查
```
