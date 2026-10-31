# 01b · 目标检测：猫脸检测（YOLOv3）

> 规格文档：原始 .docx / .pptx 因版权仅本地保留，不入库，说明见 [docs/README.md](../../docs/README.md)
> 技术报告：[`reports/01-object-detection.md`](../../reports/01-object-detection.md)

---

## 1. 任务定义

**任务**：使用猫咪数据集和卷积神经网络模型完成 **猫脸检测**。

**目的**
1. 了解数据收集以及数据集制作等系列过程；
2. 使用卷积神经网络模型对数据进行训练、测试；
3. 修改目标检测模型代码，**从头训练**自己的模型。

**基础代码**：`keras-yolo3` —— <https://github.com/qqwweee/keras-yolo3>

---

## 2. 目录结构

```
experiments/01_ObjectDetection_YOLOv3/
├─ code/            # 本模块代码 + keras-yolo3 源码
│  ├─ config.py         #   路径与类别配置（所有脚本共用，不硬编码绝对路径）
│  ├─ make_imagesets.py #   生成 ImageSets/Main 索引（替代上游 test.py）
│  ├─ voc_annotation.py #   生成 train.txt / val.txt / test.txt 与 voc_classes.txt
│  ├─ test_pipeline.py  #   数据管线自检（合成数据跑通全流程，35 项断言）
│  ├─ yolo3/            #   keras-yolo3 的网络定义（需从上游仓库下载）
│  ├─ model_data/       #   yolo_anchors.txt、voc_classes.txt、yolov3.cfg、预训练权重
│  ├─ train.py          #   训练脚本（沿用上游版本）
│  ├─ yolo.py           #   推理封装
│  ├─ yolo_video.py     #   入口：--image 批量检测图片 / --input 检测视频
│  └─ train.txt / val.txt / test.txt   # voc_annotation.py 生成
├─ models/          # 训练得到的权重（.h5）
├─ outputs/         # 检测结果图/视频（对应上游 outdir 约定）
├─ logs/            # TensorBoard events 与 checkpoint
├─ README.md
└─ requirements.txt
```

### 已实现的脚本

| 脚本 | 作用 | 需要 TF? |
|---|---|:---:|
| `config.py` | 统一路径与类别配置（`CLASSES=["cat"]`、`filters=18`） | 否 |
| `make_imagesets.py` | 由 `Annotations/*.xml` 生成 trainval/train/val/test 索引 | 否 |
| `voc_annotation.py` | XML → YOLOv3 训练标注（含坐标校验与统计） | 否 |
| `auto_annotate.py` | COCO 预训练 Faster R-CNN 自动预标注（需 PyTorch） | 否 |
| `patch_yolov3_cfg.py` | 自动改 cfg 的 3 处 `classes` 与 3 处 `filters` | 否 |
| `fetch_base_code.py` | 下载 keras-yolo3 基础代码并记录 SHA256 | 否 |
| `evaluate_detection.py` | 算 Precision / Recall / F1 / AP / mAP | 否 |
| `test_pipeline.py` | 造合成数据集跑通数据管线，35 项断言 | 否 |
| `train.py` | 从头训练，含 `--benchmark` 实测每 step 耗时 | **是** |
| `detect_images.py` | 批量检测，出结果图 + 坐标 CSV | **是** |

**除了最后两个，其余脚本都不需要 TensorFlow**，可以先用来把数据和流程都验证好。

### 完整流程（按顺序执行）

```bash
cd experiments/01_ObjectDetection_YOLOv3/code

# ── 0. 取回 keras-yolo3 基础代码（yolo3/、yolov3.cfg、anchors、font/ 字体）
python fetch_base_code.py

# ── 1. 自动预标注：用 COCO 预训练 Faster R-CNN 先把猫框出来
#      （需 PyTorch 环境；产出 VOC XML，database 标记为 auto-fasterrcnn）
python auto_annotate.py --score 0.5

# ── 2. 人工收紧：用 LabelImg 把"整只猫"的框收到"猫脸"
labelImg
#      Open Dir        -> ../../datasets/cat_face/JPEGImages
#      Change Save Dir -> ../../datasets/cat_face/Annotations
#      逐张收紧框（标签保持 cat），Ctrl+S 保存
#      查看还剩多少没处理：
python auto_annotate.py --report

# ── 3. 生成索引与训练标注
python make_imagesets.py
python voc_annotation.py

# ── 4. 修改 cfg 的类别数与 filters（3 处 classes + 3 处 filters）
python patch_yolov3_cfg.py --in-place --random 0 --check   # 先检查
python patch_yolov3_cfg.py --in-place --random 0           # 再修改

# ── 5. 先用数据管线自检和评测自检确认逻辑没问题
python test_pipeline.py            # 35 项断言
python evaluate_detection.py --selftest   # 25 项断言

# ── 6. 用 TF 环境测速，据此决定 epoch 数（本机只能 CPU）
& '..\..\..\envs\pcd01_tf\Scripts\python.exe' train.py --benchmark 10

# ── 7. 训练
& '..\..\..\envs\pcd01_tf\Scripts\python.exe' train.py --epochs 200

# ── 8. 批量检测（默认对 8 张演示图）
& '..\..\..\envs\pcd01_tf\Scripts\python.exe' detect_images.py --weights ../models/<权重>.h5

# ── 9. 算指标（需要检测的图有 VOC 标注）
python evaluate_detection.py --pred ../outputs/detect/<权重名>/detections.csv
```

---

## 3. 运行环境

上游给出三种配置，**训练图片 230 张**：

| 配置 | 硬件 | Python | Keras / TF | 速度（230 张） |
|:---:|---|---|---|---|
| 一 | win10 + GTX960M | 3.7 | keras 2.4.3 / tensorflow 2.3.0 | 900+ s |
| 二 | win10 + GTX1080Ti | 3.7 | keras 2.2 / tensorflow 1.5.0 | 480 s |
| 三 | win10 + GTX1080Ti + CUDA10.0 + cuDNN7.4 | 3.7 | keras 2.2 / tensorflow-GPU 1.5.0 | 120 s |

上游结论：**200 epoch 时检测效果最好**。

### 本机环境（只能 CPU 训练）

本机 GPU 是 **RTX 4060 Laptop（compute capability = sm_89，Ada 架构）**，它**用不了 TensorFlow**：

| TF 版本 | 绑定 CUDA | 支持的 GPU 上限 | 能跑 sm_89？ |
|---|---|---|---|
| 2.3.0（上游环境一） | CUDA 10.1 | sm_75 (Turing) | ✗ |
| 2.10.0（Windows 上最后一个有 GPU 支持的版本） | CUDA 11.2 | sm_86 (Ampere) | ✗ |

sm_89 需要 **CUDA 11.8 及以上**，而 TensorFlow 自 **2.11 起取消了 Windows 原生 GPU 支持**。
所以本机用 TensorFlow 训练**一定是 CPU**，选哪个版本都一样。

### 实际采用的环境：Python 3.10 + TensorFlow 2.10.0

上游环境一是 `python 3.7/3.8 + tensorflow 2.3.0 + keras 2.4.3`，但在本机上装不起来：

> venv 里 Python 3.8 的 OpenSSL 1.1.1 访问 pypi.org 与国内镜像时**间歇性 TLS 握手失败**，
> 报 `SSLError(142, '[ASN1: NOT_ENOUGH_DATA] not enough data')`（证书数据被截断）。
> 同一批 URL 用 Python 3.10 / 3.12 访问完全正常，说明不是网络问题，是 3.8 的 SSL 栈；
> 换明文 HTTP 源也不行——镜像会把 http 重定向到 443，照样走 TLS。

改用 **Python 3.10 + TensorFlow 2.10.0**。选 2.10 而不是更新的版本，是因为它
**仍属于 Keras 2**（`keras.backend`、`fit_generator`、`Lambda(arguments=...)` 都还在），
keras-yolo3 改动最小。实测只需补一个兼容层 `code/keras_compat.py`（见下）。

环境建议直接用脚本建好（用 pytorch 环境的 Python 3.10 建独立 venv，不动你现有环境）：

```bash
python envs/setup_tf_env.py
```

### keras-yolo3 的兼容层

keras-yolo3 写于 Keras 2.2~2.4 时代，在 Keras 2.10 上有 4 处 API 变化。
为**不改动上游源码**，本仓库用 `code/keras_compat.py` 在导入前把缺失的名字补回去，
由 `config.py` 自动加载：

| 报错 | 兼容层处理 |
|---|---|
| `No module named 'keras.layers.advanced_activations'` | 补回子模块，注入 `LeakyReLU` 等 |
| `cannot import name 'BatchNormalization' from 'keras.layers.normalization'` | 把 `keras.layers` 下的同名层注入子模块 |
| `module 'keras.backend' has no attribute 'control_flow_ops'` | 指向 `tensorflow` |
| `cannot import name 'multi_gpu_model' from 'keras.utils'` | 放占位实现（只在 `gpu_num>=2` 时调用） |

### 实测训练速度

```bash
& 'envs\pcd01_tf\Scripts\python.exe' code/train.py --benchmark 5
```

| backbone | input_shape | batch_size | 每 step | steps/epoch | 200 epoch 预计 |
|---|---|---:|---:|---:|---:|
| full（Darknet-53） | 416×416 | 10 | **8.83 s** | 14 | **≈ 6.9 小时** |

依赖见 [`requirements.txt`](requirements.txt)。**手动安装时务必加 `--only-binary=:all:`**，
否则 pip 的解析器会在大量源码包候选上反复回溯、长时间无响应（这是实际踩过的坑）。

---

## 4. 数据准备

数据集放在 [`datasets/cat_face/`](../../datasets/cat_face/)，采用 **Pascal VOC2007** 格式。详细说明见 [datasets/README.md](../../datasets/README.md)。

```
datasets/cat_face/
├─ JPEGImages/         # 205 张猫咪图片（纯英文/数字命名）
├─ Annotations/        # LabelImg 标注的 .xml（当前只有 1 个）
├─ ImageSets/Main/     # trainval/train/val/test 索引（脚本生成）
├─ test/               # 检测演示图 8 张
├─ raw/                # 网盘下载的原始文件暂存
└─ predefined_classes.txt   # LabelImg 预设类别：cat
```

自检：`python tools/check_datasets.py --only cat_face`

### 当前数据状况（重要）

| 项目 | 数量 |
|---|:---:|
| `JPEGImages/` 图片 | 205 |
| `Annotations/` 标注 | **1** |
| 标注覆盖率 | **0.5%** |

**训练前必须把剩下的 204 张补上标注。** YOLOv3 从头训练时未标注的图片会被当成纯背景，
只有 1 个框是训不出检测器的；上游环境用了 230 张全部标注的图片。
补标注的详细步骤见 [datasets/cat_face/README.md](../../datasets/cat_face/README.md)。

### 4.1 采集数据
猫咪图片已经从网盘取回并整理进 `datasets/cat_face/JPEGImages/`（数据集，本地保留，不入库）（205 张）。
如要扩充，可自行在网上收集，或使用 CIFAR-10 中的 cat 图片。

> **命名要求**：图片与文件夹命名**避免使用汉字**，否则后续读写会出错。

### 4.2 用 LabelImg 打标
下载 LabelImg，操作：
1. `Open Dir` → 打开 `JPEGImages/`
2. `Change Save Dir` → 选择 `Annotations/`
3. `Create RectBox` → 框出猫脸，标签统一填 **`cat`**
4. `Next Image` 继续下一张

> **一致性问题**：不同图片中同一类别的标签名必须完全相同（都用 `cat`）。图片过多时可分组协同标注。

### 4.3 生成索引文件（已实现）

```bash
cd experiments/01_ObjectDetection_YOLOv3/code
python make_imagesets.py
```

生成 `<voc_root>/ImageSets/Main/` 下的 `trainval.txt` / `train.txt` / `val.txt` / `test.txt`。

> **与上游 `test.py` 的差别**：上游脚本里 `trainval_percent = 0.1`、`train_percent = 0.9` **写反了**，
> 实际得到 train 90%、test 9%、val 1%，语义混乱。
> `make_imagesets.py` 按 VOC 惯例修正为 **train 81% / val 9% / test 10%**，
> 只收录有标注的图片，固定随机种子，并会报告缺标注的图片数量。
> 可用 `--trainval-percent` / `--train-percent` / `--seed` 调整。

### 4.4 生成训练文件（已实现）

```bash
python voc_annotation.py
```

生成 `code/train.txt` / `val.txt` / `test.txt`（每行一张图 + 若干 `xmin,ymin,xmax,ymax,class_id`）
以及 `code/model_data/voc_classes.txt`。

> **不需要再手动去掉 `2007_` 前缀** —— 脚本直接生成目标文件名。
> 图片路径默认写成**相对于 `code/` 目录**的形式，所以 `train.py` 必须在 `code/` 下运行。
> 也可以用 `--absolute-paths` 改成绝对路径。

脚本会做这些校验并报告：坐标越界（自动裁剪）、坐标反转（自动交换）、
退化框（宽高为 0）、类别名不在 `CLASSES` 里、`difficult=1`（按 keras-yolo3 的约定跳过）。

### 4.5 修改参数文件

**`model_data/yolov3.cfg`**：`Ctrl+F` 搜索 `yolo`，共 **3 处**，每处都要改：
- `filters = 3 * (5 + len(classes))` → 本模块 `len(classes)=1`，故 `filters = 18`
- `classes = 1`
- `random`：显存小就改成 `0`

**`model_data/voc_classes.txt` 与 `coco_classes.txt`**：内容改为一行 `cat`
（`voc_classes.txt` 已由 `voc_annotation.py` 自动生成）。

> 该 cfg 用于转换官网权重；若从头训练自己的网络则不需要处理（本模块即从头训练）。

### 4.6 数据管线自检

```bash
python test_pipeline.py
```

会在临时目录里造一份合成 VOC 数据集（含缺标注、孤立标注、坐标反转、越界、退化框、
未定义类别、`difficult=1` 等脏数据），跑通上面两个脚本并做 **35 项断言**。
真实数据目前只有 1 张有标注，覆盖不到这些分支，所以用自检来保证管线正确。

---

## 5. 训练

用上游代码**替换** `train.py` 内容，然后：

```bash
cd experiments/01_ObjectDetection_YOLOv3/code
python train.py
```

关键代码位置：
- `annotation_path = 'train.txt'`
- `log_dir = 'logs/000/'`
- `classes_path = 'model_data/voc_classes.txt'`
- `anchors_path = 'model_data/yolo_anchors.txt'`
- `input_shape = (416, 416)`
- `batch_size = 10`，`val_split = 0.1`，`epochs = 500`（上游建议 200 epoch）
- `create_model(..., load_pretrained=False)` → **从头训练**

权重与 checkpoint 保存在 `logs/000/`，最终权重 `trained_weights.h5`。

> 训练产物建议归档到 `models/`：
> `copy logs\000\trained_weights.h5 models\`

---

## 6. 推理与可视化

1. 修改 `yolo.py` 中加载的 `.h5` 权重路径，指向训练好的权重；
2. 用上游给定的 `detect_img(yolo)` 版本替换 `yolo_video.py` 中的同名函数，把 `path` 换成待检测图片目录、`outdir` 换成结果输出目录；
3. 运行（**必须带 `--image` 参数**）：

```bash
python yolo_video.py --image
```

检测结果图统一输出到 `outputs/`。

---

## 7. 训练过程可视化（TensorBoard）

附加要求是跟踪训练全过程的信息：每层参数如何变化与分布、每轮参数更新后模型在测试集/训练集上的准确率、损失值变化等。

**TensorBoard 可记录**：Scalars、Images、Audio、Graph、Distribution、Histograms、Embeddings。

**流程**
1. 建图，确定在哪些节点放置 summary operations；
2. `tf.summary.scalar` / `histogram` / `distribution` / `image` 记录信息；
3. `tf.summary.merge_all` 合并所有 summary 节点；
4. `tf.summary.FileWriter` 把运行结果写到本地磁盘；
5. 运行程序，命令行启动 tensorboard 查看。

本模块的猫脸识别代码（相较于 2.1 节）已增添 TensorBoard 相关内容。

**启动 TensorBoard**

```bash
tensorboard --logdir experiments/01_ObjectDetection_YOLOv3/logs/000
# 打不开时可追加指定 host
tensorboard --logdir <logdir> --host=127.0.0.1
```

**注意事项**
- 电脑名称必须是**英文**，含中文会导致日志文件后缀乱码；
- 浏览器建议用 **Chrome**，否则生成的网址可能打不开；
- 命令要在**正确的文件目录**下输入，且 `--logdir` 路径要与日志存放路径对应。

---

## 8. 结果

结果图保存到 [`outputs/`](outputs/)，训练日志与 events 保存到 [`logs/`](logs/)。
报告中的图表分别放到 [`reports/figures/`](../../reports/figures/) 与 [`reports/tables/`](../../reports/tables/)。

建议记录的内容：

| 项目 | 说明 |
|---|---|
| 训练集规模 | 图片数、标注框数、train/val 划分 |
| 训练配置 | epoch 数、batch size、input shape、优化器（Adam） |
| 损失曲线 | 训练/验证损失随 epoch 的变化（TensorBoard 截图） |
| 检测效果 | 测试图片的检测结果图，最好含成功与失败样例 |
| 最佳 epoch | 200 epoch（与上游结论一致） |

---

## 9. 故障排查

| 现象 | 原因与解决 |
|---|---|
| 维度不匹配 / 检测框全错 | `classes` 与 `filters` 三处未同步改。`filters = 3*(5+1) = 18`，`classes = 1` |
| 读不到图片 / 乱码 | 图片或文件夹名含中文。改成纯英文/数字 |
| 类别数不对 | `voc_classes.txt` 与 `coco_classes.txt` 未改成只有 `cat` 一行 |
| 训练文件路径找不到 | `voc_annotation.py` 生成的 `2007_train.txt` 未去掉 `2007_` 前缀 |
| 显存不足（OOM） | 把 cfg 中的 `random` 改为 `0`，或减小 `batch_size` |
| TensorBoard 网址打不开 | 计算机名含中文、非 Chrome 浏览器、`--logdir` 路径写错 |
| 训练很慢 | 检查是否用上 GPU（配置三比配置一快约 7 倍）；`os.environ["CUDA_VISIBLE_DEVICES"] = "0"` |
| 权重加载失败 | `yolo.py` 中的 `.h5` 路径未改成自己训练出的权重 |

---

## 10. 参考资料

- keras-yolo3：<https://github.com/qqwweee/keras-yolo3>
- LabelImg：<https://github.com/HumanSignal/labelImg>
- 原始规格文档：`docs/extracted_text/`（版权原因，仅本地保留）
