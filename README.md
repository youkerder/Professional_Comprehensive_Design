# 视觉任务工程集：检测 / 跟踪 / 摘要 / 检索

五个相互独立的计算机视觉模块，共用一套数据集约定、虚拟环境与评测规范。
每个模块都是可复现的最小工程：数据管线 → 训练 → 推理 → 指标评测 → 技术报告，
代码与产物分离，数据集与权重不入库。

---

## 1. 模块清单

| 编号 | 任务 | 方法 | 数据集 | 关键指标 |
|:---:|---|---|---|---|
| 01a | [图像分类](experiments/01_TensorFlow_AlexNet/README.md) | TensorFlow / Keras 从零实现基础 AlexNet；ReLU vs LeakyReLU × 学习率 0.01 / 0.001 四组对照 | CIFAR-10 | acc 0.8185，macro-F1 0.8162 |
| 01b | [目标检测](experiments/01_ObjectDetection_YOLOv3/README.md) | keras-yolo3 / YOLOv3 从头训练，自建 VOC2007 格式猫脸数据集，聚类 anchors | 自建猫脸数据集 | AP 0.5308，F1 0.4651 |
| 02 | [多目标跟踪](experiments/02_ObjectTracking_OpenCV/README.md) | OpenCV 内置 6 种跟踪器基准，OTB / MOTChallenge / VOT 三套指标 | 自建合成多目标序列 | Success AUC 0.843 |
| 03 | [视频摘要](experiments/03_VideoSummarization_FCSN/README.md) | FCSN 全卷积序列网络（有监督），BCE + 类别加权 | TVSum | F-score 0.2552 |
| 04 | [视频检索](experiments/04_VideoRetrieval/README.md) | ResNet50 深度特征 / HSV 直方图，FAISS Flat·IVF·HNSW 索引 | 程序化合成检索库 | mAP 0.9861 |

任务规格摘要（原始规格文档为 `.docx` / `.pptx`，因版权仅本地保留，见 [docs/](docs/README.md)）：

- **01a**：搭建基础 AlexNet 完成图像分类，实现「输入一张图像 → 输出类别」；用两种激活函数（ReLU 与任一对比项）和两种学习率做对照。
- **01b**：用卷积神经网络完成猫脸检测。以 `keras-yolo3` 为基础，制作 VOC2007 格式数据集（LabelImg 标注 → XML → `ImageSets/Main` → `train.txt`），修改 `yolov3.cfg` 与类别配置，从头训练后用 `yolo_video.py` 批量推理；附加训练过程可视化（TensorBoard）。
- **02**：基于 OpenCV 内置算法对多目标做跟踪（≥3 种算法），比较跟踪效率；输入视频框出多目标运动轨迹；用 ≥2 种指标评价精度、准确率与鲁棒性。
- **03**：基于 FCSN 与 TVSum 训练视频摘要模型，取 10 段视频测试，为每段视频生成摘要视频 / 关键帧并输出到独立目录，计算 Precision / Recall / F-score。
- **04**：基于深度特征实现以图搜图、以图搜视频，比较检索效率，并用 ≥2 种指标评价性能与准确率。

---

## 2. 目录结构

```
Professional_Comprehensive_Design/
├─ README.md                     # 本文件：模块清单、结构、环境、运行、结果
├─ docs/                         # 原始规格文档与纯文本抽取（版权原因：仅本地保留，不入库）
│  ├─ course_materials/          #   原始 .docx / .pptx（只读参考）
│  └─ extracted_text/            #   纯文本抽取结果，便于检索
├─ datasets/                     # 共享数据集，按模块划分
│  ├─ classification/            #   01a 图像分类：CIFAR-10
│  ├─ cat_face/                  #   01b 目标检测：猫脸数据集（VOC2007 格式）
│  ├─ tracking_videos/           #   02 多目标跟踪：视频与标注
│  ├─ TVSum/                     #   03 视频摘要：TVSum
│  └─ retrieval/                 #   04 视频检索：图像库 / 视频库 / 查询集
├─ experiments/
│  ├─ 01_TensorFlow_AlexNet/     #   01a code/ models/ logs/ outputs/ README.md requirements.txt
│  ├─ 01_ObjectDetection_YOLOv3/ #   01b code/ models/ outputs/ logs/ README.md requirements.txt
│  ├─ 02_ObjectTracking_OpenCV/  #   02  code/ outputs/ README.md requirements.txt
│  ├─ 03_VideoSummarization_FCSN/#   03  code/ models/ outputs/ logs/ README.md requirements.txt
│  └─ 04_VideoRetrieval/         #   04  code/ models/ outputs/ logs/ README.md requirements.txt
├─ reports/                      # 技术报告与图表
│  ├─ 01-object-detection.md     #   各模块技术报告（01a / 01b / 02 / 03 / 04）
│  ├─ figures/                   #   报告插图（曲线、对比图、样例检测结果）
│  └─ tables/                    #   报告表格（指标 CSV / Markdown）
├─ envs/                         # 环境构建脚本与版本记录（环境本体不入库）
└─ tools/                        # 公共工具（文档抽取、数据集校验、labelImg 修补等）
```

各模块目录约定：

| 子目录 | 用途 |
|---|---|
| `code/` | 该模块全部源代码 |
| `models/` | 预训练权重与训练产物（`.h5` / `.pth` / `.pt`） |
| `outputs/` | 检测结果图、跟踪视频、摘要视频、检索结果等运行产物 |
| `logs/` | 训练日志、TensorBoard events、评测原始输出 |
| `README.md` | 模块说明：任务、环境、数据、运行、结果、故障排查 |
| `requirements.txt` | 该模块的 Python 依赖 |

> 01a 与 01b 的依赖完全一致（TensorFlow 2.10 + numpy + matplotlib + Pillow），
> 共用同一虚拟环境 `envs/pcd01_tf`，未重复构建。

### 版本控制范围

**入库**：各模块源代码、`README.md`、`requirements.txt`、`reports/` 下的脱敏 Markdown 技术报告与插图、
数据集说明文档与校验脚本、环境构建脚本。

**不入库**（见 `.gitignore`）：数据集本体、模型权重（`.h5` / `.pth` / `.pkl`）、
`outputs/` 与 `logs/` 下的运行产物、`envs/` 下的虚拟环境本体、原始规格文档（`.docx` / `.pptx` / `.pdf`，版权原因）
以及报告的 Word / PDF 版本。数据集与权重按各模块 README 的步骤重新下载或训练即可。

---

## 3. 环境

各模块依赖栈差异较大，按模块分组使用独立环境。

```bash
# 01a / 01b 共用（Python 3.10 + TensorFlow 2.10 + Keras 2.10，CPU）
python envs/setup_tf_env.py

# 其余模块按各自 requirements.txt 安装
conda create -n pcd02 python=3.9 -y
pip install -r experiments/02_ObjectTracking_OpenCV/requirements.txt
```

> **版本说明**：上游参考实现基于 `keras 2.2~2.4 + tensorflow 1.5~2.3`；
> 本仓库统一到 `TF 2.10 + Keras 2.10`。02 依赖 `opencv-contrib-python`（跟踪器位于 contrib 模块），
> 03 / 04 使用 PyTorch。
>
> **01a / 01b 为何只用 CPU**：本机 RTX 4060 Laptop 为 sm_89（Ada），需 CUDA 11.8+；
> 而 TF 2.10 绑定 CUDA 11.2（最高支持 sm_86），TF ≥ 2.11 又取消了 Windows 原生 GPU 支持。

---

## 4. 数据

数据集体积较大，不纳入版本控制。按下表放置：

| 模块 | 放置位置 | 获取方式 |
|:---:|---|---|
| 01a | `datasets/classification/cifar10/` | 脚本自动下载：`python experiments/01_TensorFlow_AlexNet/code/data.py --download` |
| 01b | `datasets/cat_face/` | 图像放 `JPEGImages/`，LabelImg 标注结果 `.xml` 放 `Annotations/`（已完成：205 图 / 205 标注）|
| 02 | `datasets/tracking_videos/videos/` | 待跟踪视频（`.mp4` / `.avi`） |
| 03 | `datasets/TVSum/` | TVSum 视频与 `ydata-tvsum50-anno.tsv` 标注 |
| 04 | `datasets/retrieval/images/`、`videos/`、`queries/` | 检索库与查询图 |

命名规范、目录校验的细节见 [datasets/README.md](datasets/README.md)，或运行：

```bash
python tools/check_datasets.py
```

> **CIFAR-10 下载慢属预期**：到境外站点实测 0.02~0.07 MB/s，官方源每 8~9 秒停滞，全量约 1.5~2.5 小时。
> `data.py` 已实现多镜像、HTTP Range 断点续传、停滞重试与 SHA256 校验，中断后重跑同一命令即可续传。
> 详见 [datasets/classification/README.md](datasets/classification/README.md)。

---

## 5. 运行

```bash
# 01b 目标检测
python experiments/01_ObjectDetection_YOLOv3/run_all.py --dry-run
python experiments/01_ObjectDetection_YOLOv3/run_all.py --epochs 200

# 01a AlexNet 图像分类（$TF 指向 01a/01b 共用的解释器，见模块 README）
& $TF experiments/01_TensorFlow_AlexNet/run_all.py --dry-run --tf-python $TF
& $TF experiments/01_TensorFlow_AlexNet/run_all.py --epochs 15 --tf-python $TF
```

02 / 03 / 04 各有 `run_step*.cmd` 或 `env.cmd` 驱动的分步流程，入口与参数见各模块 README。

---

## 6. 结果

| 模块 | 环境 | 数据 | 代码 | 训练 / 推理 | 评测 | 报告 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01a 图像分类 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 01b 目标检测 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 02 多目标跟踪 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 03 视频摘要 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 04 视频检索 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |

### 6.1 图像分类（01a）

实现要点：

- `data.py`：CIFAR-10 下载 / 解析 / 缓存 / 划分 / 归一化 / 增强，下载部分不依赖 TF
- `alexnet.py`：基础 AlexNet（5 卷积 + 3 全连接，4.62 M 参数），激活函数与规范化方式可切换
- `train.py`：单组训练，含 `--benchmark` 测速、NaN 守卫、断点续训
- `evaluate.py`：accuracy / top-3 / 逐类 P-R-F1 / 混淆矩阵，带 `--selftest`
- `predict.py`：输入一张图像输出类别，带 `--selftest`
- `compare.py`：四组汇总与对比图；`run_all.py`：一键流程（含 `--dry-run`）
- 自检 160 项断言全过（`test_pipeline.py` 74 + 四个 `--selftest` 57 + `test_cli.py` 29）

四组正式训练各 15 epoch，总用时 3.25 小时：

| 激活函数 | 学习率 | 测试准确率 | top-3 | macro-F1 |
|---|---:|---:|---:|---:|
| **ReLU** | **0.01** | **0.8185** | **0.9633** | **0.8162** |
| ReLU | 0.001 | 0.7768 | 0.9519 | 0.7771 |
| LeakyReLU | 0.01 | 0.8055 | 0.9582 | 0.7987 |
| LeakyReLU | 0.001 | 0.7706 | 0.9484 | 0.7699 |

分组均值：ReLU 0.7976 vs LeakyReLU 0.7880（差 0.96 个百分点，不显著）；
lr=0.01 0.8120 vs lr=0.001 0.7737（差 3.83 个百分点，显著）。

三条结论：① 激活函数无显著差异（推测 BN 已缓解 ReLU 神经元死亡）；
② 学习率影响约为激活函数的 4 倍，小学习率的劣势来自训练不足而非收敛质量（损失仍在下降）；
③ `cat` 类最难（Recall 0.5310，18% 误判为 `dog`）。
跨数据集验证：4 张未参与训练的真实猫照片 3 张判为 `cat`。

技术报告：[reports/01b-alexnet-classification.md](reports/01b-alexnet-classification.md)

### 6.2 目标检测（01b）

- 数据集：205 张图 + 8 张演示图；COCO 预训练 Faster R-CNN 生成 200 个 VOC XML 后人工收紧，共 236 个目标框
- 环境：Python 3.10 + TF 2.10.0（CPU），处理 4 处 Keras API 兼容问题
- `yolov3.cfg` 修正 3 处 `classes=1` / `filters=18`
- 训练 200 epoch（中途从 ep167 续训至 200）
- 指标：AP 0.5308，Recall 最高 0.88（阈值 0.003）；工作点阈值 0.03 时 P 0.5556 / R 0.4000 / F1 0.4651
- 关键改进：默认 COCO anchors 使全部目标落入最粗尺度、细尺度零正样本；
  改用本数据集聚类 anchors 后，同样 20 epoch 下 AP 由 0.0011 提升到 0.0752（68 倍）

技术报告：[reports/01-object-detection.md](reports/01-object-detection.md)

### 6.3 多目标跟踪（02）

OpenCV 内置 6 种跟踪器（KCF / CSRT / MedianFlow + MIL / Boosting / TLD）同时跟踪 3 个目标，
自建 `cross`、`occlude` 两段合成序列（640×360、25 fps、300 帧）并生成逐帧真值，
按 OTB / MOTChallenge / VOT 三套指标评测精度、成功率与鲁棒性。

- `cross` 序列 Success AUC：BOOSTING 0.843 > MedianFlow 0.794 > CSRT 0.772 > KCF 0.679
- 速度：MedianFlow ≈ 197 FPS、KCF ≈ 92 FPS、CSRT ≈ 12 FPS、MIL 3.4 FPS
- MOSSE 在真实视频上 `update` 全部失败（0/59）：合成 patch 自检通过不等于真实序列可用
- 结论：相关滤波（KCF / CSRT）在精度-速度上最均衡；纯色块目标使 MedianFlow 无法初始化

技术报告：[reports/02-multi-object-tracking.md](reports/02-multi-object-tracking.md)

### 6.4 视频摘要（03）

FCSN + TVSum（40 段训练 / 10 段测试），补齐上游缺失的 pool5 特征库与 KTS 镜头切分，
生成摘要视频与关键帧并计算 P / R / F-score。

- 10 段测试视频平均：Precision 0.3724、Recall 0.1961、F-score 0.2552（epoch-109），末轮 0.2230
- 同口径参照：线性岭回归基线 0.2561、用 gtscore 选段（Oracle）0.2858、20 位标注者互评 0.366
  —— 模型达到理想选段的 89%、人类一致度的 70%
- 关键修复：上游 NLL + 反比加权配方会使模型退化为常数输出（AUC 0.49）；
  改为 BCE + `pos_weight = n_neg / n_pos` 后 AUC 升至 0.62

技术报告：[reports/03-video-summarization.md](reports/03-video-summarization.md)

### 6.5 视频检索（04）

ResNet50 深度特征与 HSV 颜色直方图两种表示，配暴力检索 + FAISS Flat / IVF / HNSW 四种索引，
实现以图搜图、以图搜视频，用 P@K 与 mAP 评价，并在三种库规模上测延迟。

- 以图搜图（120 库图 / 24 查询）：深度特征 mAP 0.9861、P@1 = 1.0000；直方图 mAP 0.9623
- 以图搜视频（12 视频 / 24 查询）：深度特征 mAP 1.0000（mean / max / frame-max 三种聚合同分）
- 检索库由 `make_dataset.py` 程序化合成，相关性真值由目录名推导，避免人工标注偏差

技术报告：[reports/04-video-retrieval.md](reports/04-video-retrieval.md)

---

## 7. 参考资料

- AlexNet：<https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>
- CIFAR-10：<https://www.cs.toronto.edu/~kriz/cifar.html>
- keras-yolo3（01b 基础实现）：<https://github.com/qqwweee/keras-yolo3>
- LabelImg（01b 标注工具）：<https://github.com/HumanSignal/labelImg>
- TVSum：<https://github.com/yalesong/tvsum>
- FCSN 视频摘要：<https://github.com/remega/video_summarization_fcsn>
- OpenCV 跟踪 API：<https://docs.opencv.org/4.x/d9/df8/group__tracking.html>
