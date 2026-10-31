# 专业综合设计（Professional Comprehensive Design）

本仓库是「专业综合设计」课程的完整工作区，包含 **4 个实验** 的代码、数据、模型、输出与实验报告。

其中「实验一」按题目要求分成**两个部分**：TensorFlow 编程实验（AlexNet 图像分类）
与目标检测（YOLOv3 猫脸检测）。

---

## 一、实验清单

| 编号 | 实验名称 | 题目来源 | 核心方法 | 数据集 |
|:---:|---|---|---|---|
| 01a | [TensorFlow 编程：AlexNet 图像分类](experiments/01_TensorFlow_AlexNet/README.md) | 《实验题目-4.docx》**实验一** | TensorFlow/Keras 从零搭建基础 AlexNet；对比 ReLU vs LeakyReLU、两种学习率 | [datasets/classification](datasets/classification/)（CIFAR-10）|
| 01b | [目标检测：猫脸检测](experiments/01_ObjectDetection_YOLOv3/README.md) | 《目标检测实验指导书》 | keras-yolo3 / YOLOv3 从头训练 | [datasets/cat_face](datasets/cat_face/) |
| 02 | [目标跟踪：多目标跟踪](experiments/02_ObjectTracking_OpenCV/README.md) | 《实验题目-4.docx》实验三 +《目标跟踪实验指导书》 | OpenCV 内置多种跟踪算法对比 | [datasets/tracking_videos](datasets/tracking_videos/) |
| 03 | [视频摘要：FCSN + TVSum](experiments/03_VideoSummarization_FCSN/README.md) | 《实验题目-4.docx》实验六 | FCSN（全卷积序列网络）有监督视频摘要 | [datasets/TVSum](datasets/TVSum/) |
| 04 | [视频检索：以图搜图 / 以图搜视频](experiments/04_VideoRetrieval/README.md) | 《实验题目-4.docx》综合实验1 | 深度学习特征提取 + 相似度检索 | [datasets/retrieval](datasets/retrieval/) |

### 各实验的具体要求（摘自题目与指导书）

**01a TensorFlow 编程实验**
基于 TensorFlow 搭建一个基础的 **AlexNet** 网络模型对图像分类。编写程序实现
**输入一张图像、输出图像的类别**；使用**两种不同的激活函数**进行比较
（**ReLU 以及任意另一种**）；使用**两种不同的学习率**进行比较。

**01b 目标检测**
使用猫咪数据集和卷积神经网络模型完成 **猫脸检测** 任务。以 `keras-yolo3` 为基础：制作 VOC2007 格式数据集（LabelImg 标注 → `.xml` → `ImageSets/Main` → `train.txt`），修改 `yolov3.cfg` 与类别文件，从头训练自己的模型，最后用 `yolo_video.py` 批量检测并保存结果图。附加实验：用 TensorBoard 可视化训练过程。

**02 目标跟踪**
基于 OpenCV 内置算法，用 **至少三种算法对多目标进行跟踪**，比较跟踪效率。编写代码实现输入一段视频、框出多目标的运动轨迹；使用 **两种不同的评价指标**（OTB / VOT / MOTChallenge 或其他）评价多目标跟踪算法的精度、准确率与鲁棒性。

**03 视频摘要**
基于 **FCSN 网络** 与 **TVSum 数据集** 训练视频摘要模型，从数据集中选取 **10 段视频** 测试，生成各视频的摘要视频/图片并保存到单独文件夹；计算这 10 段视频的 **精度（Precision）、召回率（Recall）、F-score** 三种指标。

**04 视频检索**
基于深度学习方法，综合利用视频检索方法，实现 **以图搜图、以图搜视频**，比较搜索效率。实现输入一张图片即可检出相关的图像和视频；使用 **两种不同的评价指标** 评价搜索算法的性能与准确率。

---

## 二、目录结构

```
Professional_Comprehensive_Design/
├─ README.md                     # 本文件：课程说明、实验清单、进度
├─ docs/                         # 实验指导书、题目、报告模板（版权原因：仅本地保留，不入库）
│  ├─ course_materials/          #   原始 .docx / .pptx 资料（只读参考）
│  └─ extracted_text/            #   资料纯文本抽取结果，便于查阅与检索
├─ datasets/                     # 共享数据集，按实验分
│  ├─ classification/            #   01a 图像分类：CIFAR-10
│  ├─ cat_face/                  #   01b 目标检测：猫咪/猫脸数据集（VOC2007 格式）
│  ├─ tracking_videos/           #   02 目标跟踪：视频与标注
│  ├─ TVSum/                     #   03 视频摘要：TVSum
│  └─ retrieval/                 #   04 视频检索：图片/视频库
├─ experiments/
│  ├─ 01_TensorFlow_AlexNet/     #   01a code/ models/ logs/ outputs/ README.md requirements.txt
│  ├─ 01_ObjectDetection_YOLOv3/ #   01b code/ models/ outputs/ logs/ README.md requirements.txt
│  ├─ 02_ObjectTracking_OpenCV/  #   code/ outputs/ README.md requirements.txt
│  ├─ 03_VideoSummarization_FCSN/#   code/ models/ outputs/ logs/ README.md requirements.txt
│  └─ 04_VideoRetrieval/         #   code/ models/ outputs/ logs/ README.md requirements.txt
├─ reports/                      # 各实验的报告、图表
│  ├─ 0X_*_实验报告.md           #   脱敏后的 Markdown 报告（01a / 01b / 02 / 03 / 04）
│  ├─ figures/                   #   报告插图（曲线、对比图、样例检测结果）
│  └─ tables/                    #   报告表格（指标 CSV / Markdown）
├─ envs/                         # 实验虚拟环境与建环境脚本（环境本身不入库，setup_tf_env.py 入库）
└─ tools/                        # 公共小工具（文档抽取、数据集校验、labelImg 修补等）
```

每个实验目录的约定：

| 子目录 | 用途 |
|---|---|
| `code/` | 该实验全部源代码 |
| `models/` | 预训练权重与训练产物（`.h5` / `.pth` / `.pt`） |
| `outputs/` | 检测结果图、跟踪视频、摘要视频、检索结果等 |
| `logs/` | 训练日志、TensorBoard events、评测原始输出 |
| `README.md` | 该实验的详细说明：目标、环境、数据、运行步骤、结果、结论 |
| `requirements.txt` | 该实验的 Python 依赖 |

> 01a 与 01b 都只需要 TensorFlow 2.10 + numpy + matplotlib + Pillow，版本要求一致，
> 因此**共用同一个环境** `envs/pcd01_tf`，没有重复建环境。

### 仓库里有什么、没有什么

**入库**：各实验的源代码、`README.md`、`requirements.txt`、`reports/` 下的脱敏 Markdown 报告与插图、
数据集的说明文档与校验脚本。

**不入库**（已由 `.gitignore` 排除）：数据集本体、模型权重（`.h5` / `.pth` / `.pkl`）、
`outputs/` 与 `logs/` 下的运行产物、`envs/` 下的虚拟环境、课程原始资料（`.docx` / `.pptx` / `.pdf`，版权原因）
以及报告的 Word / PDF 版本。数据集与权重按各实验 README 的步骤重新下载或训练即可。

---

## 三、环境准备

各实验依赖差异较大，**建议按实验分组使用独立环境**。

```bash
# 01a / 01b 共用（TensorFlow 2.10 + Keras 2.10，Python 3.10，CPU）
python envs/setup_tf_env.py

# 其余实验按各自 requirements.txt 安装
conda create -n pcd02 python=3.9 -y
pip install -r experiments/02_ObjectTracking_OpenCV/requirements.txt
```

> **版本提示**：01a/01b 的指导书基于 `keras 2.2~2.4 + tensorflow 1.5~2.3` 这一较老组合，
> 本仓库实际用 `TF 2.10 + Keras 2.10`；实验 02 需要 `opencv-contrib-python`
> （跟踪算法在 contrib 模块中）；实验 03/04 建议 PyTorch。
>
> **为什么 01a/01b 只能用 CPU**：本机 RTX 4060 Laptop 是 sm_89（Ada），需要 CUDA 11.8+，
> 而 TF 2.10 绑定 CUDA 11.2（最高支持 sm_86），TF ≥ 2.11 又取消了 Windows 原生 GPU 支持。

---

## 四、数据准备

数据集体积较大，**不纳入版本控制**（见 `.gitignore`）。请按下表把数据放到对应目录。

| 实验 | 放置位置 | 获取方式 |
|:---:|---|---|
| 01a | `datasets/classification/cifar10/` | **脚本自动下载**：`python experiments/01_TensorFlow_AlexNet/code/data.py --download` |
| 01b | `datasets/cat_face/` | 猫咪图片放 `JPEGImages/`，LabelImg 标注结果 `.xml` 放 `Annotations/`（**已完成**：205 图 / 205 标注）|
| 02 | `datasets/tracking_videos/videos/` | 待跟踪视频（`.mp4` / `.avi`） |
| 03 | `datasets/TVSum/` | TVSum 视频与 `ydata-tvsum50-anno.tsv` 标注 |
| 04 | `datasets/retrieval/images/`、`videos/`、`queries/` | 检索库与查询图 |

各数据集的详细放置说明、命名规范与校验方式见 [datasets/README.md](datasets/README.md)，也可运行：

```bash
python tools/check_datasets.py
```

> **CIFAR-10 下载很慢是正常的**：本机到境外站点实测只有 0.02~0.07 MB/s，
> 且官方源每 8~9 秒停滞一次，全量约需 1.5~2.5 小时。
> `data.py` 已实现**多镜像 + 断点续传 + 停滞重试 + SHA256 校验**，
> 中途断了直接重跑同一命令即可从断点继续。详见
> [datasets/classification/README.md](datasets/classification/README.md)。

---

## 五、进度

| 实验 | 环境 | 数据 | 代码 | 训练/运行 | 评测 | 报告 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 01a AlexNet 图像分类 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 01b 目标检测（猫脸） | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 02 目标跟踪 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 03 视频摘要 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |
| 04 视频检索 | ☑ | ☑ | ☑ | ☑ | ☑ | ☑ |

> 图例：☐ 未开始 ／ ◐ 进行中 ／ ☑ 完成
> 4 个实验全部完成，报告见 `reports/` 下的 Markdown 文件。

### 01a TensorFlow / AlexNet 图像分类 · 当前状态

**已完成**

- 目录骨架、`config.py`（10 类中英文名 + 全部超参）、`requirements.txt`
- `data.py`：CIFAR-10 下载/解析/缓存/划分/归一化/增强。
  **下载部分不依赖 TF**，含多镜像、HTTP Range 断点续传、**重叠回退**、
  停滞重试、进度看门狗、SHA256 校验
- `alexnet.py`：基础 AlexNet（5 卷积 + 3 全连接，4.62 M 参数），激活函数与规范化方式可切换
- `train.py`：单组训练，含 `--benchmark` 实测速度、NaN 守卫、断点续训
- `evaluate.py`：accuracy / top-3 / 逐类 P-R-F1 / 混淆矩阵，带 `--selftest`
- `predict.py`：**输入一张图像 → 输出类别**（题目核心要求），带 `--selftest`
- `compare.py`：四组汇总与对比图；`make_predict_samples.py`：报告示例图
- `run_all.py`：一键流程（带 `--dry-run`）
- **自检共 160 项断言全过**（`test_pipeline.py` 74 + 四个 `--selftest` 57 + `test_cli.py` 29）
- **四组正式训练完成**：各 15 epoch，总用时 **3.25 小时**
- **最终结果**：

  | 激活函数 | 学习率 | 测试准确率 | top-3 | macro-F1 |
  |---|---:|---:|---:|---:|
  | **ReLU** | **0.01** | **0.8185** | **0.9633** | **0.8162** |
  | ReLU | 0.001 | 0.7768 | 0.9519 | 0.7771 |
  | LeakyReLU | 0.01 | 0.8055 | 0.9582 | 0.7987 |
  | LeakyReLU | 0.001 | 0.7706 | 0.9484 | 0.7699 |

  分组平均：ReLU 0.7976 vs LeakyReLU 0.7880（差 0.96 个百分点，**不显著**）；
  lr=0.01 0.8120 vs lr=0.001 0.7737（差 **3.83 个百分点，显著**）
- **三条核心结论**：① 激活函数无显著差异（推测 BN 已解决 ReLU 的神经元死亡问题）；
  ② 学习率影响是激活函数的 4 倍，但小学习率是"没训够"而非"更差"（损失仍在下降）；
  ③ `cat` 最难（Recall 仅 0.5310，18% 误判为 dog）
- **跨数据集验证**：4 张未参与训练的真实猫照片，3 张判为 `cat`
- 报告与 5 张插图完成：`reports/01b_AlexNet图像分类_实验报告.md`

### 01b 目标检测（猫脸）· 当前状态

**已完成**

- 数据整理进 `datasets/cat_face/`（205 张图 + 8 张演示图）
- 自动预标注：COCO 预训练 Faster R-CNN 生成 200 个 VOC XML
- **人工标注已完成**：205 张全部确认，框已收紧为猫脸，共 236 个目标框
- TensorFlow 环境建好（Python 3.10 + TF 2.10.0，CPU），解决 4 处 Keras API 兼容问题
- `yolov3.cfg` 已修正 3 处 `classes=1` / `filters=18`
- **正式训练完成**：200 epoch（其中 172 轮后因电脑重启，从 ep167 续训到 200）
- **最终指标**：AP = 0.5308、Recall 最高 0.88（阈值 0.003）；工作点阈值 0.03 时
  P=0.5556 / R=0.4000 / F1=0.4651
- 关键改进：默认 COCO anchors 让所有目标落到最粗尺度、细尺度零正样本；
  **改用本数据集聚类的 anchors 后，同样 20 epoch 下 AP 从 0.0011 提到 0.0752（68 倍）**
- 报告与 4 张插图已完成：`reports/01_目标检测_实验报告.md`

**报告**：[reports/01_目标检测_实验报告.md](reports/01_目标检测_实验报告.md)（含 4 张插图）。

### 一键跑完整流程

```bash
# 01b 目标检测
python experiments/01_ObjectDetection_YOLOv3/run_all.py --dry-run
python experiments/01_ObjectDetection_YOLOv3/run_all.py --epochs 200

# 01a AlexNet 图像分类（$TF 见对应 README）
& $TF experiments/01_TensorFlow_AlexNet/run_all.py --dry-run --tf-python $TF
& $TF experiments/01_TensorFlow_AlexNet/run_all.py --epochs 15 --tf-python $TF
```

详细说明见各自 README。

### 实验 02 / 03 / 04 结果摘要

**02 目标跟踪**（[报告](reports/02_目标跟踪_实验报告.md)）
用 OpenCV 内置的 6 种跟踪器（KCF / CSRT / MedianFlow + MIL / Boosting / TLD）同时跟踪
3 个目标，自建 `cross`、`occlude` 两段合成序列（640×360、25 fps、300 帧）并生成逐帧真值，
按 OTB / MOTChallenge / VOT 三套指标评测精度、成功率与鲁棒性。

- `cross` 序列 AUC：BOOSTING 0.843 > MedianFlow 0.794 > CSRT 0.772 > KCF 0.679
- 速度：MedianFlow ≈ 197 FPS、KCF ≈ 92 FPS、CSRT ≈ 12 FPS、MIL 仅 3.4 FPS
- MOSSE 在真实视频上 update 全部失败（0/59），说明"合成 patch 自检通过"不等于"真实序列可用"
- 结论：相关滤波（KCF/CSRT）在精度-速度上最均衡；纯色块目标会让 MedianFlow 无法初始化

**03 视频摘要**（[报告](reports/03_视频摘要_实验报告.md)）
基于 FCSN 与 TVSum（40 段训练 / 10 段测试），补齐官方缺失的 pool5 特征库与 KTS 镜头切分，
生成摘要视频与关键帧并计算 P / R / F-score。

- 10 段测试视频平均：Precision 0.3724、Recall 0.1961、F-score 0.2552（epoch-109），末轮 0.2230
- 同口径参照：线性岭回归基线 0.2561、用标准答案 gtscore 选段 0.2858、20 位标注者互评 0.366
  —— 模型达到理想选段的 89%、人类一致度的 70%
- 关键修复：上游 NLL + 反比加权的配方会让模型退化成常数输出（AUC 0.49）；
  换成 BCE + `pos_weight = n_neg / n_pos` 后 AUC 升到 0.62

**04 视频检索**（[报告](reports/04_视频检索_实验报告.md)）
ResNet50 深度特征与 HSV 颜色直方图两种表示，配暴力检索 + FAISS Flat / IVF / HNSW 四种索引，
实现以图搜图、以图搜视频，用 P@K 与 mAP 评价，并在三种库规模上测延迟。

- 以图搜图（120 库图 / 24 查询）：深度特征 mAP 0.9861、P@1 = 1.0000；直方图 mAP 0.9623
- 以图搜视频（12 视频 / 24 查询）：深度特征 mAP 1.0000（mean/max/frame-max 三种聚合同分）
- 数据集由 `make_dataset.py` 程序化合成，相关性真值由目录名推导，避免人工标注偏差

---

## 六、参考链接

- AlexNet 原始论文：<https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html>
- CIFAR-10 数据集：<https://www.cs.toronto.edu/~kriz/cifar.html>
- keras-yolo3（01b 基础代码）：<https://github.com/qqwweee/keras-yolo3>
- LabelImg（01b 标注工具）：<https://github.com/HumanSignal/labelImg>
- TVSum 数据集：<https://github.com/yalesong/tvsum>
- FCSN 视频摘要：<https://github.com/remega/video_summarization_fcsn>
- OpenCV 跟踪 API 文档：<https://docs.opencv.org/4.x/d9/df8/group__tracking.html>

