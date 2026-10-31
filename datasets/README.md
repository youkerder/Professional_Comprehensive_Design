# datasets —— 数据集目录说明

数据集体积较大，**不纳入版本控制**。请按下表把数据放进对应位置。

> 快捷校验：`python tools/check_datasets.py`

---

## 0. `classification/` —— 实验一（TensorFlow）图像分类

题目《实验题目-4.docx》实验一「Tensorflow编程实验」用 **CIFAR-10**（32×32 彩色 10 类）。

```
datasets/classification/
└─ cifar10/
   ├─ cifar-10-python.tar.gz        # 原始压缩包（自动下载，约 162.6 MB）
   ├─ cifar10.npz                   # 解析缓存（自动生成，可删）
   └─ cifar-10-batches-py/          # 解压出的官方批次
```

**自动下载**（带断点续传、多镜像、SHA256 校验）：

```powershell
$TF = "D:\projects\Professional_Comprehensive_Design\envs\pcd01_tf\Scripts\python.exe"
& $TF experiments\01_TensorFlow_AlexNet\code\data.py --download
```

> **本机网络提示**：CIFAR-10 只有 `cs.toronto.edu` 一个官方源，实测仅 0.019 MB/s
> 且频繁停滞，全量要 2 小时以上。脚本默认优先试 Baidu BOS 镜像。
> **中途断了直接重跑同一命令即可从断点继续。**
> 详细的源对比与耗时见 [classification/README.md](classification/README.md)。

---

## 1. `cat_face/` —— 实验一 目标检测（猫咪/猫脸）

指导书以 `keras-yolo3` 为基础，数据集组织方式遵循 **Pascal VOC2007** 格式。

```
datasets/cat_face/
├─ JPEGImages/            # 原始图片：1.jpg, 2.jpg, ...（当前 205 张）
├─ Annotations/           # LabelImg 生成的标注：1.xml, 2.xml, ...（当前 1 个）
├─ ImageSets/Main/        # trainval/train/val/test 索引，由 make_imagesets.py 生成
├─ test/                  # 检测演示图 8 张（指导书 detect_img 的输入）
├─ raw/                   # 网盘下载的原始压缩包/未整理图片，先放这里
├─ predefined_classes.txt # LabelImg 预设类别，内容为 cat
└─ README.md
```

> 本目录原先是 `datasets/VOC2007/`，已整理为上面的形式。
> 详细状况与补标注步骤见 [cat_face/README.md](cat_face/README.md)。

| 位置 | 放什么 | 命名要求 |
|---|---|---|
| `raw/` | 从网盘下载的压缩包、未整理的猫咪图片 | 随意，仅作暂存 |
| `JPEGImages/` | 用于训练/验证的图片 | **纯英文/数字**，如 `000001.jpg`。指导书明确要求「图片以及相关文件夹命名应避免使用汉字」 |
| `Annotations/` | 与图片同名的 `.xml` 标注文件 | 与 `JPEGImages/` 中的文件名**一一对应**（仅扩展名不同） |
| `ImageSets/Main/` | `trainval.txt` / `train.txt` / `val.txt` / `test.txt` | 由 `make_imagesets.py` 生成，每行一个不含扩展名的文件名 |
| `test/` | 检测演示图，`yolo_video.py --image` 的输入 | 指导书里对应 `VOC2007/test/*.jpg` |

### 当前状况（重要）

| 项目 | 数量 |
|---|:---:|
| `JPEGImages/` 图片 | 205 |
| `Annotations/` 标注 | **205** |
| `test/` 演示图 | 8 |
| 目标框总数 | **236**（全部为猫脸框）|
| 标注覆盖率 | **100%** |

**标注已全部完成**：205 张图逐个人工确认，框已从自动预标注的"整只猫"收紧为"猫脸"。

> 自动预标注阶段共产生 289 个整猫框（COCO 预训练 Faster R-CNN）。
> 人工收紧到猫脸时一方面把框改小，另一方面删除了误检框，最终为 236 个猫脸框。
>
> **一个实际踩到的坑**：本想用 XML 里的 `source/database` 字段区分
> "哪些图还没人工过"，但 **LabelImg 保存时会把该字段统一覆盖成 `Unknown`**，
> 人工处理之后这个字段就完全失去区分能力了（实测 205 个 XML 全是 `Unknown`）。
> 正确做法是**把当前框坐标与预标注清单逐一比对**——
> 见 `experiments/01_ObjectDetection_YOLOv3/code/list_remaining.py`。

### 标注要点
- 用 **LabelImg**：`Open Dir` 选 `JPEGImages/`，`Change Save Dir` 选 `Annotations/`，格式切到 **PascalVOC**。
- `Create RectBox` 只框 **猫脸**，标签统一命名为 **`cat`**（全小写，不要出现 `Cat` / `cat_face` 等别名，否则类别数会算错）。
- 快捷键：`w` 画框、`d` 下一张、`a` 上一张、`Ctrl+S` 保存。
- 一键启动（目录已配好）：双击 `tools\start_labelimg.cmd`

### 生成索引与标注文件

```bash
cd experiments/01_ObjectDetection_YOLOv3/code
python make_imagesets.py     # 生成 ImageSets/Main 下的 4 个索引文件
python voc_annotation.py     # 生成 train.txt / val.txt / test.txt 与 voc_classes.txt
```

这两个脚本替代了指导书里的 `test.py` 和 keras-yolo3 自带的 `voc_annotation.py`，
**不需要再手动去掉 `2007_` 前缀**，并且会主动报告缺标注的图片。

### 关键坑
- 类别数必须三处一致：`model_data/voc_classes.txt`、`yolov3.cfg` 中 3 处 `yolo` 层的
  `classes` 与 `filters`（`filters = 3 * (5 + len(classes))`，本实验为 `classes=1`、`filters=18`）。
  否则会出现维度不匹配或检测框全错。
- 指导书原版 `test.py` 里的 `trainval_percent = 0.1` / `train_percent = 0.9` **写反了**，
  本仓库的 `make_imagesets.py` 已按 VOC 惯例修正（trainval 90%、test 10%）。

---

## 2. `tracking_videos/` —— 实验 02 目标跟踪

```
datasets/tracking_videos/
├─ videos/                # 待跟踪视频
├─ annotations/           # 真值标注（用于算指标）
└─ README.md
```

| 位置 | 放什么 | 格式要求 |
|---|---|---|
| `videos/` | 待跟踪视频 | `.mp4` / `.avi`，建议 720p 以内，时长 ≥10s 且包含 **多个运动目标** |
| `annotations/` | 每帧的目标真值框 | 与视频同名：OTB 风格 `car4.txt` 每行 `x,y,w,h`；VOT 风格 `car4/groundtruth.txt`；MOT 风格 `car4/gt/gt.txt` |

### 数据来源建议
- **OTB-100 / OTB-50**：经典单/多目标跟踪基准，自带真值框，最适合算 OTB 指标。
- **VOT** 数据集：自带 rotated bounding box，用于算 VOT 的 accuracy / robustness / EAO。
- **MOTChallenge**（如 MOT16/17）：多人多目标场景，自带 `gt.txt`，用于算 MOTA/MOTP/IDF1。
- **`chaplin.mp4`**（指导书示例用）：文档**没有提供文件也没有下载链接**，需自行准备。
  首帧固定框为 `bbox = (287, 23, 86, 320)`（`x, y, w, h`）。
- 也可以自行用手机拍摄多目标视频；**但只有自标注真值才能算指标**，否则只能在无真值时比较跟踪速度与目视稳定性。

### 命名约定
- 视频与标注文件**同名**（仅扩展名不同），例如 `videos/person01.mp4` ↔ `annotations/person01.txt`。
- 一个视频一个场景，不要把多个无关场景拼接在一起。

---

## 3. `TVSum/` —— 实验 03 视频摘要

TVSum 是 **50 段视频** 的有监督视频摘要数据集，每段视频带有逐秒的重要性评分（多个标注者的均值）。

```
datasets/TVSum/
├─ videos/                # 50 段视频：1.mp4 ... 50.mp4（或 yJY1ISQ0KcY.mp4 等 YouTube ID 命名）
├─ annotations/           # ydata-tvsum50-anno.tsv、ydata-tvsum50-info.tsv
├─ features/              # 预提取的 GoogLeNet / I3D 等特征（可选）
└─ README.md
```

| 位置 | 放什么 | 说明 |
|---|---|---|
| `videos/` | 50 段原始视频 | 必须与标注文件里的 `video_id` 对得上 |
| `annotations/` | `ydata-tvsum50-anno.tsv`（逐秒评分）、`ydata-tvsum50-info.tsv`（视频元信息、类别、时长） | 从 TVSum 官方仓库获取 |
| `features/` | 视频特征 `.npy`，形状通常为 `(帧数, 1024)` 或 `(帧数, 2048)` | 若直接跑 FCSN 需要；也可先跑特征提取脚本生成 |

### 本实验要求
- 从 50 段中 **选取 10 段** 测试，其余可用于训练。
- 每段视频的摘要输出 **保存到单独文件夹**，例如 `experiments/03_VideoSummarization_FCSN/outputs/summary/video_01/`。
- 计算这 10 段的 **Precision / Recall / F-score**（在摘要比例固定的设定下，通常取 `K = 15%` 的帧作为摘要）。

---

## 4. `retrieval/` —— 实验 04 视频检索

```
datasets/retrieval/
├─ images/                # 图像检索库（以图搜图的底库）
├─ videos/                # 视频检索库（以图搜视频的底库）
├─ queries/               # 查询图片
└─ README.md
```

| 位置 | 放什么 | 说明 |
|---|---|---|
| `images/` | 用于「以图搜图」的图片库 | 建议 ≥500 张，含明确类别，便于算准确率 |
| `videos/` | 用于「以图搜视频」的视频库 | 建议 ≥20 段；每段视频需能被抽帧提特征 |
| `queries/` | 查询图 | 每张查询图最好有已知的相关项（用于算 Precision@K / mAP） |

### 评测数据要求
要算「准确率」需要 **相关性真值**：即每张查询图对应哪些图/视频是正确答案。两种做法：
1. **按类别构造**：目录名即类别（`images/cat/*.jpg`、`images/dog/*.jpg`），同类别即相关 —— 最省事，推荐。
2. **人工标注**：维护 `queries/relevance.csv`，列为 `query, target, relevant`。

---

## 附：目录树一览

```
datasets/
├─ cat_face/
│  ├─ Annotations/        # VOC XML 标注
│  ├─ JPEGImages/         # 原图
│  ├─ ImageSets/Main/     # train/val/test 索引
│  └─ raw/                # 下载暂存
├─ tracking_videos/
│  ├─ videos/             # 待跟踪视频
│  └─ annotations/        # 跟踪真值
├─ TVSum/
│  ├─ videos/             # 50 段视频
│  ├─ annotations/        # TVSum 标注 tsv
│  └─ features/           # 预提取特征
└─ retrieval/
   ├─ images/             # 以图搜图底库
   ├─ videos/             # 以图搜视频底库
   └─ queries/            # 查询图
```
