# datasets/cat_face —— 实验 01 猫咪/猫脸数据集

完整说明见 [../README.md](../README.md#1-cat_face--实验-01-目标检测猫咪猫脸)。

> 本目录原先以 `datasets/VOC2007/` 的形式放在 `datasets/` 下，
> 已按其内容整理进本目录（`VOC2007` 的空壳子目录已移除）。

## 目录结构

```
datasets/cat_face/
├─ JPEGImages/            205 张猫咪图片（训练/验证用）
├─ Annotations/           LabelImg 生成的同名 .xml（PascalVOC 格式）
├─ ImageSets/Main/        trainval / train / val / test 索引（脚本生成）
├─ test/                  8 张检测演示图（`detect_img` 的输入）
├─ raw/                   下载的原始压缩包暂存
├─ predefined_classes.txt LabelImg 预设类别，内容为 cat
└─ README.md
```

## 当前数据状况

| 项目 | 数量 | 说明 |
|---|:---:|---|
| `JPEGImages/` 图片 | **205** | 编号 1.jpg ~ 205.jpg（非连续） |
| `Annotations/` 标注 | **1** | 只有 `1.xml` |
| `test/` 检测演示图 | **8** | `11, 60, 109, 147, 150, 165, 193, 202`.jpg |
| 标注覆盖率 | **0.5%** | 205 张里只有 1 张有标注 |

### ⚠️ 训练前必须先补标注

YOLOv3 从头训练时，**没有标注的图片会被当成纯背景**。
目前只有 1 张图有框，直接训练得不到任何可用的猫脸检测器。
上游环境用了 **230 张全部标注**的图片。

需要用 LabelImg 把 `JPEGImages/` 里剩余的 204 张补齐（标签统一填 `cat`）。

## 各文件的作用

| 位置 | 放什么 | 命名要求 |
|---|---|---|
| `raw/` | 网盘下载的压缩包、未整理的图片 | 随意，仅作暂存 |
| `JPEGImages/` | 训练/验证用的图片 | **纯英文/数字**，如 `000001.jpg`。命名规范要求「图片以及相关文件夹命名应避免使用汉字」 |
| `Annotations/` | 与图片同名的 `.xml` 标注 | 与 `JPEGImages/` 中的文件名**一一对应**（仅扩展名不同） |
| `ImageSets/Main/` | `trainval.txt` / `train.txt` / `val.txt` / `test.txt` | 由 `make_imagesets.py` 生成，每行一个不含扩展名的文件名 |
| `test/` | 检测演示图，跑 `yolo_video.py --image` 的输入 | 与 `JPEGImages/` 相区分；上游对应 `VOC2007/test/*.jpg` |

## 用 LabelImg 补标注

1. 安装并启动：
   ```bash
   pip install labelImg==1.8.6
   labelImg
   ```
2. 界面左下角切到 **PascalVOC** 格式。
3. `Open Dir` → 选 `datasets/cat_face/JPEGImages`
4. `Change Save Dir` → 选 `datasets/cat_face/Annotations`
5. `Create RectBox` → 框出**猫脸**，标签名填 **`cat`**（全小写）
6. 用快捷键提高效率：`w` 画框、`d` 下一张、`a` 上一张、`Ctrl+S` 保存

> **标签名必须完全一致**：全部用 `cat`。出现 `Cat` / `cat_face` / `catface` 等别名会被当成不同类别，
> 导致类别数算错、训练维度不匹配。
>
> `predefined_classes.txt` 里已经写好 `cat`。LabelImg 实际读取的是**它自己安装目录**下的
> `data/predefined_classes.txt`，需要自动带出默认标签的话把本文件内容复制过去，
> 或者勾选「Use default label」并填 `cat`。

## 生成索引与标注文件

标注完成后依次运行（在 `experiments/01_ObjectDetection_YOLOv3/code/` 下）：

```bash
python make_imagesets.py     # 生成 ImageSets/Main 下的 4 个索引文件
python voc_annotation.py     # 生成 train.txt / val.txt / test.txt 与 voc_classes.txt
```

两个脚本都会**主动报告缺标注的图片数量和编号**，标注有没有补完一眼就能看出来。

## 检查命令

```bash
python tools/check_datasets.py --only cat_face
```

会检查：图片与标注是否配对、文件名是否含中文、索引文件是否齐全、
**XML 里的标签名是否统一为 `cat`**、图片数量是否偏少。

## 关键坑

| 现象 | 原因与解决 |
|---|---|
| 类别数不对 / 训练维度不匹配 | `voc_classes.txt` 与 `yolov3.cfg` 里的 `classes` 没同步改成 1；`filters` 应为 `3*(5+1)=18` |
| 读不到图片 / 乱码 | 图片或文件夹名含中文 |
| 未标注的图片拖低效果 | 205 张里 204 张没框，会被当成背景。**必须补标注**，或把这些图临时移出 `JPEGImages/` |
| 索引文件是空的 | 还没标注，或标注的 `.xml` 文件名与图片对不上 |
