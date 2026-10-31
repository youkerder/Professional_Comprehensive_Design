# datasets/tracking_videos —— 实验 02 目标跟踪视频

完整说明见 [../README.md](../README.md#2-tracking_videos--实验-02-目标跟踪)。

## 放置清单

| 位置 | 放什么 | 状态 |
|---|---|:---:|
| `videos/` | 待跟踪视频（`.mp4` / `.avi`），含**多个运动目标** | ☐ |
| `annotations/` | 每帧真值框，与视频**同名** | ☐ |

## 命名约定

视频与真值文件同名，仅扩展名不同：

```
videos/person01.mp4      ↔  annotations/person01.txt          # OTB 格式
videos/person01.mp4      ↔  annotations/person01/groundtruth.txt   # VOT 格式
videos/person01.mp4      ↔  annotations/person01/gt/gt.txt         # MOT 格式
```

## 三种真值格式（互不兼容）

**OTB** —— `annotations/car4.txt`，每行 `x,y,w,h`：
```
1,1,120,80
2,2,121,81
```

**VOT** —— `annotations/car4/groundtruth.txt`，每行 `x,y,w,h`（也可能是 8 点旋转框）：
```
120.0,80.0,40.0,60.0
```

**MOTChallenge** —— `annotations/car4/gt/gt.txt`（另有 `seqinfo.ini`），
列顺序 `frame,id,x,y,w,h,conf,x,y,z`：
```
1,1,120,80,40,60,1,-1,-1,-1
```

## 视频来源建议

| 来源 | 真值 | 适用指标 |
|---|:---:|---|
| OTB-100 / OTB-50 | 自带 | OTB：Precision(20px)、Success(AUC)、SRE、TRE |
| VOT 数据集 | 自带 | VOT：Accuracy、Robustness、EAO |
| MOTChallenge（MOT16/17） | 自带 | MOT：MOTA、MOTP、IDF1、IDs、FM、ML |
| `chaplin.mp4`（指导书示例） | 无 | 只能看效果，首帧框 `(287, 23, 86, 320)` |
| 自行录制 | **需自己标** | 用 `experiments/02_.../code/make_gt.py` 标注 |

> **重要**：只有自标注真值或用公开数据集，才能计算精度类指标。自录视频不标真值就只能比较 FPS 和目视稳定性。

## 检查命令

```bash
python tools/check_datasets.py --only tracking_videos
```
