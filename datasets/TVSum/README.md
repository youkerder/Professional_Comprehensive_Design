# datasets/TVSum —— 实验 03 视频摘要数据集

完整说明见 [../README.md](../README.md#3-tvsum--实验-03-视频摘要)。

## 放置清单

| 位置 | 放什么 | 状态 |
|---|---|:---:|
| `videos/` | TVSum 的 **50 段视频** | ☐ |
| `annotations/` | `ydata-tvsum50-anno.tsv`、`ydata-tvsum50-info.tsv` | ☐ |
| `features/` | 抽好的帧级特征 `.npy`，形状 `(帧数, D)` | ☐ |

## 数据集要点

- **50 段视频**，来自 YouTube，共 10 个类别，每类 5 段。
- **20 名标注者** 对每一秒给出 1~5 的重要性评分。
- `ydata-tvsum50-anno.tsv` 列：`video_id`、`annotation_id`、`importance_scores`（**逗号分隔**的逐秒分数）。
- `ydata-tvsum50-info.tsv` 列：包含 `video_id`、`title`、`category`、`length`、`default_annotation` 等。
- 官方评测采用 **5 折交叉验证**（按类别划分折），通常报告各折平均值。

## 文件命名

用 `ydata-tvsum50-info.tsv` 里的 **`video_id`** 命名视频文件，保证与标注对得上：

```
videos/1.mp4              ← video_id = 1
videos/yJY1ISQ0KcY.mp4    ← video_id = yJY1ISQ0KcY
```

两种命名都常见，**关键是保持一致**。若你的视频文件名与 `video_id` 不同，
请维护一份映射表 `annotations/id_map.csv`（列 `video_id,filename`）。

## 特征提取

```bash
python experiments/03_VideoSummarization_FCSN/code/extract_features.py \
    --video_dir datasets/TVSum/videos \
    --out_dir   datasets/TVSum/features \
    --backbone googlenet
```

每段视频得到一个 `(帧数, 1024)` 的 `.npy`（GoogLeNet pool5）。

## 关于"选取 10 段视频"测试

实验要求从 50 段中**选 10 段**测试。请把选中的 `video_id` 列表写进
`experiments/03_VideoSummarization_FCSN/code/configs/test10.txt`，报告中也要明确列出。

## 检查命令

```bash
python tools/check_datasets.py --only TVSum
```
