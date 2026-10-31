# datasets/retrieval —— 实验 04 视频检索底库

完整说明见 [../README.md](../README.md#4-retrieval--实验-04-视频检索)。

## 放置清单

| 位置 | 放什么 | 建议规模 | 状态 |
|---|---|:---:|:---:|
| `images/` | 「以图搜图」的图片底库 | ≥500 张 | ☐ |
| `videos/` | 「以图搜视频」的视频底库 | ≥20 段 | ☐ |
| `queries/` | 查询图片 | ≥20 张 | ☐ |

## 推荐：按类别构造（最省事）

让**目录名即类别**，同类别即"相关"。这样 `evaluate.py` 能自动推导相关性真值，**无需人工标注**：

```
images/
├─ cat/    cat_001.jpg  cat_002.jpg  ...
├─ dog/    dog_001.jpg  ...
└─ car/    car_001.jpg  ...

queries/
├─ cat/    q_cat_01.jpg      ← 应与 images/cat/ 中的图不同实例
├─ dog/    q_dog_01.jpg
└─ car/    q_car_01.jpg
```

## 备选：人工标注相关性

维护 `queries/relevance.csv`：

```csv
query,target,relevant
q_cat_01.jpg,images/cat/cat_001.jpg,1
q_cat_01.jpg,images/dog/dog_001.jpg,0
```

## 视频库要求

- 每段视频建议 **10 秒 ~ 2 分钟**，内容相对单一（便于用类别定义相关性）。
- 视频文件可放在 `videos/<类别>/xxx.mp4`，同样支持按类别推导真值。

## ⚠️ 关键要求

1. **查询图不要与底库图完全相同**，否则就是"自己搜自己"，指标虚高。
   查询图应取自**同类别的不同实例**，或对底库图做裁剪 / 缩放 / 加噪增强。
2. **图像与视频的特征必须来自同一骨干、同一预处理**，否则两者相似度没有意义。
3. 底库规模越大，"比较搜索效率"的差异越明显（暴力检索的延迟会显著上升）。

## 检查命令

```bash
python tools/check_datasets.py --only retrieval
```
