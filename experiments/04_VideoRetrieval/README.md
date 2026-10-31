# 04 · 视频检索：以图搜图 / 以图搜视频

> 规格文档：原始 .docx / .pptx 因版权仅本地保留，不入库，说明见 [docs/README.md](../../docs/README.md)

---

## 1. 任务定义

**任务**：基于深度学习方法，综合利用视频检索方法，实现 **以图搜图**、**以图搜视频**，并比较搜索的效率。

**具体要求**
1. 借鉴深度学习模型框架，实现 **输入一张图片**，可以检出相关的 **图像和视频**；
2. 比较 **搜索的效率**；
3. 使用 **两种不同的评价指标** 来评价搜索算法的性能与准确率。

---

## 2. 目录结构

```
experiments/04_VideoRetrieval/
├─ code/
│  ├─ config.py            # 路径与参数配置
│  ├─ extract_image_feats.py  # 图像库特征提取
│  ├─ extract_video_feats.py  # 视频库抽帧 + 特征提取 + 视频级聚合
│  ├─ build_index.py       # 建索引（暴力 / FAISS / HNSW）
│  ├─ search.py            # 检索入口：输入一张图，输出相关图像与视频
│  ├─ evaluate.py          # 两种评价指标：Precision@K、mAP（另含 Recall@K）
│  ├─ benchmark.py         # 效率对比：建索引耗时、单次查询延迟、QPS、内存
│  └─ utils.py             # 特征归一化、相似度计算、结果可视化
├─ models/                 # 预训练权重（ResNet50 / CLIP 等）
├─ outputs/
│  ├─ features/            #   抽好的特征（.npy / .h5）
│  ├─ index/               #   检索索引文件
│  ├─ results/             #   每个查询图的检索结果（图 + 视频列表）
│  └─ metrics/             #   指标与效率对比表（csv / md）
├─ logs/
├─ README.md
└─ requirements.txt
```

---

## 3. 运行环境

```bash
conda create -n pcd04 python=3.8 -y
conda activate pcd04
pip install -r requirements.txt
```

- **必须要有 GPU 才能体现出检索效率的差异**；CPU 也能跑通，但特征提取会很慢。
- 建议同时对比 **暴力检索（brute-force）** 与 **FAISS / HNSW 近似检索**，这正是"比较搜索效率"的核心。

---

## 4. 数据准备

数据放在 [`datasets/retrieval/`](../../datasets/retrieval/)，详细说明见 [datasets/README.md](../../datasets/README.md)。

```
datasets/retrieval/
├─ images/     # 以图搜图的底库（建议按类别分子目录）
├─ videos/     # 以图搜视频的底库
└─ queries/    # 查询图片
```

### 按类别构造（推荐，最省事）
让目录名即类别，同类别即"相关"：

```
datasets/retrieval/images/
├─ cat/     *.jpg
├─ dog/     *.jpg
└─ car/     *.jpg
```

这样 `evaluate.py` 可以自动从路径推导相关性真值，无需人工标注。
若要更严谨，可维护 `datasets/retrieval/queries/relevance.csv`，列为 `query,target,relevant`。

> **注意**：检索库中的图片不要与查询图完全相同（否则就是"自己搜自己"），
> 建议查询图取自同一类别但**不同实例**，或对查询图做裁剪/缩放增强。

---

## 5. 方法

### 5.1 深度学习特征提取

用 ImageNet（或更大规模）预训练的分类/多模态模型做 **特征提取器**（去掉最后的分类层）：

| 骨干 | 特征维度 | 特点 |
|---|---|---|
| ResNet50 | 2048 | 通用、稳定、速度快，最常用 |
| VGG16 | 4096 | 经典，维度高、检索慢 |
| CLIP (ViT-B/32) | 512 | 图文对齐、语义强，跨模态检索效果好 |

要求：特征做 **L2 归一化** 后，用 **余弦相似度** 检索。

### 5.2 以图搜图
1. 遍历 `datasets/retrieval/images/`，逐图提特征 → 得到 `(N, D)` 特征矩阵；
2. 查询图提特征 → `(1, D)`；
3. 计算余弦相似度 → 排序 → 返回 Top-K 图像。

### 5.3 以图搜视频
视频不能直接送进图像模型，标准做法是 **抽帧 + 特征 + 聚合**：

1. 按固定间隔（如每 1 秒 1 帧，或均匀抽 16/32 帧）从每段视频抽帧；
2. 每帧提特征 → 得到 `(T, D)`；
3. **视频级聚合**，三种可选（建议对比）：
   - **平均池化**：`mean` over frames，最简单；
   - **最大池化**：`max`，突出最显著帧；
   - **帧级检索**：不做聚合，直接把所有帧当候选，返回命中的视频（召回高、速度慢）；
4. 得到每段视频一个 `(D,)` 特征，与查询图算相似度 → 返回 Top-K 视频。

> **关键点**：图像与视频的特征必须在**同一个特征空间**（同一骨干、同一预处理）才能比较，否则相似度没有意义。

### 5.4 检索效率优化
| 方案 | 说明 | 复杂度 |
|---|---|---|
| 暴力检索 | 与全库特征逐一算余弦相似度 | O(N·D) |
| FAISS `IndexFlatIP` | 精确内积检索，仍比 numpy 快 | O(N·D)，常数更小 |
| FAISS `IndexIVFFlat` | 倒排索引 + 聚类，近似检索 | O(N·D / nlist) |
| HNSW | 图索引，高召回低延迟，适合在线检索 | ≈ O(log N) |

---

## 6. 运行

```bash
# 1) 提取图像库特征
python code/extract_image_feats.py --data ../../datasets/retrieval/images \
    --backbone resnet50 --out outputs/features/images_resnet50.npy

# 2) 提取视频库特征（抽帧 + 聚合）
python code/extract_video_feats.py --data ../../datasets/retrieval/videos \
    --backbone resnet50 --fps 1 --pool mean --out outputs/features/videos_resnet50.npy

# 3) 建索引
python code/build_index.py --feats outputs/features/images_resnet50.npy \
    --index_type faiss_ivf --out outputs/index/images.faiss

# 4) 检索：输入一张图，同时输出相关图像和视频
python code/search.py --query ../../datasets/retrieval/queries/cat_01.jpg \
    --image_index outputs/index/images.faiss --video_feats outputs/features/videos_resnet50.npy \
    --topk 10 --out outputs/results/

# 5) 评价准确率（两种指标）
python code/evaluate.py --queries ../../datasets/retrieval/queries --topk 10

# 6) 效率对比
python code/benchmark.py --feats outputs/features/images_resnet50.npy \
    --index_types brute,faiss_flat,faiss_ivf,hnsw --repeat 100
```

---

## 7. 评价指标

任务要求 **两种不同的评价指标**。下面给出可选集合，**至少选两种**（推荐 ① + ②）：

### ① Precision@K（准确率）
返回的 Top-K 结果中相关项的比例：

```
Precision@K = |Top-K 中相关项| / K
```

### ② mAP（平均精度均值）—— 综合了排序质量
```
AP      = Σ_{k=1..K} ( Precision@k × rel(k) ) / |相关项总数|
mAP     = 所有查询图的 AP 的均值
```

### ③ Recall@K（召回率）
```
Recall@K = |Top-K 中相关项| / |全部相关项|
```

### ④ 效率指标（对应"比较搜索的效率"）
| 指标 | 含义 |
|---|---|
| 建索引耗时 (s) | 构建检索结构的代价 |
| 单次查询延迟 (ms) | 用户感知的响应速度 |
| QPS | 每秒可处理查询数 |
| 内存占用 (MB) | 索引规模 |

> 建议用一张表对比：**不同骨干（ResNet50 / VGG16 / CLIP）× 不同索引（brute / flat / IVF / HNSW）**
> 在 mAP 与查询延迟上的权衡，这是本模块最有价值的分析点。

---

## 8. 结果

| 项目 | 位置 |
|---|---|
| 准确率指标（Precision@K / mAP / Recall@K） | `outputs/metrics/accuracy.csv` |
| 效率指标（延迟 / QPS / 内存 / 建索引耗时） | `outputs/metrics/efficiency.csv` |
| 每个查询图的检索结果 | `outputs/results/<query_name>/` |
| 不同骨干 / 索引的对比表 | `outputs/metrics/compare.md` |
| 报告插图（结果拼图、延迟对比柱状图） | `reports/figures/04_*.png` |

---

## 9. 故障排查

| 现象 | 原因与解决 |
|---|---|
| 检索结果全是同一张图 | 查询图与底库图重复。查询图应取自不同实例或做裁剪增强 |
| mAP 很低但目视结果不错 | 相关性真值定义过严。按类别构造时同类别都算相关 |
| 以图搜视频结果不稳定 | 视频聚合方式不合适。对比 mean / max 池化，或改帧级检索 |
| 图像与视频无法比较 | 两者特征不是同一骨干/同一预处理，需统一 |
| FAISS 报维度错误 | 特征没做 L2 归一化或维度不一致（检查 `--backbone`） |
| 效率对比不公平 | 要在**同一硬件、同一特征、同一 K**下比较，且查询要有 warm-up |
| CPU 上跑不动 | 特征提取阶段用 GPU；用 CLIP 等小模型或降低底库规模 |

---

## 10. 参考资料

- ResNet：He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- CLIP：Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021
- FAISS：<https://github.com/facebookresearch/faiss>
- HNSW：Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, TPAMI 2018
