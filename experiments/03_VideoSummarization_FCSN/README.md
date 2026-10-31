# 实验 03 · 视频摘要：FCSN + TVSum

> 题目来源：《实验题目-4.docx》实验六（课程资料，本地保留，不入库）

---

## 一、实验任务与要求

**任务**：基于 **FCSN 网络** 以及 **TVSum 数据集** 训练视频摘要模型。

**具体要求**
1. 在 TVSum 上训练视频摘要模型；
2. 从数据集中 **选取 10 段视频** 对模型进行测试；
3. 生成各视频的 **摘要视频 / 摘要图片**，保存到 **单独的文件夹**；
4. 计算这 10 段视频的三种性能评价指标：**精度（Precision）、召回率（Recall）、F-score**。

---

## 二、目录结构

```
experiments/03_VideoSummarization_FCSN/
├─ code/
│  ├─ config.py         # 路径与超参配置
│  ├─ dataset.py        # TVSum 数据集读取（视频/特征 + 逐秒评分）
│  ├─ extract_features.py  # 用预训练 GoogLeNet/I3D 抽取帧特征
│  ├─ kts.py            # 核时域分割（KTS），把视频切成不重叠片段
│  ├─ model.py          # FCSN 网络定义（全卷积 + 空洞卷积）
│  ├─ train.py          # 训练脚本
│  ├─ test.py           # 测试脚本：生成摘要并计算 Precision/Recall/F-score
│  ├─ evaluate.py       # 指标计算（含随机基线 Random / Oracle）
│  └─ make_summary.py   # 由预测分数生成摘要视频 / 摘要图，按视频分文件夹保存
├─ models/              # FCSN 权重（.pth / .h5）
├─ outputs/
│  ├─ scores/           #   各视频的逐帧/逐片段重要性分数
│  ├─ summary/          #   摘要结果，一个视频一个子文件夹
│  │  ├─ video_01/      #     摘要视频 + 关键帧图
│  │  └─ ...
│  └─ metrics/          #   评测结果表（csv / md）
├─ logs/                # 训练日志、TensorBoard events
├─ README.md
└─ requirements.txt
```

---

## 三、实验环境

```bash
conda create -n pcd03 python=3.8 -y
conda activate pcd03
pip install -r requirements.txt
```

说明：
- FCSN 原论文实现基于 TensorFlow 1.x；本实验目录以 **PyTorch** 复现为主（更易安装、便于调试）。
  若坚持用 TF 版，请把 `requirements.txt` 中的 torch 换成 `tensorflow==1.15`，并注意 TF1 的 `contrib` 依赖。
- 训练需要 GPU（建议 ≥6 GB 显存）。CPU 也可跑通，但很慢。

---

## 四、数据准备

TVSum 数据放在 [`datasets/TVSum/`](../../datasets/TVSum/)，详细说明见 [datasets/README.md](../../datasets/README.md)。

```
datasets/TVSum/
├─ videos/          # 50 段视频
├─ annotations/     # ydata-tvsum50-anno.tsv（逐秒评分）、ydata-tvsum50-info.tsv
└─ features/        # 预提取特征 .npy（可选）
```

### TVSum 数据集要点
- **50 段视频**，来自 YouTube，共 10 个类别（每个类别 5 段）；
- 每段视频由 **20 名标注者** 对每一秒给出 1~5 的重要性评分；
- `ydata-tvsum50-anno.tsv` 中每条评分是 `(video_id, annotation_id)` 下的一串逗号分隔的逐秒分数；
- 官方默认的评测方式是 **5 折交叉验证**（按类别划分折），报告时通常给出所有折的平均值。

### 特征提取
FCSN 不直接吃原始像素，而是吃**帧级特征**：
- 论文用 **GoogLeNet（ImageNet 预训练）** 的 pool5 输出（1024 维）；
- 也可换成 ResNet / I3D 特征（维度不同，需同步改 `model.py` 的输入通道）。

```bash
python code/extract_features.py \
    --video_dir ../../datasets/TVSum/videos \
    --out_dir   ../../datasets/TVSum/features \
    --backbone googlenet
```

每段视频得到一个 `(帧数, 1024)` 的 `.npy`。

### KTS 分割
用 **Kernel Temporal Segmentation** 把视频切成不重叠的片段（shot），
FCSN 对 **片段** 打分而不是对帧打分，这样摘要更连贯。片段边界来自特征序列的核相关矩阵。

---

## 五、FCSN 模型简介

**FCSN（Fully Convolutional Sequence Network）** 的核心思想：

1. **全卷积**：不含全连接层，因此可以处理任意长度的视频，参数量与视频长度无关；
2. **空洞卷积（dilated convolution）**：在不降分辨率的前提下扩大感受野，捕捉长时序依赖；
3. **序列到序列**：输入 `(片段数, 特征维度)` 的序列，输出每个片段的**重要性分数**；
4. **训练目标**：有监督回归，用逐片段的人类评分均值作为标签，损失通常为 **MSE**；
5. 推理时把片段分数映射回帧，按 **固定摘要比例 K%**（TVSum 常用 **15%**）选取分数最高的帧作为摘要。

> 与 LSTM 类方法的对比是常见的分析点：FCSN 并行性好、长依赖建模更强，且不依赖单向递推。

---

## 六、训练

```bash
python code/train.py --config configs/tvsum_fcsn.yaml
```

关键超参（建议在报告中列出表格）：

| 超参 | 建议值 | 说明 |
|---|---|---|
| 输入特征 | GoogLeNet pool5，1024 维 | 特征维度必须与模型输入一致 |
| 学习率 | 1e-4 ~ 1e-3 | 可比较两种学习率 |
| 优化器 | Adam | |
| 损失 | MSE | 回归逐片段评分 |
| batch size | 1 段视频（不定长序列） | 变长输入一般逐样本训练 |
| epoch | 100 ~ 300 | 早停 + 保存验证集最优 |
| 摘要比例 K | 15% | TVSum 官方评测常用值 |

权重保存到 `models/`，训练日志与 events 保存到 `logs/`。

---

## 七、测试与摘要生成

```bash
# 1) 用训练好的模型对 10 段测试视频预测重要性分数
python code/test.py --weights models/fcsn_best.pth --split test10

# 2) 生成摘要视频 / 关键帧图，一视频一个文件夹
python code/make_summary.py --scores outputs/scores --out outputs/summary --top_ratio 0.15

# 3) 计算 Precision / Recall / F-score
python code/evaluate.py --scores outputs/scores --anno ../../datasets/TVSum/annotations/ydata-tvsum50-anno.tsv
```

### 指标定义
在**固定摘要比例 K%** 的设定下，把预测摘要与用户摘要（人类评分高于阈值的帧）对比：

```
Precision = |预测摘要帧 ∩ 用户摘要帧| / |预测摘要帧|
Recall    = |预测摘要帧 ∩ 用户摘要帧| / |用户摘要帧|
F-score   = 2 * Precision * Recall / (Precision + Recall)
```

由于每个视频有 20 个标注者，实践中有两种做法（建议都算，报告中说明）：
- **逐标注者计算再平均**：对 20 个用户摘要分别算 P/R/F，再取均值；
- **平均评分阈值化**：用 20 人的平均分做阈值，生成单个"用户摘要"再计算。

同时建议给出 **随机基线（Random）** 作为对照 —— 随机抽取同样比例的帧，重复若干次取均值。

### 输出要求
- 摘要视频：`outputs/summary/video_XX/summary.mp4`
- 摘要图片（关键帧拼图）：`outputs/summary/video_XX/keyframes.jpg`
- 指标表：`outputs/metrics/scores.csv` 与汇总的均值行

---

## 八、结果记录

| 项目 | 位置 |
|---|---|
| 10 段测试视频的逐视频 P / R / F | `outputs/metrics/scores.csv` |
| 三种指标的平均值 | 同上，末行 summary |
| 随机基线对比 | `outputs/metrics/random_baseline.csv` |
| 摘要结果（视频/图） | `outputs/summary/video_XX/` |
| 训练损失曲线 | `reports/figures/03_loss_curve.png` |

报告需明确写出：**选取的是哪 10 段视频**（给出 video_id 列表）、**摘要比例 K 取多少**、
指标是「逐标注者平均」还是「平均分阈值化」。

---

## 九、常见问题

| 现象 | 原因与解决 |
|---|---|
| 特征与模型维度不匹配 | 换 backbone 后特征维度变了，需同步改 `model.py` 的输入维度 |
| 变长序列无法组 batch | 变长输入用 batch_size=1 逐样本训练，或做 padding + mask |
| 摘要结果全是相邻帧 | 只对帧打分未做 KTS 片段分割；应先分割再对片段打分 |
| F-score 异常高 | 检查摘要比例是否与用户摘要比例一致（两边都必须用 K%=15%） |
| 标注读取报错 | `ydata-tvsum50-anno.tsv` 的分数是逗号分隔字符串，需按 `\t` 分列后再 split |
| 视频下载后与标注对不上 | 用 `ydata-tvsum50-info.tsv` 中的 `video_id` 重命名视频文件 |

---

## 十、参考

- FCSN 论文：Rochan et al., *Video Summarization Using Fully Convolutional Sequence Networks*, ECCV 2018
- TensorFlow 参考实现：<https://github.com/remega/video_summarization_fcsn>
- TVSum 数据集：<https://github.com/yalesong/tvsum>
- TVSum 标注说明：<https://github.com/yalesong/tvsum#dataset>
