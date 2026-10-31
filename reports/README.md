# reports —— 实验报告与图表

## 目录

```
reports/
├─ 01b_AlexNet图像分类_实验报告.md   # 实验一（TensorFlow 编程实验）
├─ 01_目标检测_实验报告.md           # 实验一（目标检测 / YOLOv3 猫脸）
├─ 02_目标跟踪_实验报告.md
├─ 03_视频摘要_实验报告.md
├─ 04_视频检索_实验报告.md
├─ figures/     # 报告插图（曲线、对比图、样例结果截图）
└─ tables/      # 报告表格（指标 CSV / Markdown）
```

> 实验一按题目要求分成两部分，所以有两份报告：
> `01b_...` 对应《实验题目-4.docx》的「实验一. Tensorflow编程实验」（AlexNet 图像分类），
> `01_...` 对应《目标检测实验指导书》的猫脸检测。
> 插图表名用 `01b_` 前缀区分。

报告格式要求见 [../docs/README.md](../docs/README.md#3-实验报告要求)。

---

## 报告章节骨架

每份报告按下面的结构写，与各实验 README 中的小节一一对应：

```markdown
# 实验 X · <实验名> 实验报告

**姓名**：　　　**学号**：　　　**班级**：　　　**日期**：

## 一、实验目的
## 二、实验环境        # 硬件 / 系统 / Python 与关键库版本，可引用 envs/ 下的环境文件
## 三、实验原理        # 所用模型或算法的原理，不要整段抄指导书
## 四、数据集          # 来源、规模、划分、预处理，给出统计表格
## 五、实验步骤        # 关键步骤 + 可复制的命令 + 超参表
## 六、实验结果        # 数值指标表 + 可视化图
## 七、结果分析        # 不同算法/超参的对比、原因分析、失败案例
## 八、实验结论与心得  # 遇到的问题、解决办法、可改进方向
```

---

## 图表编号约定

正文中的图表统一编号，并在 `figures/` / `tables/` 中留**同名文件**，便于复用与核对：

| 报告 | 图 | 文件 |
|---|---|---|
| 01b | 图 1-1 四组验证集准确率曲线 | `figures/01b_curves_val_acc.png` |
| 01b | 图 1-2 四组训练损失曲线 | `figures/01b_curves_loss.png` |
| 01b | 图 1-3 测试准确率对比（激活函数 × 学习率）| `figures/01b_test_accuracy.png` |
| 01b | 图 1-4 混淆矩阵 | `figures/01b_confusion_<run>.png` |
| 01b | 图 1-5 预测结果示例 | `figures/01b_predict_samples.png` |
| 01 | 图 1-1 训练损失曲线 | `figures/01_loss_curve.png` |
| 01 | 图 1-2 猫脸检测结果 | `figures/01_detect_samples.png` |
| 02 | 图 2-1 多目标运动轨迹 | `figures/02_trajectory.png` |
| 02 | 图 2-2 各算法 FPS 对比 | `figures/02_fps_compare.png` |
| 03 | 图 3-1 摘要关键帧 | `figures/03_keyframes.png` |
| 04 | 图 4-1 检索结果示例 | `figures/04_retrieval_samples.png` |

| 报告 | 表 | 文件 |
|---|---|---|
| 01b | 表 1-1 四组对比汇总 | `tables/01b_compare.md` |
| 01b | 表 1-2 各测试集指标明细 | `tables/01b_eval_<run>.md` |
| 02 | 表 2-1 各算法指标对比 | `tables/02_metrics_compare.csv` |
| 03 | 表 3-1 10 段视频 P/R/F | `tables/03_prf_scores.csv` |
| 04 | 表 4-1 准确率与效率对比 | `tables/04_accuracy_efficiency.csv` |

---

## 提交清单

每个实验提交时确认：

- ☐ `experiments/0X_*/README.md` 里的运行步骤能照着跑通
- ☐ `experiments/0X_*/code/` 下的代码齐全，无硬编码的绝对路径
- ☐ `experiments/0X_*/outputs/` 下有结果（图 / 视频 / 指标表）
- ☐ `reports/0X_*.md` 报告完成，图表编号与 `figures/` `tables/` 对应
- ☐ `envs/` 下有该实验的环境导出文件
- ☐ 根 `README.md` 的进度表已更新为 ☑
