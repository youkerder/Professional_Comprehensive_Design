# reports —— 技术报告与图表

## 目录

```
reports/
├─ 01b-alexnet-classification.md   # 01a 图像分类（AlexNet）
├─ 01-object-detection.md          # 01b 目标检测（YOLOv3 猫脸）
├─ 02-multi-object-tracking.md     # 02 多目标跟踪
├─ 03-video-summarization.md       # 03 视频摘要（FCSN + TVSum）
├─ 04-video-retrieval.md           # 04 视频检索
├─ figures/                        # 报告插图（曲线、对比图、样例结果截图）
└─ tables/                         # 报告表格（指标 CSV / Markdown）
```

> 01a 与 01b 是两条独立任务线，各有一份报告。
> 插图与表格文件名用 `01b_` / `01_` 前缀区分，避免混淆。
> 报告由各模块的 `code/build_report*.py` 从结果文件生成，不手工拼数。

---

## 报告结构

每份报告按以下结构组织，与对应模块 README 的小节一一对应：

```markdown
# <模块编号> · <任务名> 技术报告

## 1. 目标            # 任务定义与验收标准
## 2. 运行环境        # 硬件 / 系统 / Python 与关键库版本，引用 envs/ 下的环境记录
## 3. 原理            # 所用模型或算法的原理，不整段照搬上游资料
## 4. 数据集          # 来源、规模、划分、预处理，给出统计表格
## 5. 实现过程        # 关键步骤 + 可复制的命令 + 超参表
## 6. 结果            # 数值指标表 + 可视化图
## 7. 结果与分析      # 不同算法 / 超参的对比、原因分析、失败案例
## 8. 问题与处理      # 踩到的问题与解决办法
## 9. 结论与改进方向
```

---

## 图表编号约定

正文中的图表统一编号，并在 `figures/` / `tables/` 中留**同名文件**，便于复用与核对：

| 报告 | 图 | 文件 |
|---|---|---|
| 01a | 图 1-1 四组验证集准确率曲线 | `figures/01b_curves_val_acc.png` |
| 01a | 图 1-2 四组训练损失曲线 | `figures/01b_curves_loss.png` |
| 01a | 图 1-3 测试准确率对比（激活函数 × 学习率）| `figures/01b_test_accuracy.png` |
| 01a | 图 1-4 混淆矩阵 | `figures/01b_confusion_<run>.png` |
| 01a | 图 1-5 预测结果示例 | `figures/01b_predict_samples.png` |
| 01b | 图 1-1 训练损失曲线 | `figures/01_loss_curve.png` |
| 01b | 图 1-2 猫脸检测结果 | `figures/01_detect_samples.png` |
| 02 | 图 2-1 多目标运动轨迹 | `figures/02_trajectory.png` |
| 02 | 图 2-2 各算法 FPS 对比 | `figures/02_fps_compare.png` |
| 03 | 图 3-1 摘要关键帧 | `figures/03_keyframes.png` |
| 04 | 图 4-1 检索结果示例 | `figures/04_retrieval_samples.png` |

| 报告 | 表 | 文件 |
|---|---|---|
| 01a | 表 1-1 四组对比汇总 | `tables/01b_compare.md` |
| 01a | 表 1-2 各测试集指标明细 | `tables/01b_eval_<run>.md` |
| 02 | 表 2-1 各算法指标对比 | `tables/02_metrics_compare.csv` |
| 03 | 表 3-1 10 段视频 P/R/F | `tables/03_prf_scores.csv` |
| 04 | 表 4-1 准确率与效率对比 | `tables/04_accuracy_efficiency.csv` |

---

## 交付前自检

- [ ] `experiments/0X_*/README.md` 的运行步骤可照做跑通
- [ ] `experiments/0X_*/code/` 代码齐全，无硬编码绝对路径与本机凭据
- [ ] `experiments/0X_*/outputs/` 有结果（图 / 视频 / 指标表）
- [ ] `reports/0X-*.md` 的图表编号与 `figures/`、`tables/` 一一对应
- [ ] `envs/` 下有对应环境的版本记录或构建脚本
