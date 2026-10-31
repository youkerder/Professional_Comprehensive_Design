# 实验一（TensorFlow 编程实验）· AlexNet 图像分类

> 题目来源：《实验题目-4.docx》**实验一. Tensorflow编程实验**
> （纯文本为课程资料，本地保留，不入库）

---

## 一、实验任务与目的

**题目原文**

> 基于 tensorflow 的框架，搭建一个基础的 AlexNet 的网络模型，对图像进行分类。
> 具体要求为，编写程序实现**输入一张图像，输出图像的类别**，使用**两种不同的激活函数**
> 进行比较，**ReLU 以及任意另一种**，使用**两种不同的学习率**进行比较。

**拆解成三件事**

| # | 要求 | 本实验的做法 | 对应文件 |
|:---:|---|---|---|
| 1 | 输入一张图像 → 输出类别 | `predict.py`：打印类别序号、英文名、中文名、置信度、top-3 | [`code/predict.py`](code/predict.py) |
| 2 | 两种激活函数对比（ReLU + 另一种） | ReLU vs **LeakyReLU**（alpha=0.1） | [`code/alexnet.py`](code/alexnet.py) |
| 3 | 两种学习率对比 | SGD+momentum，lr = **0.01 vs 0.001** | [`code/train.py`](code/train.py) |

**目的**：掌握用 TensorFlow/Keras 搭建卷积网络、组织训练流程、做控制变量对比实验，
并学会用混淆矩阵与逐类 P/R/F1 分析分类结果，而不只看一个总准确率。

---

## 二、目录结构

```
experiments/01_TensorFlow_AlexNet/
├─ code/
│  ├─ config.py         # 路径、10 个类别的中英文名、全部超参（所有脚本共用）
│  ├─ data.py           # CIFAR-10 下载/解析/缓存/预处理/增强（下载部分不依赖 TF）
│  ├─ alexnet.py        # 基础 AlexNet 定义，激活函数与规范化方式可切换
│  ├─ train.py          # 训练单组（含 --benchmark 测速、NaN 守卫、断点续训）
│  ├─ evaluate.py       # 测试集指标：accuracy / top-3 / 逐类 P-R-F1 / 混淆矩阵
│  ├─ predict.py        # ★ 输入一张图像 → 输出类别
│  ├─ compare.py        # 汇总四组，出对比表与对比图
│  ├─ test_pipeline.py  # 管线自检（74 项断言）
│  ├─ test_cli.py       # ★ 用合成数据跑通 train→evaluate→compare→predict 命令行链路
│  ├─ make_predict_samples.py  # 生成报告图 1-5（含跨数据集演示）
│  └─ (可选) train.txt 等中间产物不放这里，本实验不用
├─ models/              # 四组权重（<run>.h5 最终、<run>_best.h5 验证集最优）
├─ logs/                # 每组一个目录：TensorBoard events + checkpoint + history.csv
├─ outputs/
│  ├─ predict/          # predict.py --save-dir 的输出
│  ├─ eval/             # <run>_history.json、<run>_metrics.{json,md}、混淆矩阵
│  └─ compare/          # summary.{csv,md} 与三张对比图
├─ README.md
└─ requirements.txt
```

数据集放在 **`datasets/classification/cifar10/`**（由 `data.py` 自动下载，见下）。

### 脚本一览

| 脚本 | 作用 | 需要 TF? |
|---|---|:---:|
| `config.py` | 统一路径与超参，打印四组对比配置 | 否 |
| `data.py` | 下载/解析/缓存/划分/归一化/增强 | 否（`make_dataset` 除外）|
| `alexnet.py` | 模型定义与结构预览 | **是** |
| `train.py` | 训练一组 | **是** |
| `evaluate.py` | 测试集评测与混淆矩阵 | **是**（`--selftest` 否）|
| `predict.py` | 单张/批量推理 | **是**（`--selftest` 否）|
| `compare.py` | 四组汇总与出图 | 否（出图需 matplotlib）|
| `test_pipeline.py` | 全流程自检（74 项）| 部分 |
| `test_cli.py` | **命令行入口**冒烟测试（29 项，用合成数据）| **是** |
| `make_predict_samples.py` | 生成报告图 1-5「预测结果示例」，并做**跨数据集**演示（把实验一的真实猫照片喂给分类模型）| **是** |

> `data.py --selftest`、`evaluate.py --selftest`、`predict.py --selftest`、`compare.py --selftest`
> **都不需要 TensorFlow**，可以在只装了 numpy 的环境里先跑通逻辑。
>
> **`test_cli.py` 是单独一类**：`test_pipeline.py` 验证的是各模块的函数级正确性，
> 但不覆盖命令行入口（argparse 参数名、回调链、checkpoint 命名、history.json 字段、
> evaluate 读权重、compare 汇总）。这些只有在真正调用 `main()` 时才会暴露，
> 而真正训练要 3.5 小时——等那么久才发现"某个参数名写错了"代价太大。
> `test_cli.py` 把 `data.load_splits` 换成合成数据，几秒就能把整条链路走一遍，
> **不碰 `datasets/`**，只写 `smoketest-*` 开头的临时产物并在结束时自动清理。
>
> 它实际抓到过两个 bug：`evaluate.py` 在**未 compile** 的模型上调用 `model.evaluate()`
> （必然抛 `RuntimeError`，也就是说该脚本此前根本跑不起来）；
> 以及 TensorBoard 事件写在 `log_dir/train/` 子目录里、按 `log_dir/*` 找不到。

---

## 三、环境

**直接复用实验一（目标检测）已建好的环境**，两者依赖完全一致：

```powershell
$TF = ".\envs\pcd01_tf\Scripts\python.exe"   # 在仓库根目录执行
& $TF -c "import tensorflow as tf; print(tf.__version__)"   # 应为 2.10.0
```

若环境丢失，重建：

```powershell
python envs\setup_tf_env.py
```

**为什么是 CPU 训练**：本机 RTX 4060 Laptop 是 sm_89（Ada），需要 CUDA 11.8+；
而 TF 2.10 绑定 CUDA 11.2（最高支持 sm_86），TF ≥ 2.11 又取消了 Windows 原生 GPU 支持。
所以两个实验都只能跑 CPU，实测速度见第四节。

---

## 四、数据集：CIFAR-10

| 项目 | 数值 |
|---|---|
| 图像 | 32×32 彩色，10 类 |
| 训练集 | 50000 张 |
| 测试集 | 10000 张 |
| 下载 | 约 163 MB |
| 下载位置 | `datasets/classification/cifar10/` |

**类别**：airplane 飞机 / automobile 汽车 / bird 鸟 / **cat 猫** / deer 鹿 /
dog 狗 / frog 青蛙 / horse 马 / ship 船 / truck 卡车

**划分**：从 50000 张训练集里固定划出 **5000 张作验证集**（`seed=42`），
其余 45000 张训练；**测试集 10000 张全程不参与训练与选型**，只在最终评测用一次。

**预处理**：缩放到 32×32（数据集原生尺寸，不做缩放）、按通道归一化。
均值/标准差**只从训练集统计**——用全量数据统计会把测试集信息泄漏进训练流程。

**增强**（只用于训练集）：4 像素零填充 + 随机裁剪回 32×32 + 随机水平翻转。

### 下载

```powershell
& $TF experiments\01_TensorFlow_AlexNet\code\data.py --download
```

> **本机网络提示**：实测官方源 `cs.toronto.edu` 只有 **0.019 MB/s** 且每 8~9 秒停滞一次，
> 全量要 2 小时以上。因此 `data.py` 内置了：
> * **多镜像**（默认先试 Baidu BOS，实测约 0.060 MB/s，快 3 倍）
> * **断点续传（HTTP Range）+ 停滞自动重试**
> * **SHA256 校验**
>
> 中途断了直接重新运行同一条命令即可**从断点继续**，不会白下。
> 指定单个镜像：`--mirror Baidu`（或直接给 URL）。

---

## 五、模型：基础 AlexNet

保留 AlexNet 的**定义性拓扑**：5 个卷积 + 3 个全连接。

| 层 | 本实验配置 |
|---|---|
| 输入 | 32×32×3 |
| conv1 | 64 @ 3×3, s1 → MaxPool 3×3 s2 |
| conv2 | 192 @ 3×3, s1 → MaxPool 3×3 s2 |
| conv3 / conv4 / conv5 | 384 / 256 / 256 @ 3×3, s1 → MaxPool 2×2 |
| FC1 / FC2 | 512 / 512，各接 Dropout 0.5 |
| 输出 | Dense 10 + softmax |
| 规范化 | BatchNorm（可切 `--norm lrn` 回到原论文做法）|
| 正则 | L2 weight decay 5e-4（加在卷积核与全连接核上）|
| 参数量 | **4.62 M** |

### 与原论文的三处差异（报告里必须写明）

| 项 | 原论文 | 本实验 | 原因 |
|---|---|---|---|
| 输入尺寸 / 首层卷积 | 224×224 / 11×11 stride 4 | 32×32 / 3×3 stride 1 | CIFAR-10 是 32×32；11×11 在 32×32 上无法使用 |
| 全连接宽度 | 4096 / 4096 | 512 / 512 | 原尺寸下 FC 占绝大部分参数，是 CPU 上的主要瓶颈 |
| 规范化 | LRN | BatchNorm | BN 在 epoch 数有限时收敛更快，利于"15 epoch 内看出差异" |

> **weight decay 的实现方式**：用 **L2 正则**，而不是 `SGD(decay=...)`。
> 因为 Keras 的 `decay` 是按 `lr/(1+decay*iter)` 让学习率随迭代下降的，
> 那样"恒定学习率"这个前提就不成立了，会污染本实验对学习率的对比。

---

## 六、四组对比实验

| run_name | 激活函数 | 学习率 |
|---|---|---|
| `relu-lr0.01` | ReLU | 0.01 |
| `relu-lr0.001` | ReLU | 0.001 |
| `leakyrelu-lr0.01` | LeakyReLU | 0.01 |
| `leakyrelu-lr0.001` | LeakyReLU | 0.001 |

**公平性约束**（四组**只允许**这两项不同）：

- 同一随机种子 `seed=42`
- 同一 train/val 划分
- 同一 epoch 数、同一 batch size
- 同一数据增强策略
- **不启用 EarlyStopping**（否则各组实际训练轮数不同，对比失去意义）
- 优化器都是 SGD + momentum 0.9 + L2 weight decay 5e-4，**恒定学习率**

---

## 七、运行步骤

> 下面用 `$TF` 代表 TF 解释器：
> `.\envs\pcd01_tf\Scripts\python.exe`（在仓库根目录执行）

### 0. 先跑自检（不需要数据，也不需要等下载）

```powershell
& $TF experiments\01_TensorFlow_AlexNet\code\test_pipeline.py
```

不装 TF 的环境也能跑大部分：

```powershell
python experiments\01_TensorFlow_AlexNet\code\test_pipeline.py --no-tf
```

### 1. 检查环境与数据（不训练、不下载）

```powershell
& $TF experiments\01_TensorFlow_AlexNet\run_all.py --dry-run --tf-python $TF
```

### 2. 实测当前机器的训练速度

```powershell
& $TF experiments\01_TensorFlow_AlexNet\code\train.py --benchmark 5
```

### 3. 小样本冒烟验证（几分钟，确认链路通）

```powershell
& $TF experiments\01_TensorFlow_AlexNet\run_all.py --epochs 1 --subset 2000 --tf-python $TF
```

### 4. 正式跑四组（约 3.5 小时）

```powershell
& $TF experiments\01_TensorFlow_AlexNet\run_all.py --epochs 15 --tf-python $TF
```

也可以单独跑某一组：

```powershell
& $TF experiments\01_TensorFlow_AlexNet\code\train.py --act leaky_relu --lr 0.001 --epochs 15
& $TF experiments\01_TensorFlow_AlexNet\code\evaluate.py `
    --model experiments\01_TensorFlow_AlexNet\models\leakyrelu-lr0.001_best.h5 `
    --act leaky_relu --run-name leakyrelu-lr0.001 --report
```

### 5. 用训练好的模型做预测（题目要求的核心功能）

```powershell
# 单张
& $TF experiments\01_TensorFlow_AlexNet\code\predict.py --image 某张图.jpg

# 批量，并把结果画在图上保存
& $TF experiments\01_TensorFlow_AlexNet\code\predict.py `
    --dir datasets\classification\cifar10 --limit 10 `
    --save-dir experiments\01_TensorFlow_AlexNet\outputs\predict
```

输出形如：

```
  排名 1: cat（猫）        置信度 0.9231
  排名 2: dog（狗）        置信度 0.0512
  排名 3: horse（马）      置信度 0.0103
  => 类别：cat（猫），置信度 0.9231
```

---

## 八、训练速度（本机 CPU 实测）

| AlexNet 变体 | 参数量 | 每图耗时 | 一个 epoch（5 万张）|
|---|---:|---:|---:|
| 标准通道 96/256/384/384/256 | 4.8 M | 7.6 ms | 6.3 分钟 |
| **本实验采用：64/192/384/256/256** | **4.6 M** | **约 4~5 ms** | **约 3.5 分钟** |
| 轻量通道 + 无 BN | 3.7 M | 3.4 ms | 2.8 分钟 |

**四组合计（每组 15 epoch）**：

| 阶段 | 时间 |
|---|---:|
| CIFAR-10 下载（受本机网络影响很大）| 5 分钟 ~ 2 小时 |
| 训练 4 组 × 15 epoch | **约 3.5 小时** |
| 评测 + 汇总出图 | 约 10 分钟 |

> 训练前先跑 `--benchmark` 实测，按真实速度重算 ETA，不要照搬上表。

---

## 九、结果

四组各训 15 epoch，测试集 10000 张，总用时 **3.25 小时**（CPU）。

| 激活函数 | 学习率 | 最佳验证准确率 | 测试准确率 | top-3 | macro-F1 | 训练耗时 |
|---|---:|---:|---:|---:|---:|---:|
| **ReLU** | **0.01** | **0.8238** | **0.8185** | **0.9633** | **0.8162** | 0.82 h |
| ReLU | 0.001 | 0.7768 | 0.7768 | 0.9519 | 0.7771 | 0.77 h |
| LeakyReLU | 0.01 | 0.8098 | 0.8055 | 0.9582 | 0.7987 | 0.81 h |
| LeakyReLU | 0.001 | 0.7828 | 0.7706 | 0.9484 | 0.7699 | 0.79 h |

**分组平均**

| 自变量 | 水平 | 测试准确率均值 | 组内差距 |
|---|---|---:|---:|
| 激活函数 | ReLU | **0.7976** | 0.96 个百分点 |
| 激活函数 | LeakyReLU | 0.7880 | 同上 |
| 学习率 | 0.01 | **0.8120** | **3.83 个百分点** |
| 学习率 | 0.001 | 0.7737 | 同上 |

**最优先组合**：ReLU + lr = 0.01，测试准确率 **81.85%**。

**三条核心结论**（详细分析见[报告](../../reports/01b_AlexNet图像分类_实验报告.md)第六、七、九节）：

1. **激活函数没有显著差异**（0.96 个百分点，且方向随学习率翻转）。
   推测原因是 BN 已经解决了 ReLU "神经元死亡"的问题，LeakyReLU 想解决的事被 BN 做掉了；
2. **学习率影响显著**（3.83 个百分点，是激活函数的 4 倍），
   但 lr=0.001 到第 15 轮**损失仍在稳定下降**，属于"没训够"而非"更差"——
   固定 epoch 数对比学习率本身有偏；
3. **`cat` 是最难的类**（Recall 仅 0.5310，其余 9 类 0.74~0.92），
   18.0% 被误判为 dog、10.4% 为 deer，且与 dog 构成双向混淆。
   根因是 32×32 分辨率下猫狗的外观线索大量丢失。

**跨数据集验证**：把实验一（目标检测）`datasets/cat_face/` 里
**4 张未参与训练的真实猫照片**喂给模型，**3 张判为 `cat`** ——
说明模型学到的是猫的视觉特征，不只是拟合 CIFAR-10 的分布。

**产物**

| 内容 | 位置 |
|---|---|
| 四组权重 | `models/{relu,leakyrelu}-lr{0.01,0.001}{,_best}.h5` |
| 训练历史 | `outputs/eval/<run>_history.json` |
| 测试集指标 | `outputs/eval/<run>_metrics.{json,md}` |
| 混淆矩阵 | `outputs/eval/<run>_confusion.{csv,png}` |
| 对比表 | `outputs/compare/summary.{csv,md}` |
| 对比图 | `outputs/compare/curves_{val_acc,loss}.png`、`test_accuracy.png` |
| 预测示例 | `outputs/predict/samples/` 与 `reports/figures/01b_predict_samples.png` |
| 报告 | [`reports/01b_AlexNet图像分类_实验报告.md`](../../reports/01b_AlexNet图像分类_实验报告.md) |

---

## 十、常见问题

| 现象 | 原因与处理 |
|---|---|
| 下载很慢甚至失败 | 本机到境外站点只有 0.02 MB/s 量级。直接重跑同一条命令会**断点续传**；也可 `--mirror Baidu` |
| 下载完了但解压报 `invalid distance code` | 续传时 `resp.read()` 在超时抛异常前可能丢掉若干字节，拼出的文件**大小对、内容错**。已修复：续传时**重叠回退 1 MB**保证边界对齐，并把校验判据改成"能否解压"；重跑命令会自动定位损坏点、**只重下损坏之后的部分** |
| `UnicodeEncodeError: 'latin-1' codec can't encode ...` | 早期版本 `draw_result` 用 PIL 默认位图字体画中文。已修复：自动找系统中文字体，找不到就只画英文 |
| **图上的中文变成方框 `□□□`** | matplotlib 默认字体不含中文字形。**这是静默失败**：程序不报错、退出码 0、指标数字也对，只有 stderr 有 `Glyph xxxxx missing from current font` 警告，很容易漏掉。已修复：统一用 `config.setup_matplotlib()` 设置字体 |
| `norm='lrn'` 构图报错 | LRN 只能用于 4 维张量，早期版本误把它套到全连接层上。已修复：FC 层只用 BN |
| 某一组 loss 变成 NaN | 学习率过大导致发散。这**本身是学习率对比的有效结果**，报告里如实记录即可；`train.py` 会在保存后检测 NaN 并明确报警 |
| `--act leakyrelu` 报错 | argparse 只认 `leaky_relu`（带下划线）。`run_name` 里才是 `leakyrelu` |
| 训练中断想接着跑 | `train.py --resume models\<run>.h5 --initial-epoch N`，续训前会校验权重可读且无 NaN |

---

## 十一、一键流程

```powershell
$TF = ".\envs\pcd01_tf\Scripts\python.exe"   # 在仓库根目录执行

# 只检查
& $TF experiments\01_TensorFlow_AlexNet\run_all.py --dry-run --tf-python $TF

# 冒烟验证
& $TF experiments\01_TensorFlow_AlexNet\run_all.py --epochs 1 --subset 2000 --tf-python $TF

# 正式跑（约 3.5 小时）
& $TF experiments\01_TensorFlow_AlexNet\run_all.py --epochs 15 --tf-python $TF
```
