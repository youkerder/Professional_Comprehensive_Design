# 实验 02 · 目标跟踪：OpenCV 多目标跟踪（已完成）

> 题目来源：《实验题目-4.docx》实验三「目标跟踪实验」+《目标跟踪实验指导书》+《使用 OpenCV 进行目标跟踪》
> 实验报告：[reports/02_目标跟踪_实验报告.md](../../reports/02_目标跟踪_实验报告.md)（按实验模板结构改写）

---

## 一、任务拆解与完成情况

| # | 题目要求 | 本实验的做法 |
|:---:|---|---|
| 1 | 基于 OpenCV 内置算法，**≥3 种**算法对**多目标**跟踪 | 6 种：KCF、CSRT、MedianFlow、MIL、Boosting、TLD；每帧 3 个目标同时跟踪（`cv2.legacy.MultiTracker`），另做一次「三个目标各用不同算法」的混用实验 |
| 2 | **比较跟踪效率** | 单帧 `update()` 耗时（均值 / P95）、等效 FPS、`init` 耗时；13 次运行全部记录在 `outputs/metrics/summary.csv` |
| 3 | 输入视频，**框出多目标运动轨迹** | 逐帧画框 + 淡出轨迹尾迹并写 MP4；轨迹点写 CSV；另有按目标着色的轨迹对比图（真值 vs 预测） |
| 4 | **≥2 种评价指标**评价性能、准确率、鲁棒性 | OTB（Success AUC、OP@0.5/0.75、P@20、中心误差）、MOTChallenge（MOTA/MOTP/IDSW/Frag/MT/ML）、VOT 轻量口径（accuracy/robustness），另加「按可见比例分档」的遮挡鲁棒性 |

---

## 二、环境

**不要用系统默认 Python 3.14，也不要用 `C:\anaconda3`**：那里的 OpenCV 是 5.0.0 基础版，没有 `cv2.legacy` 命名空间，
也就没有 `MultiTracker_create` 与经典跟踪器。

| 项目 | 版本 |
|---|---|
| Python | 3.11.9（独立 venv：`envs/pcd02_tracking`） |
| OpenCV | `opencv-contrib-python==4.8.1.78`（含 `cv2.legacy`） |
| 其他 | numpy 1.26.4、matplotlib 3.8.4 |

创建方式（在仓库根目录执行，`--without-pip` 是为了绕开本机 venv 引导阶段的 `WinError 5`）：

```bat
python ^
    -m venv --without-pip --copies envs\pcd02_tracking
envs\pcd02_tracking\Scripts\python.exe -m ensurepip --upgrade
envs\pcd02_tracking\Scripts\python.exe -m pip install --no-cache-dir -r experiments\02_ObjectTracking_OpenCV\requirements.txt
```

为什么锁 4.8.1：OpenCV ≥ 4.5.2 把跟踪器挪进 `cv2.legacy`（代码里已做兼容查找），而 5.0.0 基础包直接不再提供；
`opencv-python` 与 `opencv-contrib-python` **不能共存**，装之前先卸掉 `opencv-python`。

---

## 三、数据集（自建合成多目标序列）

`datasets/tracking_videos/`：两段 640×360、25 fps、300 帧（12 s）的视频，带精确真值。

| 序列 | 内容 | 考察点 |
|---|---|---|
| `cross` | 3 个带纹理目标：左上→右下、右下→左上（中段交叉）、绕中心小圆运动并放大 | 交叉、尺度变化、多目标同屏 |
| `occlude` | 静态遮挡条 `(280,0,90,250)`：target1 完全不可见 28 帧，target2 约 50% 可见 100 帧，target3 不受影响 | 三档可见性下的遮挡鲁棒性 |

真值三份并存，供不同指标口径直接读取：

```
datasets/tracking_videos/
├─ videos/{cross,occlude}.mp4
└─ annotations/<seq>/
   ├─ otb/<target>.txt      # 每行 x,y,w,h（OTB 口径）
   ├─ gt.txt                # frame,id,x,y,w,h,conf,class,visibility（MOTChallenge 口径）
   ├─ first_boxes.json      # 首帧初始框
   └─ seqinfo.ini, meta.json
```

`visibility` 由遮挡掩膜精确统计可见像素比例得到，是遮挡分档的依据。目标刻意做成「带内部高对比纹理」的
sprite——纯色块会让 MedianFlow 无法初始化、CSRT 的尺度崩到 7×7，那样比较的是退化行为而不是算法本身。

重新生成：`python code/make_synthetic_video.py`（固定随机种子，结果可复现）。

---

## 四、代码结构（10 个脚本，约 2270 行）

```
experiments/02_ObjectTracking_OpenCV/
├─ code/
│  ├─ common.py               # 路径常量、视频/真值读写、IoU 与中心误差、匈牙利匹配、轨迹 CSV
│  ├─ trackers.py             # 7 种跟踪器的工厂（cv2.legacy 兼容查找）与可用性检查
│  ├─ viz.py                  # 画框、虚线丢失框、轨迹尾迹、信息面板
│  ├─ make_synthetic_video.py # 造视频 + 三种格式真值 + 首帧预览图
│  ├─ single_target_demo.py   # 单目标 demo（selectROI 或真值首帧框）
│  ├─ run_multitracker.py     # 多目标：MultiTracker 后端 / 逐目标独立后端，写视频+轨迹+运行时 JSON
│  ├─ evaluate.py             # OTB / MOTChallenge / VOT 三套指标 + 可见比例分档 → summary.csv
│  ├─ make_figures.py         # 14 张图 + 8 张表
│  ├─ make_report.py          # 生成报告 docx（python-docx，不依赖 Word）
│  ├─ run_all.py              # 一键全流程
│  └─ tools/check_trackers.py # 跟踪器可用性自检（合成 patch + 真实序列两段对照）
├─ outputs/{videos,traces,metrics,figures,tables}/
├─ README.md
└─ requirements.txt
```

---

## 五、运行

先自检（确认本机 OpenCV 里哪些跟踪器真的能用），再跑全流程：

```bat
cd experiments\02_ObjectTracking_OpenCV\code

:: 1) 跟踪器可用性自检（约 10 秒）
..\..\..\envs\pcd02_tracking\Scripts\python.exe tools\check_trackers.py

:: 2) 单目标 demo：KCF / CSRT / MEDIANFLOW 各跑一次
..\..\..\envs\pcd02_tracking\Scripts\python.exe single_target_demo.py --seq cross --algo KCF --target target1

:: 3) 多目标：三种算法同时跟踪 3 个目标（backend=multi 用 cv2.legacy.MultiTracker）
..\..\..\envs\pcd02_tracking\Scripts\python.exe run_multitracker.py --seq cross --algos KCF CSRT MEDIANFLOW

:: 4) 一键全流程（造视频 → demo → 多目标 → 评价 → 出图，约 8 分钟）
..\..\..\envs\pcd02_tracking\Scripts\python.exe run_all.py
..\..\..\envs\pcd02_tracking\Scripts\python.exe run_all.py --fast     :: 只跑主算法（KCF/CSRT/MedianFlow）

:: 5) 重新生成报告（需要 python-docx）
python make_report.py
```

常用参数：`--seq cross|occlude`、`--algos KCF CSRT MEDIANFLOW`、`--backend multi|single`、
`--min`、`--mixed`（同一 MultiTracker 内每目标不同算法）、`--select`（用 `cv2.selectROI` 手动画框）。

---

## 六、产物

| 位置 | 内容 |
|---|---|
| `outputs/videos/` | 13 个结果视频：`mt_*` 多目标、`single_*` 单目标 demo（带框 + 轨迹） |
| `outputs/traces/` | 逐帧轨迹 CSV：`frame,target,x,y,w,h,ok,ms,cx,cy` |
| `outputs/metrics/` | `run_mt_*.json`（运行时）、`eval_*.json`（指标）、`summary.csv`（总表） |
| `outputs/figures/` | 14 张图：success/precision 曲线、逐帧 IoU 时间线、轨迹对比、速度-精度、遮挡分档、定性拼图 |
| `outputs/tables/` | 8 张表：速度、OTB 精度、MOT、遮挡分档（csv + md） |
| `reports/figures/02_*.png`、`reports/tables/02_*.csv` | 报告引用的副本 |
| `reports/02_目标跟踪_实验报告.md` | 实验报告（含 7 张图、9 张表） |

---

## 七、关键结果（本机实测，详见报告表 3-4 ~ 3-8）

三种目标同时跟踪、300 帧的平均单帧耗时与精度：

| 算法 | 耗时 ms（cross） | FPS | mean IoU（cross） | mean IoU（occlude） |
|---|---:|---:|---:|---:|
| MEDIANFLOW | 5.08 | 196.7 | 0.794 | 0.792 |
| KCF | 10.81 | 92.5 | 0.677 | 0.766 |
| CSRT | 81.57 | 12.3 | 0.771 | 0.704 |
| BOOSTING | 72.60 | 13.8 | 0.843 | 0.772 |
| TLD | 97.46 | 10.3 | 0.664 | 0.608 |
| MIL | 290.95 | 3.4 | 0.668 | 0.652 |

- **速度**：MedianFlow ≈ 197 FPS、KCF ≈ 92 FPS 可实时；CSRT/Boosting/TLD 只有 10~14 FPS；MIL 3.4 FPS。
- **精度**：cross 上 Boosting（AUC 0.843）与 CSRT（0.772）最好；KCF 因不更新尺度，target3 只有 0.294。
- **遮挡**：部分遮挡时 CSRT 保持率最高（94%）、MedianFlow 91%、TLD 最低（59%）；完全遮挡 28 帧里全部失效，
  且遮挡结束后**无一恢复**——短时跟踪器没有重检测能力。

---

## 八、踩坑记录

1. **MOSSE 在 `opencv-contrib-python==4.8.1.78` 上不可用**：`init` 返回 True，但之后每帧 `update` 都返回 False
   且框被置零 `(0,0,0,0)`。同一条序列上其它 6 种算法全部正常，干净 patch 上 MOSSE 也 39/39 成功 →
   不是调用错误，而是 PSR 判据在杂乱背景上把每帧判为不可信，而 OpenCV 判失败时清零框。
   本版也没有 `set()` 可以调门限。**正式对比排除了 MOSSE**，把它的失败过程写进报告的「问题与解决」。
2. **MultiTracker 的 `ok` 是全员与运算**：单目标丢失会把整帧标记为失败，无法逐目标评估。因此额外实现了
   `--backend single`（每目标一个实例，逐目标拿 ok），两种后端输出格式一致。
3. **`init` 首次调用有预热假象**：第一轮实测 KCF init 1482 ms，预热后只有 79 ms。正式计时前统一做一次预热。
4. **中文乱码**：Windows 控制台需 `setup_console()` 切 UTF-8；matplotlib 需按序探测
   `Microsoft YaHei / SimHei / Noto Sans CJK`，否则中文变方框。
5. **`pip list` 在 Git Bash 下会卡住**：改用 `importlib.metadata` 读版本；Git Bash 里 `../../../envs/...` 这种相对
   路径有时解析失败，用绝对路径 `/d/projects/.../python.exe`。
6. **交付视频用 mp4v 写**：`VideoWriter` 写 MP4 必须指定 `mp4v`，否则静默写出 0 字节文件。
