# tools —— 公共小工具

## `check_datasets.py` —— 数据集自检

检查 4 个实验的数据是否放好、命名是否合规。**只依赖 Python 标准库**，所以在还没装任何第三方库时就能跑。

```bash
python tools/check_datasets.py                          # 检查全部
python tools/check_datasets.py --only cat_face          # 只查一个
python tools/check_datasets.py --only TVSum,retrieval   # 查多个
python tools/check_datasets.py --root datasets          # 指定 datasets 目录
```

**退出码**：`0` = 没有错误；`1` = 存在错误（可直接用于 CI）。

### 检查内容

| 数据集 | 检查项 |
|---|---|
| `classification` | CIFAR-10 是否就绪；7 个批次文件是否齐全；**压缩包大小是否等于官方值 170498071 字节**；若正在下载则报告**断点续传的进度百分比**；是否有 `cifar10.npz` 缓存 |
| `cat_face` | 目录是否齐全；图片数；图片与标注**是否同名配对**；文件名**是否含中文**；`ImageSets/Main` 四个索引文件；**XML 里的标签名是否统一**（应为 `cat`）；图片数是否 <230 |
| `tracking_videos` | 视频列表与大小；真值文件数量；**自动识别真值格式**（OTB / VOT / MOT / 旋转框）；视频与真值是否同名配对；是否完全没有真值（会导致算不出精度指标） |
| `TVSum` | 视频数量（应为 50）；两个标注 tsv 是否存在；解析 `ydata-tvsum50-anno.tsv` 表头、覆盖视频数、每条视频的标注者数量；**视频文件名与标注 `video_id` 是否对得上**；特征文件；是否已指定 10 段测试视频 |
| `retrieval` | 图片/视频/查询图数量与规模建议；类别目录结构；`queries/relevance.csv`；**查询图是否与底库图同名**（可能是同一张，会致指标虚高） |

> `classification` 这一项**不做 SHA256 全量校验**（162 MB 要十几秒，日常检查没必要）；
> 真正下载时 `data.py` 会校验。

### Windows 中文乱码

脚本已强制以 UTF-8 输出。若终端仍显示乱码，先切换代码页：

```powershell
chcp 65001
```

---

## labelImg 相关工具

实验一（目标检测）用 LabelImg 人工标注，**labelImg 1.8.3 与新版 PyQt5 有 8 处类型不兼容**，
会在画框/滚动/缩放时崩溃。这几个脚本用来修补、验证与启动。

```powershell
$PY = "python"   # 换成装了 labelImg 的解释器

& $PY tools\check_labelimg_env.py          # 检查 PyQt5 与补丁状态
& $PY tools\patch_labelimg.py              # 打 8 处 float->int 补丁（幂等）
& $PY tools\patch_labelimg.py --check      # 只看当前状态，不改文件
& $PY tools\patch_labelimg.py --revert     # 从 .bak 还原
& $PY tools\verify_labelimg_paint.py       # 离屏验证绘制路径不再崩（不用开图形界面）
```

| 文件 | 作用 |
|---|---|
| `patch_labelimg.py` | 修正 8 处 float→int 类型不兼容（`labelImg.py` 4 处、`libs/canvas.py` 3 处、`libs/shape.py` 1 处）。**幂等**，改前自动备份 `.bak` |
| `check_labelimg_env.py` | 无界面探测 PyQt5 是否接受 float，并结合补丁状态给出结论 |
| `verify_labelimg_paint.py` | 在 offscreen 模式下**真实执行** `Canvas.paintEvent` 与 `Shape.paint`，确认不崩 |
| `start_labelimg.cmd` | 双击启动 LabelImg（图片目录、预设类别、保存目录都已配好） |
| `start_labelimg_remaining.cmd` | 只打开"还没人工改过"的那些图，用于集中补标 |

> **为什么不是降级 PyQt5**：实测 PyQt5 5.15.7 与 5.15.11 **都**拒绝 float，
> 所以这不是版本问题，而是 labelImg 自身的类型 bug（`canvas.py` 里
> `self.prevPoint = QPointF()`，而 `QPointF.x()` 返回 float）。
> 只能改调用点。

---

## `dump_docx.py` —— DOCX 正文抽取

把 `.docx` 的段落（带样式名）和表格按顺序输出为纯文本，便于全文检索与引用。

```bash
python tools/dump_docx.py "docs/course_materials/目标检测实验指导书.docx" > docs/extracted_text/目标检测实验指导书.txt
```

需要 `python-docx`。用途：`docs/course_materials/` 下的指导书正文在 Word 里不好搜，
转成文本后可以直接 `grep`。

> 注意：**图片里的信息（如安装步骤截图）不会被抽取**，关键信息仍需回看原始 docx。
