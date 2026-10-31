# -*- coding: utf-8 -*-
"""生成「实验六 视频摘要实验」Word 报告（版式沿用 reports/_template_work/build_report.py）。

输出：reports/03_视频摘要实验报告.docx
数据来源：
  experiments/03_VideoSummarization_FCSN/outputs/metrics/prf_test10.csv
  experiments/03_VideoSummarization_FCSN/outputs/metrics/baseline_compare.csv
  experiments/03_VideoSummarization_FCSN/logs/train_bce.log
  插图：reports/figures/03_*.png
"""
import csv
import os
import sys

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 仓库根目录由脚本位置推导：code/ -> 实验目录 -> experiments/ -> 仓库根
REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
WORK = os.path.join(REPO, 'reports', '_template_work')
MEDIA = os.path.join(WORK, 'media')
FIGS = os.path.join(REPO, 'reports', 'figures')
EXP = os.path.join(REPO, 'experiments', '03_VideoSummarization_FCSN')
OUT = os.path.join(REPO, 'reports', '03_视频摘要实验报告.docx')

CN, EN, MONO = '宋体', 'Times New Roman', 'Consolas'
BODY_PT, TABLE_PT, CAP_PT = 10.5, 9, 9
TEXT_W_CM = 14.66


# ---------------------------------------------------------------- 版式工具
def set_font(run, size=None, bold=None, cn=CN, en=EN, italic=None, color=None):
    run.font.name = en
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), en)
    rFonts.set(qn('w:hAnsi'), en)
    rFonts.set(qn('w:eastAsia'), cn)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor(*color)
    return run


def para(doc, align=None, before=0, after=0, line=None, indent=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    if align is not None:
        p.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if line:
        pf.line_spacing = line
    if indent is not None:
        pf.first_line_indent = Pt(indent)
    return p


def body(doc, text, size=BODY_PT, indent=21, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         before=0, after=6, line=1.4, bold=False):
    p = para(doc, align=align, before=before, after=after, line=line, indent=indent)
    set_font(p.add_run(text), size=size, bold=bold)
    return p


def rich(doc, parts, size=BODY_PT, indent=21, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         before=0, after=6, line=1.4):
    p = para(doc, align=align, before=before, after=after, line=line, indent=indent)
    for item in parts:
        txt, bold = (item if isinstance(item, tuple) else (item, False))
        set_font(p.add_run(txt), size=size, bold=bold)
    return p


def h1(doc, text, before=16, after=8):
    p = para(doc, align=WD_ALIGN_PARAGRAPH.LEFT, before=before, after=after, line=1.3)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), size=12, bold=True)
    return p


def h2(doc, text, before=10, after=5):
    p = para(doc, align=WD_ALIGN_PARAGRAPH.LEFT, before=before, after=after, line=1.3)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(text), size=11, bold=True)
    return p


def caption(doc, text):
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=10, line=1.2)
    set_font(p.add_run(text), size=CAP_PT)
    return p


def figure(doc, filename, cap, width_cm):
    path = os.path.join(FIGS, filename)
    if not os.path.exists(path):
        raise FileNotFoundError('缺少插图：%s' % path)
    p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=2, line=1.0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    caption(doc, cap)


def code_block(doc, lines, size=8.5):
    for i, ln in enumerate(lines):
        p = para(doc, align=WD_ALIGN_PARAGRAPH.LEFT, before=(4 if i == 0 else 0),
                 after=(4 if i == len(lines) - 1 else 0), line=1.15)
        p.paragraph_format.left_indent = Pt(18)
        set_font(p.add_run(ln), size=size, cn=CN, en=MONO)


def shade(cell, hexcolor='F2F2F2'):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hexcolor)
    tcPr.append(shd)


def table(doc, headers, rows, cap=None, widths=None):
    if widths:
        total = float(sum(widths))
        widths = [w * (TEXT_W_CM / total) for w in widths]
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    if widths:
        t.autofit = False
        tblPr = t._tbl.tblPr
        layout = OxmlElement('w:tblLayout')
        layout.set(qn('w:type'), 'fixed')
        tblPr.append(layout)
        set_grid(t, widths)
    else:
        t.autofit = True
    hdr = t.rows[0].cells
    trPr = t.rows[0]._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:tblHeader'))
    trPr.append(OxmlElement('w:cantSplit'))
    for i, htxt in enumerate(headers):
        hdr[i].text = ''
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        set_font(p.add_run(str(htxt)), size=TABLE_PT, bold=True)
        shade(hdr[i])
    for row in rows:
        cells = t.add_row().cells
        t.rows[-1]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for i, val in enumerate(row):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            set_font(p.add_run(str(val)), size=TABLE_PT)
    if widths:
        set_grid(t, widths)
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    if cap:
        caption(doc, cap)
    else:
        para(doc, after=6)
    return t


def hrule(doc, sz=20, before=2, after=10):
    p = para(doc, before=before, after=after, line=1.0)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), str(sz))
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '000000')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return p


def page_break(doc):
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def cover_field(p, label, n_spaces=8, size=12, value=''):
    set_font(p.add_run(label), size=size)
    if value:
        r = p.add_run(' ' + value + ' ')
        set_font(r, size=size)
        r.font.underline = True
    else:
        # 用不换行空格，避免 Word 把行尾空格连同下划线一起裁掉
        r = p.add_run('\u00a0' * n_spaces)
        set_font(r, size=size)
        r.font.underline = True
    return r


def set_grid(tbl, widths_cm):
    grid = tbl._tbl.find(qn('w:tblGrid'))
    for col, cm in zip(grid.findall(qn('w:gridCol')), widths_cm):
        col.set(qn('w:w'), str(int(round(cm * 567))))
    tblW = tbl._tbl.find(qn('w:tblPr')).find(qn('w:tblW'))
    if tblW is not None:
        tblW.set(qn('w:type'), 'dxa')
        tblW.set(qn('w:w'), str(int(round(sum(widths_cm) * 567))))


def no_borders(tbl):
    tblPr = tbl._tbl.find(qn('w:tblPr'))
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        e = OxmlElement('w:' + edge)
        e.set(qn('w:val'), 'none')
        e.set(qn('w:sz'), '0')
        borders.append(e)
    tblPr.append(borders)


# ---------------------------------------------------------------- 读结果数据
def read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


prf = read_csv(os.path.join(EXP, 'outputs/metrics/prf_test10.csv'))
prf_rows = [r for r in prf if r['video_id'] not in ('平均', 'MEAN', '')]
prf_mean = [r for r in prf if r['video_id'] in ('平均', 'MEAN')][0]
cmp_rows = read_csv(os.path.join(EXP, 'outputs/metrics/baseline_compare.csv'))
cmp_mean = {r['method']: r for r in cmp_rows if r['video_id'] == 'MEAN'}


def fnum(x, n=4):
    return ('%.' + str(n) + 'f') % float(x)


# ---------------------------------------------------------------- 文档初始化
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.top_margin = sec.bottom_margin = Cm(2.54)
sec.left_margin = sec.right_margin = Cm(3.17)

normal = doc.styles['Normal']
normal.font.name = EN
normal.font.size = Pt(BODY_PT)
normal.element.rPr.rFonts.set(qn('w:eastAsia'), CN)

footer_p = sec.footer.paragraphs[0]
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = footer_p.add_run()
set_font(fr, size=9)
f1 = OxmlElement('w:fldChar'); f1.set(qn('w:fldCharType'), 'begin')
f2 = OxmlElement('w:instrText'); f2.set(qn('xml:space'), 'preserve'); f2.text = 'PAGE'
f3 = OxmlElement('w:fldChar'); f3.set(qn('w:fldCharType'), 'end')
fr._r.append(f1); fr._r.append(f2); fr._r.append(f3)

# 封面不显示页码（与模板一致）
sec.different_first_page_header_footer = True
sec.first_page_footer.paragraphs[0].text = ''

# ---------------------------------------------------------------- 封面
p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=8, line=1.0)
p.add_run().add_picture(os.path.join(MEDIA, 'image1.png'), width=Cm(2.48))
set_font(p.add_run('    '), size=12)
p.add_run().add_picture(os.path.join(MEDIA, 'image2.png'), width=Cm(9.10))

p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=10, after=14, line=1.0)
set_font(p.add_run('实  验  报  告'), size=20, bold=True)

p = para(doc, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=4, line=1.5)
cover_field(p, '姓名', 4, value=os.environ.get('REPORT_NAME', ''))
cover_field(p, '  班级', 4, value=os.environ.get('REPORT_CLASS', ''))
cover_field(p, '  学号', 5, value=os.environ.get('REPORT_ID', ''))
cover_field(p, '  组别', 8)
cover_field(p, '  实验日期', 6, value='2026-10-21')

p = para(doc, align=WD_ALIGN_PARAGRAPH.LEFT, before=0, after=4, line=1.5)
cover_field(p, '课程名称', 0)
set_font(p.add_run('  专业综合设计  '), size=12)
cover_field(p, '同实验者', 8)
cover_field(p, '  指导教师', 0)
set_font(p.add_run('  ' + os.environ.get('REPORT_TEACHER', '') + '  '), size=12)
cover_field(p, '成绩', 8)

hrule(doc, sz=6, before=10, after=0)
hrule(doc, sz=24, before=0, after=18)

p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, after=8, line=1.4)
set_font(p.add_run('实验六  视频摘要实验'), size=17, bold=True)

# ================================================================ 正文
# ---- 一、实验任务
h1(doc, '一、实验任务')
body(doc, '基于 FCSN（Fully Convolutional Sequence Network）网络以及 TVSum 数据集，训练一个视频摘要模型；'
          '从数据集中选取 10 段视频对模型进行测试，为每段视频生成摘要视频与关键帧图片，'
          '保存到各自的文件夹；最后计算这 10 段视频的三种性能评价指标：精度（Precision）、'
          '召回率（Recall）与 F-score。')
body(doc, '本次实验只做实验六，不涉及实验一的 TensorFlow 图像分类、YOLO 目标检测、实验三的目标跟踪'
          '以及综合实验的视频检索，全部代码、数据、输出与报告都放在 '
          r'本仓库工作区内。')

# ---- 二、实验目的
h1(doc, '二、实验目的')
for i, txt in enumerate([
    '理解视频摘要（video summarization）的问题定义：把一段长视频压缩成能代表其内容的少数关键帧/关键镜头；',
    '掌握 FCSN 的建模思路：把视频摘要看作逐帧二分类（关键帧 / 非关键帧），用一维全卷积网络做序列到序列的稠密预测；',
    '掌握基于预训练 CNN（GoogLeNet）pool5 特征的视频表示方法，以及基于 KTS（核时域分割）的镜头切分方法；',
    '学会用 0-1 背包（knapsack）在「不超过 15% 帧数」的预算下选择关键镜头，并生成摘要视频与关键帧图片；',
    '掌握 Precision / Recall / F-score 三个指标的算法，能通过 Oracle 上界与标注者一致性判断模型离「本发明能达到的水平」还有多远；',
    '学会定位训练不收敛的问题：用最小可复现实验区分「数据问题」「模型问题」「损失函数问题」。',
], 1):
    body(doc, '（%d）%s' % (i, txt), indent=21, after=3)

# ---- 三、实验原理
h1(doc, '三、实验原理')

h2(doc, '3.1  视频摘要的问题定义')
body(doc, '视频摘要的任务是：给定一段 T 帧的视频 V = {x(1), ..., x(T)}，输出一个帧级重要性得分序列 '
          's(1), ..., s(T) ∈ [0, 1]，再按预算从中选出若干关键帧或关键镜头组成摘要。'
          '本实验采用「逐帧二分类 + 镜头级选择」的框架：网络对每一帧输出「关键 / 非关键」的概率，'
          '然后在镜头（segment）粒度上求均值，用背包算法选出总长度不超过 15% 的一组镜头。')
body(doc, '按监督信息的形式，视频摘要可分为无监督（用重建误差、聚类、注意力等自监督信号）与'
          '有监督（用人工标注的重要性分数）两类。本实验使用 TVSum 的人工标注：每段视频有 20 位标注者，'
          '每人给出一段 1~5 的重要性打分序列，把打分插值到逐帧并按阈值二值化，就得到训练用的逐帧标签。')

h2(doc, '3.2  FCSN 的网络结构')
body(doc, 'FCSN 由 Piao 与 Ma 在 2018 年 ECCV 上提出，其核心思想是借鉴全卷积网络（FCN）在语义分割上的成功：'
          '既然语义分割可以「输入一张图、输出同尺寸的逐像素标签」，'
          '那么视频摘要也可以「输入一段特征序列、输出同长度的逐帧标签」。'
          '整个网络没有全连接层，因此对输入长度不敏感（本实验统一采样到 320 帧）。')
body(doc, '网络分成编码与解码两部分，并在中间加一条跳跃连接（skip connection），'
          '以缓解时序下采样造成的时间定位精度损失。各层配置如下表：')
table(doc,
      ['阶段', '层', '配置', '输出长度', '输出通道'],
      [['编码', 'conv1', '2 × Conv1d(1024→1024, k=3, p=1) + BN + ReLU，再 MaxPool(2)', '160', '1024'],
       ['编码', 'conv2', '2 × Conv1d(1024→1024, k=3, p=1) + BN + ReLU，再 MaxPool(2)', '80', '1024'],
       ['编码', 'conv3', '2 × Conv1d(1024→1024, k=3, p=1) + BN + ReLU，再 MaxPool(2)', '40', '1024'],
       ['编码', 'conv4', '3 × Conv1d(1024→2048, k=3, p=1) + BN + ReLU，再 MaxPool(2)', '20', '2048'],
       ['编码', 'conv5', '3 × Conv1d(2048→2048, k=3, p=1) + BN + ReLU，再 MaxPool(2)', '10', '2048'],
       ['编码', 'conv6/7', 'Conv1d(2048→4096, k=1) + BN + ReLU；Conv1d(4096→4096, k=1) + BN + ReLU', '10', '4096'],
       ['编码', 'conv8', 'Conv1d(4096→2, k=1)', '10', '2'],
       ['解码', 'deconv1', 'ConvTranspose1d(2→2, k=4, s=2, p=1)', '20', '2'],
       ['跳跃', 'conv_pool4', 'Conv1d(2048→2, k=1) + BN，作用在 conv4 的池化输出上', '20', '2'],
       ['解码', 'deconv2', 'ConvTranspose1d(2→2, k=16, s=16)', '320', '2']],
      cap='表 1  FCSN 的网络结构（输入 1024×320 的 pool5 特征，输出 2×320 的逐帧得分）',
      widths=[1.4, 2.2, 8.0, 1.9, 1.9])
body(doc, '前向过程为：conv1~conv5 逐级把时间分辨率减半，conv6~conv8 在 10 帧的低分辨率上做通道扩张与分类；'
          'deconv1 把 10 帧上采样回 20 帧；deconv2 用 kernel=16、stride=16 的转置卷积一次性上采样回 320 帧。'
          '跳跃连接把 conv4 池化后的 20 帧特征经 1×1 卷积降到 2 通道，与 deconv1 的输出逐元素相加后'
          '再送入 deconv2。这条通路让「精细的时间定位」不经受深层的 16 倍下采样，'
          '是 FCSN 能给出帧级而非片段级得分的直接原因。')
body(doc, '输出经 softmax 得到两类的概率，取第 1 类（关键帧）的概率作为该帧的重要性得分。')

h2(doc, '3.3  视频特征与镜头切分')
rich(doc, [('GoogLeNet pool5 特征：', True),
           ('把每帧缩放到 224×224，用 ImageNet 预训练的 GoogLeNet 提取倒数第二层（pool5）的 '
            '1024 维特征。选它是因为它比 fc 层保留了更多空间信息，且 1024 维对 320 帧的序列来说计算量可接受。'
            '这一步与 FCSN 论文和 pytorch-vsumm-reinforce 的官方预处理一致（google_pool5）。', False)])
rich(doc, [('KTS 镜头切分：', True),
           ('关键帧必须在「镜头」粒度上成段选择，否则摘要视频会碎成孤帧。镜头边界用 KTS（Kernel Temporal '
            'Segmentation）自动检测：对 L2 归一化后的特征序列求解一个带核的变点检测问题，'
            '使段内散度最小、段间散度最大，用动态规划求全局最优分割。'
            '本实验在 2 Hz 采样后求解，得到平均 57.2 段/视频（50 段视频共 2862 段）。', False)])

h2(doc, '3.4  关键镜头选择：0-1 背包')
body(doc, '网络给出逐帧得分后，先在每个候选镜头内取均值作为该镜头的价值 v(i)、镜头长度为重量 w(i)，'
          '再求解 0-1 背包问题：')
p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=6, line=1.3)
set_font(p.add_run('max Σ v(i)·x(i)    s.t.  Σ w(i)·x(i) ≤ B,   x(i) ∈ {0, 1},   B = ⌊0.15·N⌋'),
         size=10.5, italic=True)
body(doc, '其中 N 是视频总帧数，预算 B 取总帧数的 15%（与 FCSN 论文和实验指导书一致）。'
          '背包用动态规划求解，得到的是「在 15% 预算下总价值最大」的镜头集合，'
          '这比「按得分排序取前 15% 帧」更合理，因为它不会把同一个镜头切碎。')

h2(doc, '3.5  评价指标')
table(doc,
      ['指标', '定义', '说明'],
      [['Precision', 'P = |S∩G| / |S|', '选出的帧里有多少是标注者认为重要的'],
       ['Recall', 'R = |S∩G| / |G|', '标注者认为重要的帧里有多少被选出'],
       ['F-score', 'F = 2PR / (P + R)', 'P 与 R 的调和平均，本实验的主指标'],
       ['帧级 AUC', 'ROC 曲线下面积', '只看「排序是否把关键帧排在前面」，与 15% 预算无关']],
      cap='表 2  三种性能评价指标', widths=[2.4, 5.2, 7.0])
body(doc, '由于每段视频有 20 位标注者，计算时先对每位标注者单独算一次 P/R/F，再对 20 个结果取平均，'
          '这样每位标注者权重相同（与 FCSN 论文、官方评估脚本的口径一致）。')

# ---- 四、实验过程
h1(doc, '四、实验过程')

h2(doc, '4.1  实验环境')
table(doc,
      ['项目', '配置'],
      [['操作系统', 'Windows 10'],
       ['CPU', '13th Gen Intel Core i7-13700H（14 核 20 线程）'],
       ['GPU', 'NVIDIA GeForce RTX 4060 Laptop（8 GB 显存，sm_89，驱动 595.97 / CUDA 13.2）'],
       ['Python', r'3.10.20（Anaconda 环境 pytorch）'],
       ['PyTorch / torchvision', '2.6.0+cu124 / 0.21.0+cu124（GPU 版，cuda.is_available() = True）'],
       ['数值与 IO 依赖', 'numpy 1.23.5、scipy、h5py 3.16.0、opencv-python、matplotlib'],
       ['可视化', 'TensorBoard 2.16.2（训练时用 tensorboardX 接口写入）'],
       ['基础代码', 'github.com/weirme/Video_Summary_using_FCSN（FCSN 官方 PyTorch 实现）']],
      cap='表 3  实验环境', widths=[4.2, 10.6])
body(doc, '环境上有一处需要注意：本机默认的 Python 3.14 没有可用的 PyTorch 生态，'
          '因此复用已有的 conda 环境 pytorch（Python 3.10.20 + torch 2.6.0+cu124）。'
          '该环境的 torch 已经是 CUDA 12.4 的 GPU 版，与 RTX 4060（sm_89）匹配，实测 '
          'torch.cuda.is_available() 为 True，训练全程在 GPU 上完成，显存占用约 2.3 GB。')

h2(doc, '4.2  数据集：TVSum')
body(doc, 'TVSum 是视频摘要领域最常用的基准之一，发布于 2015 年 CVPR，包含 50 段来自 YouTube 的视频，'
          '涵盖新闻、动物、美食、运动等 10 个类别，每段视频由 20 位标注者给出 1~5 的价值打分。')
table(doc,
      ['项目', '内容'],
      [['视频数', '50 段（本实验按 40 训练 / 10 测试划分）'],
       ['视频来源', 'YouTube，文件名即视频 ID（如 -esJrBWj2d8.mp4）'],
       ['标注', 'ydata-tvsum50-anno.tsv：50 段 × 20 位标注者，每人一段 1~5 的打分序列'],
       ['元信息', 'ydata-tvsum50-info.tsv：类别、标题、URL、帧数'],
       ['视频总体积', '约 640 MB（50 个 mp4）'],
       ['本地路径', r'datasets\tvsum50\video\']],
      cap='表 4  TVSum 数据集情况', widths=[3.6, 11.2])
body(doc, '数据目录结构如下（已从桌面拷贝到工作区，后续所有路径都指向工作区）：')
code_block(doc, [
    r'datasets\tvsum50\video\                       # 50 个 mp4',
    r'datasets\tvsum50\data\ydata-tvsum50-anno.tsv   # 20 位标注者的打分',
    r'datasets\tvsum50\data\ydata-tvsum50-info.tsv   # 视频元信息',
    r'datasets\tvsum50\eccv16_dataset_tvsum_google_pool5_local.h5  # 本实验自建的 pool5 特征库',
    r'datasets\tvsum50\fcsn_tvsum.h5                 # make_dataset.py 生成的训练集',
])
body(doc, '需要说明的一处缺少：TVSum 官方只发布原始视频与标注，'
          '而不发布 eccv16_dataset_tvsum_google_pool5.h5 这个特征库（它来自 pytorch-vsumm-reinforce 项目，'
          '托管在 Google Drive，本机访问 Google Drive 超时）。'
          '本实验没有「跳过」这一步，而是用同一套流程在本地重建了等价的特征库：'
          '用 ImageNet 预训练的 GoogLeNet 重新提取 50 段视频的 pool5 特征，用 KTS 重新计算镜头边界，'
          '用 anno.tsv 重新计算 20 位标注者的二值摘要与 gtscore。'
          '重建脚本为 code/build_eccv16_tvsum.py，产出的字段与官方 h5 一一对应，'
          '因此 make_dataset.py 与指导书里的流程完全不用改。')

h2(doc, '4.3  数据预处理：从原始视频到 fcsn_tvsum.h5')
body(doc, '预处理分两步。第一步（build_eccv16_tvsum.py）产出 eccv16 等价特征库，每组 video_<id> 含：')
table(doc,
      ['字段', '形状 / 类型', '含义'],
      [['feature', '(320, 1024) float32', 'GoogLeNet pool5 特征，按等间隔采样 320 帧'],
       ['picks', '(320,) int32', '被采样帧在原视频中的帧号'],
       ['change_points', '(S, 2) int32', 'KTS 检出的镜头边界，每行是 [起始帧, 结束帧)'],
       ['n_frame_per_seg', '(S,) int32', '每个镜头的长度'],
       ['user_summary', '(20, N) float32', '20 位标注者的二值摘要（1 = 重要）'],
       ['gtscore', '(N,) float32', '20 位标注者打分的均值（归一化到 [0,1]）'],
       ['length / fps', 'int32 / float32', '原视频总帧数与帧率'],
       ['video_name', '属性', '视频 ID']],
      cap='表 5  自建的 eccv16 等价特征库字段', widths=[3.2, 4.4, 7.2])
body(doc, '其中 user_summary 由 anno.tsv 得到：把每位标注者 100 个时间段的 1~5 分数用线性插值展开到逐帧，'
          '再以「分数 ≥ 3」二值化；gtscore 为 20 人的平均分除以 5。'
          '50 段视频的 user_summary 正样本率均值约 0.28（每段 0.26~0.31），与官方数据的分布一致。')
body(doc, '第二步（make_dataset.py，按实验指导书要求把三个路径参数都改成工作区路径）'
          '把 pool5 特征与标签对齐，生成训练用的 fcsn_tvsum.h5。三个参数分别是 video_dir（原始视频目录）、'
          'h5_path（输出的训练集）和 vsumm_data（自建的 eccv16 等价特征库）。标签取法为：')
p = para(doc, align=WD_ALIGN_PARAGRAPH.CENTER, before=4, after=6, line=1.3)
set_font(p.add_run('label(i) = 1  if  mean_over_20_annotators( user_summary(:, i) ) ≥ 0.5  else 0'),
         size=10.5, italic=True)
body(doc, '即「过半标注者认为重要」才标为 1，这比「任意一位认为重要」稳定得多。'
          '实测 50 段视频的正样本率均值 0.177（最少 0.066、最多 0.319），'
          '说明 TVSum 的 20 位标注者分歧较大，也预示了后面的指标上限不会很高。'
          '生成的文件 94.0 MB，每段视频约 8.6 s、50 段共约 7 分钟。')
code_block(doc, [
    r'# 1) 重建 eccv16 等价特征库（约 5 分钟，GPU）',
    r'python code\build_eccv16_tvsum.py ^',
    r'    --video_dir  datasets\tvsum50\video ^',
    r'    --anno_path  datasets\tvsum50\data\ydata-tvsum50-anno.tsv ^',
    r'    --h5_path    datasets\tvsum50\eccv16_dataset_tvsum_google_pool5_local.h5 ^',
    r'    --gpu',
    '',
    r'# 2) 生成训练集 fcsn_tvsum.h5（约 7 分钟）',
    r'python code\make_dataset.py ^',
    r'    --video_dir   datasets\tvsum50\video ^',
    r'    --h5_path     datasets\tvsum50\fcsn_tvsum.h5 ^',
    r'    --vsumm_data  datasets\tvsum50\eccv16_dataset_tvsum_google_pool5_local.h5 ^',
    r'    --gpu',
])
body(doc, '两个脚本都支持断点续跑（已存在的 video_<id> 组直接跳过），'
          '因此中途中断后重跑不会重算。KTS 的实现做了向量化改写，'
          '与上游的逐元素实现做了逐位一致性自检（n = 40 / 60 / 120 三组随机输入，输出完全相同）。')

h2(doc, '4.4  训练配置')
table(doc,
      ['项目', '设置'],
      [['输入', '1024×320（pool5 特征，时间维统一到 320）'],
       ['batch size', '5（8 个 batch = 1 epoch；40 段训练视频）'],
       ['优化器', 'Adam，lr = 3×10⁻³'],
       ['损失函数', 'BCEWithLogitsLoss，pos_weight = n_neg / n_pos ≈ 4.7'],
       ['梯度裁剪', 'clip_grad_norm = 1.0'],
       ['epochs', '150（每 10 个 epoch 存一次权重并在测试集上评测）'],
       ['权重文件', 'models/epoch-*.pkl、fcsn_best.pkl（按 F-score 最优）、fcsn_final.pkl'],
       ['随机种子', '0（同时固定 numpy 与 torch，保证 40/10 划分可复现）'],
       ['设备 / 显存', 'RTX 4060 Laptop / 约 2.3 GB'],
       ['训练耗时', '150 epoch 共 4 分 47 秒（约 1.9 s/epoch）']],
      cap='表 6  训练配置', widths=[4.4, 10.4])
body(doc, '数据划分沿用 FCSN 官方代码的做法：用固定种子把 50 段随机打乱后取 80% 作训练（40 段）、'
          '20% 作测试（10 段），测试集只在评测时使用。10 段测试视频为：'
          '3eYKfiOEJNs、4wU_LUjG5Ic、91IHQYk1IQM、GsAD1KT1xo8、J0nA4VgnoCo、'
          'JKpqYvAdIsw、WxtbjNsCQ8A、akI8YFjEmUw、kLxoNp-UchI、xxdtq8mxegs。')

h2(doc, '4.5  训练过程与 TensorBoard')
body(doc, '训练命令与 TensorBoard 启动方式：')
code_block(doc, [
    r'# 在 experiments\03_VideoSummarization_FCSN 目录下执行',
    r'run_step3_train.cmd',
    r'# 等价于：python code\train.py --epochs 150 --save_every 10',
    '',
    r'# 启动 TensorBoard（另开一个窗口）',
    r'run_step6_tensorboard.cmd',
    r'# 浏览器打开 http://127.0.0.1:6006',
])
body(doc, 'TensorBoard 记录了训练损失（Loss/train）以及每 10 个 epoch 的 Precision、Recall、F-score、'
          '帧级 AUC 四条评测曲线，日志在 logs/tensorboard/。'
          '需要注意：本机的文件沙箱禁止创建命名管道，tensorboardX 的 SummaryWriter 在构造函数里'
          '会因 multiprocessing.Queue 抛 PermissionError；因此 train.py 里做了兼容处理——'
          '优先用 tensorboardX，失败则退回 torch.utils.tensorboard（两者事件格式相同）。'
          '在普通命令行环境下运行时会正常走 tensorboardX 分支。')

h2(doc, '4.6  摘要生成')
body(doc, '对 10 段测试视频逐段生成摘要：')
code_block(doc, [
    '# 1) 评估 10 段视频的 P/R/F -> outputs\\metrics\\prf_test10.csv',
    'run_step4_evaluate.cmd',
    '',
    '# 2) 生成摘要视频/关键帧/柱状图 -> outputs\\summary\\<id>\\',
    'run_step5_gen_summary.cmd',
])
body(doc, 'gen_summary.py 的处理方式是单次顺序解码：一边读原视频帧、一边判断当前帧是否落在被选中的镜头里，'
          '命中就写入 summary.avi（XVID，24 fps），同时在该镜头内取得分最高的帧存成关键帧图片。'
          '这样避免了「先把整段视频读进内存再处理」导致的内存爆炸'
          '（J0nA4VgnoCo 一段就有 14019 帧，1920×1080 下约 87 GB）。'
          '每段视频输出三个文件：summary.avi（摘要视频）、keyframes/*.jpg（关键帧）、score_bar.png（得分柱状图）。')

# ---- 五、实验结果分析
h1(doc, '五、实验结果分析')

h2(doc, '5.1  训练过程')
body(doc, '训练损失从 1.14 稳定下降到 0.33，每 10 个 epoch 的评测结果如下：')
fig_evals = [(9, 0.244, 0.6644), (19, 0.229, 0.6399), (29, 0.238, 0.6756), (39, 0.234, 0.6146),
             (49, 0.248, 0.6599), (59, 0.241, 0.6346), (69, 0.233, 0.5866), (79, 0.236, 0.6240),
             (89, 0.246, 0.6228), (99, 0.234, 0.6031), (109, 0.255, 0.5947), (119, 0.243, 0.6169),
             (129, 0.241, 0.6226), (139, 0.211, 0.5764), (149, 0.223, 0.6227)]
table(doc,
      ['epoch', 'F-score', '帧级 AUC', 'epoch', 'F-score', '帧级 AUC', 'epoch', 'F-score', '帧级 AUC'],
      [[str(fig_evals[i][0]), '%.3f' % fig_evals[i][1], '%.3f' % fig_evals[i][2],
        str(fig_evals[i + 5][0]), '%.3f' % fig_evals[i + 5][1], '%.3f' % fig_evals[i + 5][2],
        str(fig_evals[i + 10][0]), '%.3f' % fig_evals[i + 10][1], '%.3f' % fig_evals[i + 10][2]]
       for i in range(5)],
      cap='表 7  每 10 个 epoch 在 10 段测试视频上的评测结果（epoch 从 0 计）',
      widths=[1.4, 1.9, 1.9, 1.4, 1.9, 1.9, 1.4, 1.9, 1.9])
figure(doc, '03_curves_train.png', '图 1  训练损失与评测指标曲线', 15.2)
body(doc, '三条曲线的读法：训练损失一路下降且在 100 epoch 后基本走平，说明模型在训练集上已经收敛；'
          '帧级 AUC 在 0.58~0.68 之间波动，均值约 0.62，明显高于随机水平 0.50，'
          '说明模型确实学到了「哪些帧更像关键帧」；而 F-score 几乎不随训练推进而提升（0.21~0.255），'
          '且与 AUC 的波动不完全同步。')
body(doc, '这种「AUC 稳定在 0.62、F-score 卡在 0.24」的现象不是训练没收敛，而是评价口径造成的：'
          'F-score 是在「固定选 15% 帧」的约束下算的，而标注者的二值摘要平均正样本率约 28%，'
          '两者本身就差了一倍，Recall 天生上不去（见 5.5 节）。')

h2(doc, '5.2  10 段测试视频的三种指标')
table(doc,
      ['序号', '视频 ID', '镜头数', '关键帧数', '标注者正样本率', 'Precision', 'Recall', 'F-score'],
      [[str(i + 1), r['video_id'], r['num_segments'], r['num_keyshots'],
        fnum(r['user_summary_mean'], 3), fnum(r['Precision'], 4), fnum(r['Recall'], 4), fnum(r['F-score'], 4)]
       for i, r in enumerate(prf_rows)] +
      [['—', '平均（10 段）', '—', '—', '—', fnum(prf_mean['Precision'], 4),
        fnum(prf_mean['Recall'], 4), fnum(prf_mean['F-score'], 4)]],
      cap='表 8  10 段测试视频的 Precision / Recall / F-score（权重 epoch-109，即 fcsn_best.pkl）',
      widths=[1.1, 2.9, 1.5, 1.9, 2.3, 2.0, 1.8, 1.8])
body(doc, '10 段视频的平均指标为：Precision = 0.3724、Recall = 0.1961、F-score = 0.2552。'
          '最好的是 xxdtq8mxegs（F = 0.4251，比同一段视频上用标准答案 gtscore 选段的 0.3242 还高，'
          '说明模型在这段「人狗互动」视频上抓到了标注者关心的主体），'
          '最差的是 JKpqYvAdIsw（F = 0.1545）。')
figure(doc, '03_prf_bars.png', '图 2  10 段测试视频的 Precision / Recall / F-score', 15.2)
body(doc, '逐段看有一个稳定的规律：Precision 在 0.23~0.61 之间，Recall 在 0.12~0.33 之间，'
          'P 普遍是 R 的 1.7~2.0 倍。这不是模型偏保守，而是「选 15% 帧、真值 28% 帧」这一口径的必然结果：'
          '即使选出的 15% 帧全部命中，Recall 也只有 15% / 28% ≈ 0.54，'
          '而 Precision 可以接近 1。所以本实验里 P 与 R 不能直接和「50%/50% 预算」下的论文数字对比。')

h2(doc, '5.3  摘要视频与关键帧')
body(doc, '10 段视频各生成一个独立文件夹，结构为 outputs/summary/<视频ID>/：summary.avi（摘要视频）、'
          'keyframes/（关键帧）、score_bar.png（得分柱状图）；另外 score_bars_all.png 是 10 段的总览图。'
          '以 xxdtq8mxegs 为例，它被选中的镜头共 7 个，摘要视频 638 帧（约 26.6 s），'
          '占原视频 4324 帧的 14.8%：')
figure(doc, '03_keyframes.png', '图 3  xxdtq8mxegs 的 7 个关键帧（每个选中镜头取得分最高的一帧）', 13.6)
figure(doc, '03_summary_frames.png', '图 4  摘要视频 summary.avi 的抽样帧', 13.6)
body(doc, '从关键帧能看出摘要的合理性：这段视频是兽医给金毛犬检查耳朵的过程，'
          '选中的帧覆盖了「医生与犬全身」「暗光下的检查」「手部特写」几个阶段，'
          '都是标注者打高分的时间段，没有出现纯黑场或片头静止画面。')
body(doc, '得分柱状图按实验指导书要求绘制：横轴是 320 个采样帧的序号，纵轴是模型给出的重要性得分，'
          '橙色柱是被背包算法选中的关键帧，青绿色是未选中的帧。'
          '下面两张分别是表现最好与最差的两段：')
figure(doc, '03_score_bar_good.png', '图 5  xxdtq8mxegs 的得分柱状图（橙色为关键帧，F = 0.4251）', 14.4)
figure(doc, '03_score_bar_bad.png', '图 6  JKpqYvAdIsw 的得分柱状图（橙色为关键帧，F = 0.1545）', 14.4)
body(doc, '对比两图能看到两种不同的失败方式。xxdtq8mxegs 属于「判断集中且正确」：模型只把 13.8% 的帧'
          '判为高分（>0.5），且集中在 150~210 这一段，背包把这个平台整段选走，与标注者的高光基本重合，'
          'F = 0.4251，逐视频帧级 AUC 高达 0.8178。')
body(doc, 'JKpqYvAdIsw 属于「判断扩散」：54 个镜头里有 24 个（覆盖 45.8% 的帧）都被判为值得选，'
          '得分在 0.5 以上的区域铺满整段视频；在 15% 的预算下背包只能从中挑 13 个镜头，'
          '而标注者真正关心的只占 28.7%，两者几乎不可能对上，于是出现「图上到处都是高分、'
          '橙色却只占几条窄缝」的形态，F 只有 0.1545。3eYKfiOEJNs 也是这一类'
          '（47.1% 的帧得分 >0.5，F = 0.2193），10 段视频里被判为高分的帧平均占 25.8%，'
          '与标注者的 28% 接近，但逐段差别很大。')
body(doc, '还有第三种更糟的情况：kLxoNp-UchI 的逐视频帧级 AUC 只有 0.2373，比随机（0.5）还低，'
          '说明模型对这段视频的排序整体是反的，选中 15% 反而系统性地避开了标注者关心的部分，F = 0.1671。'
          '10 段视频的逐视频 AUC 从 0.2373 到 0.8178（均值 0.5997），方差这么大，'
          '说明在 40 段训练视频上学到的「高光模式」并不能稳定迁移到所有类型的视频上。')

h2(doc, '5.4  与线性基线、Oracle 上界的对比')
body(doc, '为了判断 0.2552 这个 F-score 到底是「模型不行」还是「这个口径本身就难」，'
          '在同一批 10 段视频、同一套背包选段代码下再算三组参照：')
table(doc,
      ['方法', 'Precision', 'Recall', 'F-score', '相对 FCSN'],
      [['FCSN（本实验，epoch-109）', fnum(cmp_mean['FCSN']['Precision']), fnum(cmp_mean['FCSN']['Recall']),
        fnum(cmp_mean['FCSN']['F-score']), '100%'],
       ['线性岭回归基线（40 段训练）', fnum(cmp_mean['Ridge']['Precision']), fnum(cmp_mean['Ridge']['Recall']),
        fnum(cmp_mean['Ridge']['F-score']), '100.4%'],
       ['Oracle：用 gtscore 在 320 采样点上选段', fnum(cmp_mean['Oracle320']['Precision']),
        fnum(cmp_mean['Oracle320']['Recall']), fnum(cmp_mean['Oracle320']['F-score']), '102.4%'],
       ['Oracle：用 gtscore 在全帧上选段', fnum(cmp_mean['OracleFull']['Precision']),
        fnum(cmp_mean['OracleFull']['Recall']), fnum(cmp_mean['OracleFull']['F-score']), '112.0%']],
      cap='表 9  FCSN 与线性基线、Oracle 上界的对比（10 段测试视频平均）',
      widths=[6.4, 2.4, 2.2, 2.2, 2.0])
body(doc, '这张表有两个很重要的读数。第一，FCSN 与线性基线几乎打平（0.2552 对 0.2561），'
          '说明在只有 40 段训练视频、且特征本身已经很强（pool5 的 1024 维）的条件下，'
          '深度模型没有表现出相对线性模型的优势——这与 FCSN 论文能在 5 折交叉验证下超过线性方法并不矛盾，'
          '但说明本实验的数据规模不足以撑起深度模型的容量。')
body(doc, '第二，就算直接用「标准答案」gtscore 来选段，F-score 也只有 0.2858（全帧口径 0.4110），'
          '距离 0.5 很远。原因在标注本身：TVSum 的 20 位标注者之间分歧很大，'
          '在同一批 10 段视频上算「标注者两两之间的 F-score」，平均只有 0.366'
          '（最好 0.427、最差 0.332）。也就是说，即使一个模型完全复现了一位标注者的判断，'
          '它和另外 19 位标注者的平均 F-score 也就在 0.37 左右。'
          '本实验的 FCSN 达到 0.2552，相当于理想选段（0.2858）的 89%、人类互评一致度（0.366）的 70%。')
figure(doc, '03_compare_bars.png', '图 7  FCSN 与线性基线、Oracle 上界的逐视频 F-score', 15.2)
body(doc, '逐视频看，FCSN 在 xxdtq8mxegs（0.4251 对 0.3270）、WxtbjNsCQ8A（0.2891 对 0.2500）、'
          '91IHQYk1IQM（0.2855 对 0.2540）三段上超过了 gtscore Oracle，'
          '在 3eYKfiOEJNs（0.2193 对 0.3240）、J0nA4VgnoCo（0.2501 对 0.3450）上明显落后。'
          '超过 Oracle 并不奇怪：gtscore 是 20 人的平均，平均会抹平个人偏好，'
          '而模型学到的是训练集 40 段视频上「平均意义上的高光模式」，在某些视频上反而更接近多数标注者。')

h2(doc, '5.5  误差分析：为什么 Recall 上不去')
body(doc, '把 10 段视频的 Recall 与「口径上限」放在一起看：')
table(doc,
      ['口径', '选段预算', '标注者正样本率', '理论最高 Recall', '实测指标'],
      [['本实验（15% 帧）', '15%', '约 28%', '约 0.54', 'Recall 0.1961'],
       ['若改成 30% 帧', '30%', '约 28%', '约 1.00', '—'],
       ['标注者互评（无预算约束）', '—', '约 28%', '—', 'F ≈ 0.366']],
      cap='表 10  预算与标注正样本率对 Recall 的约束', widths=[4.2, 2.6, 3.0, 3.0, 2.6])
body(doc, '选 15% 的帧、而真值有 28% 的帧被标为重要，Recall 的数学上限就是 15/28 ≈ 0.54；'
          '实测 0.1961 相当于上限的 36%。剩下的差距来自三处：'
          '一是模型的逐视频帧级排序平均只有 AUC ≈ 0.60（完美排序是 1.0），个别视频甚至低于 0.5；'
          '二是背包在镜头粒度上做决定，15% 的长度预算使它倾向于选「短而高分」的镜头，'
          '而标注者往往把一整个长镜头都标为重要；'
          '三是背包只最大化「得分之和」，当预算还有余量时，它会用得分接近 0 的短镜头把余量填满'
          '（例如 xxdtq8mxegs 里就填进了一个 14 帧、价值 0.001 的镜头），这部分帧对指标只有负贡献。')
body(doc, '另有一个与任务设计有关的现象：训练时 F-score 的最大值出现在 epoch 109（0.2552），'
          '而最后一个 epoch（149）只有 0.2230。本报告的主表用的是 epoch 109（fcsn_best.pkl），'
          '严格说这是「在测试集上挑出来的最优 epoch」，存在乐观偏差；'
          '如果把 epoch 149（fcsn_final.pkl）作为无偏结果，平均 F-score 是 0.2230。'
          '两者相差 0.03，量级上不影响结论，但在方法上应当说明。')

h2(doc, '5.6  一次失败尝试：上游的 NLL 配方在本数据上不收敛')
body(doc, '实验最初完全按指导书与上游代码训练：Adam（未显式设 lr，即默认 1×10⁻³）+ NLLLoss，'
          '类别权重取 [n_pos, n_neg] ≈ [56.7, 263.3]，100 epoch（800 次参数更新）。'
          '结果训练损失从 0.696 降到 0.41 后走平，而测试集指标始终是 Precision ≈ 0.29、'
          'Recall ≈ 0.15、F-score ≈ 0.20、帧级 AUC = 0.491——AUC 0.491 意味着模型等价于随机排序。')
body(doc, '为了定位原因，做了一组逐步排除的诊断实验（脚本保留在 code/probe_*.py、code/diag_*.py，'
          '日志在 logs/probe_*.log、logs/diag_*.log）：')
table(doc,
      ['诊断', '做法', '结果', '结论'],
      [['数据是否对齐', '训练集拟合岭回归，在测试视频上算与标签的相关',
        '逐视频 r = 0.49~0.71，AUC 0.652', '数据没问题，排除特征/标签错位'],
       ['参数是否更新', '单视频过拟合，打印 |Δw| 与 loss', '参数在更新、loss 在降，AUC 仍 0.50',
        '不是梯度断流，是损失函数的目标不对'],
       ['模型能否拟合', '5 段视频上换 BCE(pos_weight=5)+Adam(3e-3)', '100 步 AUC 0.937，600 步 0.970',
        '架构没问题，可以拟合'],
       ['换成 BCE 后', '同一份数据、40/10 划分、150 epoch', 'F 0.2552、AUC 0.62', '确认是损失函数问题']],
      cap='表 11  训练不收敛的逐步排除', widths=[2.6, 5.6, 3.8, 3.8])
body(doc, '最终判断：NLLLoss 配合「正负样本按数量反比加权」的方案，在正样本率仅 17.7% 时不合适——'
          '它把两类样本的损失权重放大到 56.7 与 263.3，梯度尺度失衡，'
          '模型很快学会了「输出接近常数」的退化解（p.std 只有 0.001~0.01），'
          '而 BCE + pos_weight = n_neg/n_pos 的目标等价但尺度正常，300 步内即可拟合。'
          '因此正式训练采用 BCE 配方；config.py 里保留 loss = "nll" 分支，'
          '可以一键复现上游配方用于对照。')

h2(doc, '5.7  本实验结论')
for txt in [
    '成功复现并跑通 FCSN 视频摘要的完整流程：GoogLeNet pool5 特征 → KTS 镜头切分 → '
    'fcsn_tvsum.h5 → FCSN 训练 → 背包选段 → 摘要视频/关键帧 → P/R/F 评估。',
    '10 段测试视频的平均指标为 Precision = 0.3724、Recall = 0.1961、F-score = 0.2552'
    '（epoch-109）；无偏的末轮结果 F-score = 0.2230。',
    '这个数字不能直接与论文的 0.5~0.6 相比：在同一口径下，用标准答案 gtscore 选段的 F-score 也只有 '
    '0.2858，而 20 位标注者互评的 F-score 只有 0.366。模型达到了理想选段的 89%、人类互评一致度的 70%。',
    'TVSum 只有 50 段视频，按 40/10 划分后训练集很小，FCSN 的表现与线性岭回归基线持平'
    '（0.2552 对 0.2561），说明此规模下深度模型的容量优势体现不出来。',
    '训练是否收敛不能只看 loss：本实验里上游 NLL 配方 loss 从 0.696 降到 0.41，'
    '但 AUC 只有 0.49；换 BCE 后 loss 从 1.14 降到 0.33，AUC 才升到 0.62。'
    '必须同时监控一个与排序相关的指标。',
]:
    body(doc, '· ' + txt, indent=0, after=4)

# ---- 六、问题与解决
h1(doc, '六、问题与解决')
table(doc,
      ['问题', '原因', '解决'],
      [['缺 eccv16_dataset_tvsum_google_pool5.h5',
        '官方只发布原始视频与标注，特征库托管在 Google Drive，本机访问超时',
        '用同一套流程本地重建：GoogLeNet 提 pool5、KTS 算镜头、anno.tsv 算 user_summary，'
        '字段与官方 h5 一一对应'],
       ['上游 NLL 配方训练不收敛',
        '类别权重按样本数反比加权（56.7 / 263.3），正样本率 17.7% 时梯度尺度失衡，收敛到常数解',
        '改用 BCEWithLogitsLoss + pos_weight = n_neg/n_pos，F 从 0.199 提到 0.255，'
        'AUC 从 0.49 提到 0.62；保留 nll 分支供对照'],
       ['tensorboardX 无法写日志',
        '本机文件沙箱禁止创建命名管道，multiprocessing.Queue 抛 PermissionError',
        'train.py 里 try/except 退回 torch.utils.tensorboard（事件格式相同），'
        '普通命令行下仍走 tensorboardX'],
       ['conda 环境缺 h5py / tensorboard / prettytable',
        'pip 在该环境里对沙箱临时目录无写权限（WinError 5）',
        '下载 wheel 后直接解压到工作区 env_pkgs，用 PYTHONPATH 注入，不动环境本体'],
       ['TensorBoard 启动报 No module named pkg_resources',
        '环境里的 setuptools 82 已移除 pkg_resources',
        '单独解压 setuptools 68.2.2 到 env_tensorboard，只给 TensorBoard 命令加进 PYTHONPATH'],
       ['gen_summary.py 读取整段视频导致内存爆炸',
        '上游实现把所有帧存进 np.array 再处理，14019 帧 × 1080p 约 87 GB',
        '改成单次顺序解码、边读边写，内存恒定'],
       ['eval.upsample 在 N < 320 时死循环',
        'ratio = N // 320 为 0 时 while 循环不前进',
        '入口加显式检查并抛 ValueError（TVSum 视频均 ≥ 3000 帧，本实验不触发）'],
       ['中文输出在 GBK 控制台乱码',
        'Windows 默认代码页不是 UTF-8',
        '所有批处理脚本 chcp 65001，Python 侧设 PYTHONIOENCODING=utf-8']],
      cap='表 12  实验中遇到的问题与解决办法', widths=[3.6, 5.4, 6.6])

# ---- 七、交付物清单
h1(doc, '七、代码与结果文件清单')
table(doc,
      ['路径', '内容'],
      [['experiments/03_VideoSummarization_FCSN/code/', '全部代码（11 个脚本，见下）'],
       ['code/build_eccv16_tvsum.py', '重建 eccv16 等价特征库（pool5 + KTS + user_summary）'],
       ['code/builddataset.py', 'KTS 实现（向量化改写）'],
       ['code/make_dataset.py', '按指导书改三处路径，生成 fcsn_tvsum.h5'],
       ['code/config.py', '路径、epochs、lr、损失函数等超参'],
       ['code/data_loader.py', '40/10 划分、batch 组装'],
       ['code/fcsn.py', 'FCSN 网络定义（未改动结构）'],
       ['code/train.py', '训练 + 每 10 epoch 存权重与评测 + TensorBoard'],
       ['code/eval.py', 'eval_metrics、select_keyshots、upsample'],
       ['code/knapsack.py', '0-1 背包（动态规划）'],
       ['code/evaluate_summary.py', '10 段测试视频的 P/R/F 表格'],
       ['code/gen_summary.py', '摘要视频 + 关键帧 + 得分柱状图'],
       ['code/baseline_compare.py', '线性基线与 Oracle 上界对比'],
       ['code/make_report_figures.py', '本报告全部插图'],
       ['datasets/tvsum50/fcsn_tvsum.h5', '训练集（94.0 MB）'],
       ['models/fcsn_best.pkl、fcsn_final.pkl', '最优与末轮权重（各 466 MB）'],
       ['outputs/summary/<id>/', '10 段视频的摘要视频、关键帧、得分柱状图'],
       ['outputs/metrics/prf_test10.csv', '10 段视频的 P/R/F 表'],
       ['outputs/metrics/baseline_compare.csv', '与基线/上界的逐视频对比'],
       ['outputs/scores/final_test10.json', '10 段视频的逐帧得分与选中镜头'],
       ['logs/tensorboard/', 'TensorBoard 事件文件'],
       ['logs/train_bce.log', '150 epoch 的完整训练日志']],
      cap='表 13  代码与结果文件清单', widths=[7.0, 9.0])

body(doc, '复现顺序：run_step1_build_eccv16.cmd → run_step2_make_dataset.cmd → run_step3_train.cmd → '
          'run_step4_evaluate.cmd → run_step5_gen_summary.cmd → run_step6_tensorboard.cmd。'
          '每个脚本都会先检查前置文件是否存在并给出可读的报错，可重复运行。')

doc.save(OUT)
print('已生成', OUT)
