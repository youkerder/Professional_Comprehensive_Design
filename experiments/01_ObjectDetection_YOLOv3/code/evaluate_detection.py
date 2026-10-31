#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 01 检测效果评测：用 VOC 标注算 Precision / Recall / F1 / AP / mAP。

上游 keras-yolo3 **没有**提供检测精度评测，只有逐张画框的可视化，
所以实验报告里"检测效果"这一节需要有据可依的数字，就靠这个脚本。

数据来源：
  * 预测：detect_images.py 生成的 `<结果目录>/detections.csv`
          boxes 列格式 `x0,y0,x1,y1,score,class_id`，多个框用 `;` 分隔
  * 真值：datasets/cat_face/Annotations/*.xml（PascalVOC）

评测口径（与 VOC2007 / COCO 的通用做法一致）：
  * 一张预测框与同类别、IoU >= --iou 且**尚未被匹配**的真值框配对，即为 TP；
  * 每个真值框最多被匹配一次（贪心，按置信度从高到低）；
  * 剩余预测框为 FP；未匹配的真值框为 FN；
  * AP 用 VOC2010 之后的 all-point 插值法（PR 曲线下的面积）；
  * mAP = 各类别 AP 的平均。

只依赖 Python 标准库，**不需要 TensorFlow**，所以在任何环境里都能跑。

用法：
    python evaluate_detection.py --pred ../outputs/detect/run1/detections.csv
    python evaluate_detection.py --pred <csv> --iou 0.5 --out ../outputs/metrics
    python evaluate_detection.py --selftest        # 用合成数据自检
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

try:
    from config import ANNO_DIR, CLASSES_PATH, IMAGE_EXTS, OUTPUTS_DIR, VOC_ROOT
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ANNO_DIR, CLASSES_PATH, IMAGE_EXTS, OUTPUTS_DIR, VOC_ROOT


# --------------------------------------------------------------------------
# 基础几何 / 读取
# --------------------------------------------------------------------------

def iou(a, b) -> float:
    """两个 (x0, y0, x1, y1) 框的 IoU。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_classes(path: Path):
    """读类别文件，返回 {class_id: name}。"""
    with open(path, "r", encoding="utf-8") as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    return {i: n for i, n in enumerate(names)}


def read_gt(anno_dir: Path, class_to_id: dict):
    """读所有 XML，返回 {stem: [(x0,y0,x1,y1,class_id), ...]}。

    跳过 difficult=1 的目标（与训练时的处理保持一致）。
    """
    gt = {}
    skipped_difficult = 0
    for xml in sorted(anno_dir.glob("*.xml")):
        try:
            root = ET.parse(str(xml)).getroot()
        except ET.ParseError:
            continue
        boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in class_to_id:
                continue
            diff = obj.findtext("difficult")
            try:
                if diff is not None and int(float(diff)) == 1:
                    skipped_difficult += 1
                    continue
            except (TypeError, ValueError):
                pass
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                x0 = float(bnd.findtext("xmin"))
                y0 = float(bnd.findtext("ymin"))
                x1 = float(bnd.findtext("xmax"))
                y1 = float(bnd.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append((x0, y0, x1, y1, class_to_id[name]))
        gt[xml.stem] = boxes
    return gt, skipped_difficult


def parse_pred_boxes(field: str):
    """解析 detections.csv 的 boxes 列。"""
    out = []
    if not field:
        return out
    for item in field.split(";"):
        parts = item.strip().split(",")
        if len(parts) < 6:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in parts[:4])
            score = float(parts[4])
            cid = int(float(parts[5]))
        except ValueError:
            continue
        out.append((x0, y0, x1, y1, score, cid))
    return out


def read_pred(csv_path: Path):
    """读 detections.csv，返回 {stem: [(x0,y0,x1,y1,score,class_id), ...]}。"""
    pred = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            stem = Path(fn).stem
            pred[stem] = parse_pred_boxes(row.get("boxes") or "")
    return pred


# --------------------------------------------------------------------------
# AP / mAP
# --------------------------------------------------------------------------

def voc_ap(rec, prec):
    """VOC2010+ 的 all-point 插值 AP。"""
    mrec = [0.0] + list(rec) + [1.0]
    mpre = [0.0] + list(prec) + [0.0]
    # 从后往前取精度包络
    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate(gt: dict, pred: dict, classes: dict, iou_thr: float):
    """按类别算 AP、Precision、Recall、F1，并给出 PR 曲线点。"""
    stems = sorted(set(gt) | set(pred))
    results = {}

    for cid, cname in classes.items():
        # 收集该类别所有预测，按分数降序
        dets = []
        n_gt = 0
        for stem in stems:
            g = [b for b in gt.get(stem, []) if b[4] == cid]
            n_gt += len(g)
            for (x0, y0, x1, y1, score, pcid) in pred.get(stem, []):
                if pcid == cid:
                    dets.append((score, stem, (x0, y0, x1, y1)))
        dets.sort(key=lambda t: -t[0])

        # 贪心匹配
        matched = defaultdict(set)  # stem -> 已匹配的 gt 下标
        tp_flags = []
        n_fp = 0
        per_image = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        gt_by_stem = {s: [b for b in gt.get(s, []) if b[4] == cid] for s in stems}

        for (score, stem, box) in dets:
            g = gt_by_stem.get(stem, [])
            best_i, best_iou = -1, 0.0
            for i, gb in enumerate(g):
                if i in matched[stem]:
                    continue
                v = iou(box, gb[:4])
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_iou >= iou_thr and best_i >= 0:
                matched[stem].add(best_i)
                tp_flags.append(1)
                per_image[stem]["tp"] += 1
            else:
                tp_flags.append(0)
                n_fp += 1
                per_image[stem]["fp"] += 1

        n_tp = sum(tp_flags)
        n_fn = n_gt - n_tp
        for stem in stems:
            per_image[stem]["fn"] = len(gt_by_stem.get(stem, [])) - per_image[stem]["tp"]

        # PR 曲线
        cum_tp = 0
        rec, prec = [], []
        for k, flag in enumerate(tp_flags, 1):
            cum_tp += flag
            r = cum_tp / n_gt if n_gt else 0.0
            p = cum_tp / k if k else 0.0
            rec.append(r)
            prec.append(p)
        ap = voc_ap(rec, prec) if n_gt else 0.0

        # 用默认阈值下的最终点作为 P/R/F1（即取全部预测）
        final_p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
        final_r = n_tp / n_gt if n_gt else 0.0
        f1 = (2 * final_p * final_r / (final_p + final_r)) if (final_p + final_r) else 0.0

        results[cname] = {
            "class_id": cid,
            "n_gt": n_gt,
            "n_pred": len(dets),
            "tp": n_tp, "fp": n_fp, "fn": n_fn,
            "precision": final_p, "recall": final_r, "f1": f1,
            "ap": ap,
            "pr_curve": list(zip(rec, prec)),
            "per_image": dict(per_image),
        }

    aps = [v["ap"] for v in results.values() if v["n_gt"] > 0]
    return results, (sum(aps) / len(aps) if aps else 0.0)


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------

def print_report(results, mAP, iou_thr, n_images, skipped_difficult):
    print("")
    print("=" * 78)
    print("检测结果评测（IoU 阈值 = %.2f）" % iou_thr)
    print("=" * 78)
    print("评测图片数 : %d" % n_images)
    if skipped_difficult:
        print("跳过 difficult=1 的目标 : %d 个" % skipped_difficult)
    print("")
    print("%-10s %6s %6s %6s %6s %6s %8s %8s %8s" %
          ("类别", "真值", "预测", "TP", "FP", "FN", "Precision", "Recall", "AP"))
    print("-" * 78)
    for cname, r in results.items():
        print("%-10s %6d %6d %6d %6d %6d %8.4f %8.4f %8.4f" %
              (cname, r["n_gt"], r["n_pred"], r["tp"], r["fp"], r["fn"],
               r["precision"], r["recall"], r["ap"]))
    print("-" * 78)
    print("%-10s %6s %6s %6s %6s %6s %8s %8s %8.4f" %
          ("mAP", "", "", "", "", "", "", "", mAP))
    print("")
    for cname, r in results.items():
        print("  %s: Precision=%.4f  Recall=%.4f  F1=%.4f  AP=%.4f"
              % (cname, r["precision"], r["recall"], r["f1"], r["ap"]))


def write_outputs(out_dir: Path, results, mAP, iou_thr, n_images, skipped_difficult):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "iou_threshold": iou_thr,
        "num_images": n_images,
        "skipped_difficult": skipped_difficult,
        "mAP": mAP,
        "classes": {
            c: {k: v for k, v in r.items() if k not in ("pr_curve", "per_image")}
            for c, r in results.items()
        },
    }
    (out_dir / "detection_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总表（Markdown，可直接贴进报告）
    lines = ["# 实验 01 检测指标", "",
             "IoU 阈值：%.2f　评测图片数：%d" % (iou_thr, n_images), "",
             "| 类别 | 真值数 | 预测数 | TP | FP | FN | Precision | Recall | F1 | AP |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cname, r in results.items():
        lines.append("| %s | %d | %d | %d | %d | %d | %.4f | %.4f | %.4f | %.4f |"
                     % (cname, r["n_gt"], r["n_pred"], r["tp"], r["fp"], r["fn"],
                        r["precision"], r["recall"], r["f1"], r["ap"]))
    lines.append("| **mAP** |  |  |  |  |  |  |  |  | **%.4f** |" % mAP)
    (out_dir / "detection_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 逐图明细（便于做错误分析）
    with open(out_dir / "per_image.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "filename", "tp", "fp", "fn"])
        for cname, r in results.items():
            for stem, d in sorted(r["per_image"].items()):
                w.writerow([cname, stem, d["tp"], d["fp"], d["fn"]])

    # PR 曲线点（画图用）
    with open(out_dir / "pr_curve.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "recall", "precision"])
        for cname, r in results.items():
            for (rec, prec) in r["pr_curve"]:
                w.writerow([cname, "%.6f" % rec, "%.6f" % prec])

    print("")
    print("已输出：")
    for name in ("detection_metrics.json", "detection_metrics.md",
                 "per_image.csv", "pr_curve.csv"):
        print("  %s" % (out_dir / name))


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------

def selftest() -> int:
    """用构造好的数据验证 IoU、匹配、AP 是否符合预期。"""
    print("=" * 70)
    print("evaluate_detection.py 自检")
    print("=" * 70)
    failed = []
    passed = [0]

    def check(cond, label, detail=""):
        if cond:
            passed[0] += 1
            print("  [ OK ] %s" % label)
        else:
            print("  [FAIL] %s  %s" % (label, detail))
            failed.append(label)

    # --- IoU ---
    check(abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9, "完全重合 IoU=1")
    check(abs(iou((0, 0, 10, 10), (10, 0, 20, 10))) < 1e-9, "相切 IoU=0")
    check(abs(iou((0, 0, 10, 10), (20, 20, 30, 30))) < 1e-9, "不相交 IoU=0")
    # 半重叠：交集 50，并集 150 -> 1/3
    check(abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9,
          "半重叠 IoU=1/3", "实际 %.6f" % iou((0, 0, 10, 10), (5, 0, 15, 10)))
    check(abs(iou((0, 0, 10, 10), (0, 0, 0, 0))) < 1e-9, "退化框 IoU=0")

    classes = {0: "cat"}

    # --- 完美预测：1 个真值，1 个完全重合的预测 ---
    gt = {"a": [(0, 0, 10, 10, 0)]}
    pred = {"a": [(0, 0, 10, 10, 0.9, 0)]}
    res, mAP = evaluate(gt, pred, classes, 0.5)
    r = res["cat"]
    check(r["tp"] == 1 and r["fp"] == 0 and r["fn"] == 0, "完美预测 TP=1 FP=0 FN=0",
          "实际 TP=%d FP=%d FN=%d" % (r["tp"], r["fp"], r["fn"]))
    check(abs(r["precision"] - 1) < 1e-9 and abs(r["recall"] - 1) < 1e-9,
          "完美预测 P=R=1")
    check(abs(mAP - 1.0) < 1e-9, "完美预测 mAP=1", "实际 %.4f" % mAP)

    # --- 漏检：有真值无预测 -> FN=1, recall=0, AP=0 ---
    res, mAP = evaluate({"a": [(0, 0, 10, 10, 0)]}, {}, classes, 0.5)
    r = res["cat"]
    check(r["fn"] == 1 and r["tp"] == 0, "漏检 FN=1", "实际 FN=%d" % r["fn"])
    check(abs(r["recall"]) < 1e-9, "漏检 recall=0")
    check(abs(mAP) < 1e-9, "漏检 mAP=0", "实际 %.4f" % mAP)

    # --- 误检：无真值有预测 -> FP=1, precision=0 ---
    res, _ = evaluate({}, {"a": [(0, 0, 10, 10, 0.9, 0)]}, classes, 0.5)
    r = res["cat"]
    check(r["fp"] == 1 and r["tp"] == 0, "误检 FP=1", "实际 FP=%d" % r["fp"])
    check(abs(r["precision"]) < 1e-9, "误检 precision=0")
    check(r["n_gt"] == 0, "无真值时 n_gt=0")

    # --- IoU 刚好低于阈值应判为 FP ---
    res, _ = evaluate({"a": [(0, 0, 10, 10, 0)]},
                      {"a": [(0, 0, 14, 10, 0.9, 0)]}, classes, 0.5)
    r = res["cat"]
    # 交集 10*10=100，并集 100+140-100=140 -> 0.714 > 0.5，应为 TP
    check(r["tp"] == 1, "IoU=0.714 > 0.5 判为 TP", "实际 TP=%d" % r["tp"])

    res, _ = evaluate({"a": [(0, 0, 10, 10, 0)]},
                      {"a": [(0, 0, 30, 10, 0.9, 0)]}, classes, 0.5)
    r = res["cat"]
    # 交集 100，并集 100+300-100=300 -> 0.333 < 0.5，应为 FP
    check(r["tp"] == 0 and r["fp"] == 1, "IoU=0.333 < 0.5 判为 FP",
          "实际 TP=%d FP=%d" % (r["tp"], r["fp"]))

    # --- 一个真值只能被匹配一次 ---
    gt = {"a": [(0, 0, 10, 10, 0)]}
    pred = {"a": [(0, 0, 10, 10, 0.9, 0), (0, 0, 10, 10, 0.8, 0)]}
    res, _ = evaluate(gt, pred, classes, 0.5)
    r = res["cat"]
    check(r["tp"] == 1 and r["fp"] == 1, "重复检测：1 TP + 1 FP",
          "实际 TP=%d FP=%d" % (r["tp"], r["fp"]))

    # --- 匹配应按分数从高到低 ---
    # 真值只有一个，两个预测：低分的更准、高分的更偏
    # 正确行为：高分的先来，如果它与真值 IoU 达标就占用真值，低分的变 FP
    gt = {"a": [(0, 0, 10, 10, 0)]}
    pred = {"a": [(0, 0, 12, 12, 0.9, 0), (0, 0, 10, 10, 0.5, 0)]}
    res, _ = evaluate(gt, pred, classes, 0.5)
    r = res["cat"]
    check(r["tp"] == 1 and r["fp"] == 1, "按分数贪心匹配：1 TP + 1 FP",
          "实际 TP=%d FP=%d" % (r["tp"], r["fp"]))

    # --- 多类别相互独立 ---
    classes2 = {0: "cat", 1: "dog"}
    gt = {"a": [(0, 0, 10, 10, 0)]}
    pred = {"a": [(0, 0, 10, 10, 0.9, 1)]}
    res, _ = evaluate(gt, pred, classes2, 0.5)
    check(res["cat"]["fn"] == 1 and res["dog"]["fp"] == 1,
          "类别不混：cat 漏检、dog 误检",
          "cat.fn=%d dog.fp=%d" % (res["cat"]["fn"], res["dog"]["fp"]))

    # --- AP 的插值：2 个真值，2 个预测，都命中 -> AP=1 ---
    gt = {"a": [(0, 0, 10, 10, 0), (20, 20, 30, 30, 0)]}
    pred = {"a": [(0, 0, 10, 10, 0.9, 0), (20, 20, 30, 30, 0.8, 0)]}
    res, mAP = evaluate(gt, pred, classes, 0.5)
    check(abs(mAP - 1.0) < 1e-9, "两个真值全命中 AP=1", "实际 %.4f" % mAP)

    # --- AP 半对：1 命中 1 漏 -> AP ≈ 0.5（VOC all-point 插值）---
    gt = {"a": [(0, 0, 10, 10, 0), (20, 20, 30, 30, 0)]}
    pred = {"a": [(0, 0, 10, 10, 0.9, 0)]}
    res, mAP = evaluate(gt, pred, classes, 0.5)
    check(abs(mAP - 0.5) < 1e-9, "命中一半 AP=0.5", "实际 %.4f" % mAP)

    # --- 解析预测框字符串 ---
    parsed = parse_pred_boxes("1,2,3,4,0.9,0;5,6,7,8,0.5,0")
    check(len(parsed) == 2, "解析 2 个框", "实际 %d" % len(parsed))
    check(parsed[0] == (1.0, 2.0, 3.0, 4.0, 0.9, 0), "框字段顺序正确", str(parsed[0]))
    check(parse_pred_boxes("") == [], "空字符串解析为空列表")
    check(parse_pred_boxes("bad,data") == [], "非法数据被忽略")

    print("")
    print("=" * 70)
    if failed:
        print("通过 %d 项，失败 %d 项" % (passed[0], len(failed)))
        for f in failed:
            print("  [FAIL] %s" % f)
    else:
        print("通过 %d 项，全部通过" % passed[0])
    print("=" * 70)
    return 1 if failed else 0


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="实验 01：用 VOC 标注评测检测效果（P/R/F1/AP/mAP）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pred", default=None,
                    help="detect_images.py 生成的 detections.csv")
    ap.add_argument("--anno-dir", default=str(ANNO_DIR), help="VOC 标注目录")
    ap.add_argument("--classes", default=str(CLASSES_PATH), help="类别文件")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU 匹配阈值")
    ap.add_argument("--out", default=str(OUTPUTS_DIR / "metrics"), help="指标输出目录")
    ap.add_argument("--selftest", action="store_true", help="用合成数据自检")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.selftest:
        return selftest()

    if not args.pred:
        print("[错误] 需要 --pred 指向 detections.csv")
        print("       先用 detect_images.py 跑出结果：")
        print("         python detect_images.py --weights <权重>")
        print("       或者跑自检看看评测逻辑：")
        print("         python evaluate_detection.py --selftest")
        return 1

    pred_path = Path(args.pred).resolve()
    anno_dir = Path(args.anno_dir).resolve()
    classes_path = Path(args.classes).resolve()

    print("=" * 78)
    print("实验 01 · 检测效果评测")
    print("=" * 78)
    print("预测文件 : %s" % pred_path)
    print("标注目录 : %s" % anno_dir)
    print("类别文件 : %s" % classes_path)
    print("")

    if not pred_path.is_file():
        print("[错误] 找不到预测文件：%s" % pred_path)
        return 1
    if not anno_dir.is_dir():
        print("[错误] 找不到标注目录：%s" % anno_dir)
        return 1
    if not classes_path.is_file():
        print("[错误] 找不到类别文件：%s" % classes_path)
        return 1

    classes = load_classes(classes_path)
    name_to_id = {v: k for k, v in classes.items()}
    print("类别     : %s" % classes)

    gt, skipped = read_gt(anno_dir, name_to_id)
    pred = read_pred(pred_path)

    eval_stems = sorted(set(pred))
    gt_eval = {s: gt.get(s, []) for s in eval_stems}
    n_gt_boxes = sum(len(v) for v in gt_eval.values())
    print("预测覆盖 : %d 张图" % len(eval_stems))
    print("这些图的真值框 : %d 个" % n_gt_boxes)
    print("")
    if not eval_stems:
        print("[错误] 预测文件里没有任何图片记录。")
        return 1
    if n_gt_boxes == 0:
        print("[警告] 这些图**没有对应的 VOC 标注**，无法算精度指标。")
        print("       检测演示用的 datasets/cat_face/test/ 里的 8 张图是没有标注的，")
        print("       要对它们评测必须先手工标注；")
        print("       或者改用 --input 指向有标注的目录（例如 JPEGImages）来跑检测。")
        return 1

    results, mAP = evaluate(gt_eval, pred, classes, args.iou)
    print_report(results, mAP, args.iou, len(eval_stems), skipped)
    write_outputs(Path(args.out).resolve(), results, mAP, args.iou,
                  len(eval_stems), skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
