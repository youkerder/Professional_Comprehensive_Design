"""
生成 FCSN 训练用的 hdf5 数据集（fcsn_tvsum.h5）。

相对上游仓库的改动：
1. 不再硬编码 .cuda()，按 --gpu / 显卡可用性自动选设备；
2. Windows 下 cv2.VideoCapture 直接吃普通路径，不用 as_uri()；
3. 视频 id 用文件名（youtube id），与本项目自建的 eccv16 等价 h5 对齐；
4. 标签二值化：20 位标注者的平均重要性 >= 0.5 记为 1，可直接给 NLLLoss 用；
5. 支持 --limit（调试）与断点续跑（已存在的 video_ 组跳过）。

用法：
  python make_dataset.py --video_dir <mp4目录> --h5_path <输出h5> --vsumm_data <eccv16 h5> --gpu
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

pbar = argparse.ArgumentParser()
pbar.add_argument('--video_dir', type=str, required=True,
                  help='directory containing mp4 files of TVSum.')
pbar.add_argument('--h5_path', type=str, required=True,
                  help='save path of the generated dataset (hdf5).')
pbar.add_argument('--vsumm_data', type=str, required=True,
                  help='preprocessed hdf5 (eccv16 格式)，从里面复制 change_points / user_summary。')
pbar.add_argument('--gpu', action='store_true', help='use GPU for feature extraction.')
pbar.add_argument('--limit', type=int, default=0, help='只处理前 N 个视频（调试用）。')
pbar.add_argument('--target_len', type=int, default=320, help='每条视频采样的帧数。')
args = pbar.parse_args()


class Rescale(object):
    """把 PIL.Image 缩放到指定尺寸。"""

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        return image.resize((int(new_w), int(new_h)), resample=Image.BILINEAR)


transform = transforms.Compose([
    Rescale(224, 224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')
print('特征提取设备:', device)
net = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).float().to(device)
net.eval()
fea_net = nn.Sequential(*list(net.children())[:-2])
for p in fea_net.parameters():
    p.requires_grad_(False)


def get_oracle_summary(user_summary):
    """（保留上游实现，仅作参考）贪心生成 oracle summary，并打印 F-score。"""
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        precision = overlap_arr / (oracle_sum + 1e-8)
        recall = overlap_arr / (true_sum_arr + 1e-8)
        fscores = 2 * precision * recall / (precision + recall + 1e-8)
        cur_fscore = float(np.mean(fscores))
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break
    return oracle_summary


def video2fea(video_path, h5_f, vsumm_data, target_len=320):
    vid = os.path.splitext(os.path.basename(video_path))[0]
    key = 'video_' + vid
    if key in h5_f:            # 断点续跑
        tqdm.write('skip existing ' + vid)
        return True

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        tqdm.write('[WARN] 打不开视频: ' + video_path)
        return False
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio = max(length // target_len, 1)
    tqdm.write('Processing %s  frames=%d ratio=%d' % (vid, length, ratio))

    usr_sum_arr = vsumm_data[key]['user_summary'][()]
    # 20 位标注者的平均重要性（0~1），>=0.5 记为关键帧
    usr_mean = usr_sum_arr.mean(axis=0)
    cps = vsumm_data[key]['change_points'][()]
    n_frame_per_seg = vsumm_data[key]['n_frame_per_seg'][()]

    fea, label, picks = [], [], []
    i = 0
    success, frame = video.read()
    while success:
        if (i + 1) % ratio == 0 and len(fea) < target_len:
            small = cv2.resize(frame, (224, 224))
            with torch.no_grad():
                feat = fea_net(transform(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
                               .unsqueeze(0).to(device)).squeeze().cpu()
            fea.append(feat)
            label.append(int(usr_mean[min(i, len(usr_mean) - 1)] >= 0.5))
            picks.append(i)
        i += 1
        success, frame = video.read()
    video.release()

    if len(fea) == 0:
        tqdm.write('[WARN] 解不出帧: ' + vid)
        return False

    fea = torch.stack(fea)[:target_len].numpy()
    label = np.asarray(label[:target_len], dtype=np.int8)
    # 不足 320 帧时用最后一帧复制补齐（与原版 fea[:320] 行为兼容）
    if len(fea) < target_len:
        pad = target_len - len(fea)
        fea = np.vstack([fea, np.repeat(fea[-1:], pad, axis=0)])
        label = np.concatenate([label, np.repeat(label[-1:], pad)])
        picks = picks + [picks[-1] + (k + 1) * ratio for k in range(pad)]

    v_data = h5_f.create_group(key)
    v_data['feature'] = fea.astype(np.float32)
    v_data['label'] = label
    v_data['length'] = len(usr_mean)
    v_data['change_points'] = cps
    v_data['n_frame_per_seg'] = n_frame_per_seg
    v_data['picks'] = np.asarray(picks, dtype=np.int32)
    v_data['user_summary'] = usr_sum_arr
    tqdm.write('  saved %s  正样本占比 %.3f' % (vid, label.mean()))
    return True


def make_dataset(video_dir, h5_path, vsumm_data, limit=0, target_len=320):
    video_list = sorted(f for f in os.listdir(video_dir) if f.lower().endswith('.mp4'))
    if limit:
        video_list = video_list[:limit]
    ok = 0
    with h5py.File(h5_path, 'a') as h5_f:
        for fn in tqdm(video_list, desc='Video', ncols=80):
            if video2fea(os.path.join(video_dir, fn), h5_f, vsumm_data, target_len):
                ok += 1
            h5_f.flush()
    print('done: %d/%d 个视频写入 %s' % (ok, len(video_list), h5_path))


if __name__ == '__main__':
    with h5py.File(args.vsumm_data, 'r') as vsumm:
        make_dataset(args.video_dir, args.h5_path, vsumm, args.limit, args.target_len)
