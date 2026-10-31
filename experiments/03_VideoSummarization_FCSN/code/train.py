# -*- coding: utf-8 -*-
"""训练 FCSN 视频摘要模型（TVSum 50 段，40 训练 / 10 测试）。

与上游 weirme/Video_Summary_using_FCSN 的 train.py 相比，本实现保留了
「每 save_every 个 epoch 存权重 + 在测试集上算 P/R/F 并写 json」的流程，
有四处必要修改（原因见 README / 实验报告「问题与解决」）：

1. 设备：`gpu=True` 但机器没显卡时自动退回 CPU，不再硬编码 .cuda()；
2. 损失函数：默认 BCEWithLogitsLoss(pos_weight=n_neg/n_pos)。上游用
   NLLLoss + 类别权重 [n_pos, n_neg]，在自建特征上训练/测试 AUC 一直停在 0.50
   （等价随机），F-score 只有 0.21；换成 BCE 后 100 步即可拟合（见 logs/probe_*）；
   把 config.loss 设成 'nll' 可复现上游配方；
3. 监控：TensorBoard 里除了 Loss，还记录每 save_every 个 epoch 的
   Precision / Recall / F-score / AUC；
4. 权重保存：`torch.save` 到 models/，训练结束再存 fcsn_final.pkl。

用法：
  python train.py                          # 用 config.py 的默认配置
  python train.py --epochs 150 --lr 3e-3   # 临时覆盖
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from prettytable import PrettyTable
from tqdm import tqdm, trange

import eval as eval_utils
from fcsn import FCSN


def make_writer(log_dir):
    """建 TensorBoard writer。tensorboardX 不可用时退回 torch 自带实现（事件格式相同）。"""
    try:
        from tensorboardX import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception as exc:
        print('[WARN] tensorboardX SummaryWriter 不可用（%s: %s），改用 torch.utils.tensorboard'
              % (type(exc).__name__, exc))
        from torch.utils.tensorboard import SummaryWriter as TorchWriter
        return TorchWriter(log_dir=log_dir)


class Solver(object):
    """构建并训练 FCSN。"""

    def __init__(self, config=None, train_loader=None, test_dataset=None):
        self.config = config
        self.train_loader = train_loader
        self.test_dataset = test_dataset

        self.use_gpu = bool(self.config.gpu and torch.cuda.is_available())
        self.model = FCSN(self.config.n_class)
        if self.use_gpu:
            self.model = self.model.cuda()

        if self.config.mode == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.model.train()

        for d in (self.config.score_dir, self.config.save_dir, self.config.log_dir):
            os.makedirs(d, exist_ok=True)

    def compute_loss(self, pred_score, gt_labels):
        """pred_score: [B, 2, L]，gt_labels: [B, L]（0/1）。"""
        if self.config.loss == 'bce':
            logits = pred_score[:, 1, :].reshape(-1)
            target = gt_labels.reshape(-1).float()
            if self.config.pos_weight and self.config.pos_weight > 0:
                pw = torch.tensor(float(self.config.pos_weight))
            else:
                n_pos = float(target.mean())
                pw = torch.tensor((1.0 - n_pos) / max(n_pos, 1e-3))
            if self.use_gpu:
                pw = pw.cuda()
            return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

        # 上游原始配方
        n_batch, n_class, n_frame = pred_score.shape
        log_p = torch.log_softmax(pred_score, dim=1).reshape(-1, n_class)
        gt_labels = gt_labels.reshape(-1)
        label_1 = gt_labels.float().mean() * n_frame
        label_0 = n_frame - label_1
        weight = torch.tensor([float(label_1), float(label_0)])
        if self.use_gpu:
            weight = weight.cuda()
        return torch.nn.NLLLoss(weight)(log_p, gt_labels)

    @torch.no_grad()
    def predict(self, feature):
        self.model.eval()
        logits = self.model(feature.unsqueeze(0).cuda() if self.use_gpu else feature.unsqueeze(0))
        return torch.softmax(logits.squeeze(0), dim=0)[1]

    def train(self):
        writer = make_writer(self.config.log_dir)
        best = {'fscore': -1.0, 'epoch': -1}
        t = trange(self.config.n_epochs, desc='Epoch', ncols=80)
        for epoch_i in t:
            self.model.train()
            sum_loss_history = []

            for feature, label, _ in tqdm(self.train_loader, desc='Batch', ncols=80, leave=False):
                if self.use_gpu:
                    feature, label = feature.cuda(), label.cuda()

                pred_score = self.model(feature)
                loss = self.compute_loss(pred_score, label)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                sum_loss_history.append(loss.detach())

            mean_loss = torch.stack(sum_loss_history).mean().item()
            t.set_postfix(loss='%.4f' % mean_loss)
            writer.add_scalar('Loss/train', mean_loss, epoch_i)

            if (epoch_i + 1) % self.config.save_every == 0:
                ckpt_path = os.path.join(self.config.save_dir, 'epoch-{}.pkl'.format(epoch_i))
                torch.save(self.model.state_dict(), ckpt_path)
                tqdm.write('Save parameters at {}'.format(ckpt_path))

                eval_mean = self.evaluate(epoch_i)
                writer.add_scalar('Eval/Precision', eval_mean[0], epoch_i)
                writer.add_scalar('Eval/Recall', eval_mean[1], epoch_i)
                writer.add_scalar('Eval/F-score', eval_mean[2], epoch_i)
                writer.add_scalar('Eval/AUC', eval_mean[3], epoch_i)
                if eval_mean[2] > best['fscore']:
                    best = {'fscore': eval_mean[2], 'epoch': epoch_i}
                    torch.save(self.model.state_dict(),
                               os.path.join(self.config.save_dir, 'fcsn_best.pkl'))

        writer.close()
        print('最佳 F-score = %.4f（epoch %d）→ models/fcsn_best.pkl' % (best['fscore'], best['epoch']))

    @torch.no_grad()
    def evaluate(self, epoch_i):
        self.model.eval()
        out_dict = {}
        eval_arr = []
        scores, labels = [], []
        table = PrettyTable()
        table.title = 'Eval result of epoch {}'.format(epoch_i)
        table.field_names = ['ID', 'Precision', 'Recall', 'F-score']
        table.float_format = '1.3'

        with h5py.File(self.config.data_path, 'r') as data_file:
            for feature, label, idx in tqdm(self.test_dataset, desc='Evaluate', ncols=80, leave=False):
                if self.use_gpu:
                    feature = feature.cuda()
                pred_score = torch.softmax(self.model(feature.unsqueeze(0)).squeeze(0), dim=0)[1]
                scores.append(pred_score.cpu().numpy())
                labels.append(label.numpy())

                video_info = data_file['video_' + str(idx)]
                pred_score, pred_selected, pred_summary = eval_utils.select_keyshots(video_info, pred_score)
                true_summary_arr = video_info['user_summary'][()]
                eval_res = np.mean([eval_utils.eval_metrics(pred_summary, ts) for ts in true_summary_arr],
                                   axis=0).tolist()

                eval_arr.append(eval_res)
                table.add_row([idx] + eval_res)
                out_dict[str(idx)] = {
                    'pred_score': pred_score,
                    'pred_selected': [int(s) for s in pred_selected],
                    'pred_summary': [int(v) for v in pred_summary],
                }

        score_save_path = os.path.join(self.config.score_dir, 'epoch-{}.json'.format(epoch_i))
        with open(score_save_path, 'w', encoding='utf-8') as f:
            json.dump(out_dict, f)

        eval_mean = np.mean(eval_arr, axis=0).tolist()
        auc = rank_auc(np.concatenate(scores), np.concatenate(labels))
        table.add_row(['mean'] + ['%.3f' % v for v in eval_mean])
        tqdm.write(str(table))
        tqdm.write('  帧级 AUC = %.4f' % auc)
        return eval_mean + [auc]


def rank_auc(score, label):
    order = np.argsort(score)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(score))
    n1 = label.sum()
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((rank[label == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--loss', default=None, choices=['bce', 'nll'])
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--save_every', type=int, default=None)
    args = ap.parse_args()

    from config import Config
    from data_loader import get_loader

    overrides = {k: v for k, v in dict(n_epochs=args.epochs, lr=args.lr, loss=args.loss,
                                       batch_size=args.batch_size, save_every=args.save_every).items()
                 if v is not None}
    train_config = Config(**overrides)
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)

    print(train_config)
    print('CUDA 可用:', torch.cuda.is_available(), '| 设备:',
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

    train_loader, test_dataset, all_keys = get_loader(
        train_config.data_path, batch_size=train_config.batch_size, seed=train_config.seed)
    print('训练批量数: %d | 测试视频数: %d' % (len(train_loader), len(test_dataset)))
    test_ids = sorted([all_keys[i] for i in test_dataset.indices])
    print('测试视频(10):', test_ids)
    os.makedirs(os.path.dirname(os.path.join(train_config.log_dir, '..', 'test_ids.txt')) or '.', exist_ok=True)
    with open(os.path.join(train_config.log_dir, '..', 'test_ids.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_ids))

    solver = Solver(train_config, train_loader, test_dataset)
    solver.train()
    final_path = os.path.join(train_config.save_dir, 'fcsn_final.pkl')
    torch.save(solver.model.state_dict(), final_path)
    print('训练完成，最终权重:', final_path)
