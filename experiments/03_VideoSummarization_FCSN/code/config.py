import os
import pprint

# 实验根目录：.../experiments/03_VideoSummarization_FCSN
EXP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 工作区根目录（由实验目录向上两级推导，无需硬编码绝对路径）
WS_ROOT = os.path.dirname(os.path.dirname(EXP_ROOT))
# 数据集目录（已从桌面拷贝到工作区）
DATA_ROOT = os.path.join(WS_ROOT, 'datasets', 'tvsum50')


class Config():
    """Config class"""

    def __init__(self, **kwargs):

        # Path
        self.data_path = os.path.join(DATA_ROOT, 'fcsn_tvsum.h5')
        self.save_dir = os.path.join(EXP_ROOT, 'models')
        self.score_dir = os.path.join(EXP_ROOT, 'outputs', 'scores')
        self.log_dir = os.path.join(EXP_ROOT, 'logs', 'tensorboard')

        # Model
        self.mode = 'train'
        self.gpu = True
        self.n_epochs = 150
        self.n_class = 2
        self.lr = 3e-3
        self.momentum = 0.9
        self.batch_size = 5
        self.save_every = 10      # 每 N 个 epoch 存权重 + 评估
        self.seed = 0
        # 损失函数：
        #   'bce' —— BCEWithLogitsLoss(pos_weight=n_neg/n_pos)，本实验最终采用的配方；
        #   'nll' —— 上游原始实现（NLLLoss + 类别权重 [n_pos, n_neg]），在自建特征上不收敛，
        #            只用于对照，见 logs/probe_*。
        self.loss = 'bce'
        # pos_weight <= 0 表示按 batch 内正负样本比例自动计算（n_neg / n_pos）。
        self.pos_weight = 0.0

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        config_str = 'Configurations\n' + pprint.pformat(self.__dict__)
        return config_str


if __name__ == '__main__':
    config = Config()
    print(config)
