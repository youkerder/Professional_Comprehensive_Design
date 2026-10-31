import torch
from torch.utils.data import DataLoader

import h5py


class VideoData(object):
    """Dataset class

    本实验的 fcsn_tvsum.h5 用「video_<youtube_id>」作为组名（原版 FCSN 仓库的 h5 是
    video_1..video_50 编号），所以这里按排序后的 group key 取数据，并把 youtube id
    作为 idx 返回；train.py / gen_summary.py 都能直接用 id 对应原视频文件。
    """

    def __init__(self, data_path):
        self.data_file = h5py.File(data_path, 'r')
        self.keys = sorted(self.data_file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        idx = key[len('video_'):]
        video = self.data_file[key]
        feature = torch.tensor(video['feature'][()]).t()
        label = torch.tensor(video['label'][()], dtype=torch.long)
        return feature, label, idx


def get_loader(path, batch_size=5, seed=0):
    """按 80% / 20% 固定随机划分训练集与测试集（测试集即实验要求的 10 段视频）。"""
    dataset = VideoData(path)
    torch.manual_seed(seed)
    n_test = len(dataset) // 5
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [len(dataset) - n_test, n_test])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_dataset, dataset.keys


if __name__ == '__main__':
    loader, test, keys = get_loader('fcsn_tvsum.h5')
    print('训练 batch 数:', len(loader), ' 测试视频数:', len(test))
