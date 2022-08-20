import torch
from base import BaseDataLoader
from data_loader.datasets import *


# ========================================
# Speed DataLoader
# ========================================
class SpeedDataLoader(BaseDataLoader):
    """
    Dataset module
    Args:
        mode (str): train, val or test
        data (str): chengdu, newyork
        ratio (float): 0.4, 0.5, 0.6, 0.7 or 0.8
        fold (int): 4
        batch_size (int): e.g. 64
        shuffle (bool): True or False
    """
    def __init__(self, mode="train", data="chengdu", ratio=0.5, fold=4, batch_size=32, n_temporal=0, data_aug=0,
                 get_his_his=False, scale=False, return_m_his=False, shuffle=True, validation_split=0.0, num_workers=0):
        assert mode in ["train", "val", "test"], "Invalid mode, please choose one from train, val or test"
        assert data in ["chengdu", "newyork_full"], \
            "Invalid traffic dataset, please choose one from chengdu, newyork or london"
        assert ratio in ["mix", 0.0, 0.4, 0.5, 0.6, 0.7, 0.8], "Invalid ratio, please choose one from 0.4, 0.5, 0.6, 0.7 or 0.8"
        assert fold in [0, 1, 2, 3, 4], "Invalid fold, please choose one from 0, 1, 2, 3 or 4"
        torch.manual_seed(123)
        np.random.seed(123)
        self.mode = mode
        self.data = data
        self.ratio = ratio
        self.fold = fold
        self.n_temporal = n_temporal
        self.data_aug = data_aug
        self.get_his_his = get_his_his
        self.scale = scale
        self.return_m_his = return_m_his

        dataset = SpeedDataset(mode=mode, data=data, ratio=ratio, fold=fold, n_temporal=n_temporal, data_aug=data_aug,
                               get_his_his=get_his_his, scale=scale, return_m_his=return_m_his)
        self.dataset = dataset
        if mode == "train":
            drop_last = True
        else:
            drop_last = False
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, drop_last=drop_last, mode=mode)
