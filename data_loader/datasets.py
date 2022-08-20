import os
import numpy as np
import torch
import sys
import pickle as pkl
from utils.util import uniform_sampler
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from sklearn.preprocessing import StandardScaler

dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)


class SpeedDataset(Dataset):
    """
    Dataset module
    Args:
        mode (str): train, val or test
        data (str): chengdu, newyork
        ratio (float): 0.4, 0.5, 0.6, 0.7 or 0.8
        fold (int): 4
        n_temporal (int)
        data_aug (int): 0, 1, 2, 3, 4
    """

    def __init__(self, mode="train", data="chengdu", ratio=0.5, fold=4, n_temporal=0, data_aug=0, get_his_his=False,
                 scale=False, return_m_his=False):
        # Scalers:
        #    chengdu => 0m/s - 40m/s  meter/second
        #    newyork => 0m/h - 40m/h  mile/hour
        assert mode in ["train", "val", "test"], "Invalid mode, please choose one from train, val or test"
        assert data in ["chengdu", "newyork_full"], \
            "Invalid traffic dataset, please choose one from chengdu, newyork or london"
        assert ratio in ["mix", 0.0, 0.4, 0.5, 0.6, 0.7, 0.8], "Invalid ratio, please choose one from 0.4, 0.5, 0.6, 0.7 or 0.8"
        assert fold in [0, 1, 2, 3, 4], "Invalid fold, please choose one from 0, 1, 2, 3 or 4"
        assert data_aug in [0, 1, 2, 3, 4], "Invalid data augmentation ways, pleas choose one from 0, 1, 2, 3, 4"
        self.mode = mode
        self.data = data
        self.ratio = ratio
        self.fold = fold
        self.n_temporal = n_temporal
        # data_aug:
        # 0 => do not use data augmentation
        # 1 => x + gau_noise * mask / y + gau_noise * mask / mask
        # 2 => x + gau_noise * mask / y (original) / mask
        # 3 => lambda * x1 + (1 - lambda) * x2 / lambda * y1 + (1 - lambda) * y2 / lambda * mask1 + (1 - lambda) * mask2
        # 4 => lambda * x1 + (1 - lambda) * x2 / y1 & y2 / mask1 & mask2
        self.data_aug = data_aug
        # get_his_his:
        # True  =>  get the historical matrices of each historical matrix (2 * n_temporal)
        # False =>  just get historical matrices (n_temporal)
        self.get_his_his = get_his_his
        # scale:
        # True  =>  use standard scaler
        # False =>  use min/max scaler
        self.scale = scale
        self.return_m_his = return_m_his

        self.train_imputation = None

        if n_temporal > 0:
            if return_m_his:
                self.train_x, self.train_y, self.train_w, self.val_x, self.val_y, self.val_w,\
                self.test_x, self.test_y, self.test_w,\
                self.train_x_his, self.train_w_his,\
                self.val_x_his, self.val_w_his,\
                self.test_x_his, self.test_w_his,\
                self.edge_index, self.scaler = \
                    load_matrix(data, ratio=ratio, fold=fold, return_edge_index=True, n_temporal=n_temporal,
                                get_his_his=get_his_his, scale=scale, return_m_his=return_m_his)
            else:
                self.train_x, self.train_y, self.train_w, self.val_x, self.val_y, self.val_w, self.test_x, self.test_y, \
                self.test_w, self.train_x_his, self.val_x_his, self.test_x_his, self.edge_index, self.scaler = \
                    load_matrix(data, ratio=ratio, fold=fold, return_edge_index=True, n_temporal=n_temporal,
                                get_his_his=get_his_his, scale=scale)
        else:
            self.train_x, self.train_y, self.train_w, self.val_x, self.val_y, self.val_w, self.test_x, self.test_y, \
            self.test_w, self.edge_index, self.scaler = \
                load_matrix(data, ratio=ratio, fold=fold, return_edge_index=True, n_temporal=n_temporal, scale=scale)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_x)
        elif self.mode == "val":
            return len(self.val_x)
        elif self.mode == "test":
            return len(self.test_x)

    def __getitem__(self, idx):
        if self.n_temporal > 0:
            if self.mode == "train":
                if self.return_m_his:
                    x = self.train_x[idx]
                    y = self.train_y[idx]
                    w = self.train_w[idx]
                    x_his = self.train_x_his[idx]
                    w_his = self.train_w_his[idx]

                    if self.train_imputation is not None:
                        imp = self.train_imputation[idx]
                        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w),\
                               torch.Tensor(x_his), torch.Tensor(w_his), torch.Tensor(imp)
                    else:
                        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w),\
                               torch.Tensor(x_his), torch.Tensor(w_his)
                else:
                    x = self.train_x[idx]
                    y = self.train_y[idx]
                    w = self.train_w[idx]
                    x_his = self.train_x_his[idx]

                    if self.data_aug == 0:
                        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w), torch.Tensor(x_his)
                    elif self.data_aug == 1:
                        uniform_noise = uniform_sampler(-0.1, 0.1, 1, x.shape[0]).squeeze(0)
                        x = x + uniform_noise * w
                        y = y + uniform_noise * w
                        x = np.clip(x, 0, 1)
                        y = np.clip(y, 0, 1)
                        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w), torch.Tensor(x_his)
                    elif self.data_aug == 2:
                        uniform_noise = uniform_sampler(-0.1, 0.1, 1, x.shape[0]).squeeze(0)
                        x = x + uniform_noise * w
                        x = np.clip(x, 0, 1)
                        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w), torch.Tensor(x_his)
                    elif self.data_aug == 3:
                        x1 = x
                        y1 = y
                        w1 = w

                        idx2 = np.random.choice(len(self.train_x))
                        x2 = self.train_x[idx2]
                        y2 = self.train_y[idx2]
                        w2 = self.train_w[idx2]

                        lambd = np.random.beta(1, 1)
                        x = lambd * x1 + (1 - lambd) * x2
                        y = lambd * y1 + (1 - lambd) * y2
                        w = lambd * w1 + (1 - lambd) * w2
                        return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w), torch.Tensor(x_his)
                    elif self.data_aug == 4:
                        x1 = x
                        y1 = y
                        w1 = w

                        idx2 = np.random.choice(len(self.train_x))
                        x2 = self.train_x[idx2]
                        y2 = self.train_y[idx2]
                        w2 = self.train_w[idx2]

                        lambd = np.random.beta(1, 1)
                        x = lambd * x1 + (1 - lambd) * x2
                        return torch.Tensor(x), torch.Tensor(y1), torch.Tensor(w1),\
                               torch.Tensor(y2), torch.Tensor(w2), lambd, torch.Tensor(x_his)
            elif self.mode == "val":
                return torch.Tensor(self.val_x[idx]), torch.Tensor(self.val_y[idx]), torch.Tensor(self.val_w[idx]), \
                       torch.Tensor(self.val_x_his[idx])
            elif self.mode == "test":
                return torch.Tensor(self.test_x[idx]), torch.Tensor(self.test_y[idx]), torch.Tensor(self.test_w[idx]), \
                       torch.Tensor(self.test_x_his[idx])
        else:
            if self.mode == "train":
                x = self.train_x[idx]
                y = self.train_y[idx]
                w = self.train_w[idx]

                if self.data_aug == 0:
                    return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w)
                elif self.data_aug == 1:
                    uniform_noise = uniform_sampler(-0.1, 0.1, 1, x.shape[0]).squeeze(0)
                    x = x + uniform_noise * w
                    y = y + uniform_noise * w
                    x = np.clip(x, 0, 1)
                    y = np.clip(y, 0, 1)
                    return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w)
                elif self.data_aug == 2:
                    uniform_noise = uniform_sampler(-0.1, 0.1, 1, x.shape[0]).squeeze(0)
                    x = x + uniform_noise * w
                    x = np.clip(x, 0, 1)
                    return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w)
                elif self.data_aug == 3:
                    x1 = x
                    y1 = y
                    w1 = w

                    idx2 = np.random.choice(len(self.train_x))
                    x2 = self.train_x[idx2]
                    y2 = self.train_y[idx2]
                    w2 = self.train_w[idx2]

                    lambd = np.random.beta(1, 1)
                    x = lambd * x1 + (1 - lambd) * x2
                    y = lambd * y1 + (1 - lambd) * y2
                    w = lambd * w1 + (1 - lambd) * w2
                    return torch.Tensor(x), torch.Tensor(y), torch.Tensor(w)
                elif self.data_aug == 4:
                    x1 = x
                    y1 = y
                    w1 = w

                    idx2 = np.random.choice(len(self.train_x))
                    x2 = self.train_x[idx2]
                    y2 = self.train_y[idx2]
                    w2 = self.train_w[idx2]

                    lambd = np.random.beta(1, 1)
                    x = lambd * x1 + (1 - lambd) * x2
                    return torch.Tensor(x), torch.Tensor(y1), torch.Tensor(w1), \
                           torch.Tensor(y2), torch.Tensor(w2), lambd
            elif self.mode == "val":
                return torch.Tensor(self.val_x[idx]), torch.Tensor(self.val_y[idx]), torch.Tensor(self.val_w[idx])
            elif self.mode == "test":
                return torch.Tensor(self.test_x[idx]), torch.Tensor(self.test_y[idx]), torch.Tensor(self.test_w[idx])


def load_matrix(dataset, ratio=0.5, fold=4, return_edge_index=False, n_temporal=0, get_his_his=False, scale=False,
                return_m_his=False):
    # Only matter STCPG (G1=G2=Att)
    # if get_his_his is True, get double historical matrices for imputing historical matrices for 2nd att model,
    # used in the 1st att model.
    if n_temporal > 0 and get_his_his:
        n_temporal = 2 * n_temporal

    if dataset == 'chengdu':
        start_months = [10, 10, 10, 11, 11]
        start_dates = [1, 13, 25, 6, 18]
        end_months = [10, 10, 11, 11, 11]
        end_dates = [13, 25, 6, 18, 30]

        test_period = "{}_{}-{}_{}".format(start_dates[fold], start_months[fold], end_dates[fold], end_months[fold])
        base_dir = "../GraphCompletion/data/{}".format(dataset)

        if return_edge_index:
            adj_path = "{}/{}".format(base_dir, "edge_adj.pickle")
            adj = pkl.load(open(adj_path, "rb"))
            edge_index, edge_weight = from_scipy_sparse_matrix(adj)
        else:
            edge_index = None

        data_dir = "{}/15_1/estimation/avg/{}".format(base_dir, test_period)
        data_dir = "{}/rm{}".format(data_dir, ratio)
        train_data_dict_file = "{}/train_data_dict.pickle".format(data_dir)
        test_data_dict_file = "{}/validate_data_dict.pickle".format(data_dir)

        train_data_dict = pkl.load(open(train_data_dict_file, "rb"))
        test_data_dict = pkl.load(open(test_data_dict_file, "rb"))

        train_vel_xs = train_data_dict["velocity_x"].squeeze(-1)
        train_vel_ys = train_data_dict["velocity_y"].squeeze(-1)
        train_weights = train_data_dict["weight_y"]

        train_len = len(train_vel_xs)
        val_ratio = 0.2
        val_num = int(train_len * val_ratio)
        train_num = train_len - val_num
        train_vel_x, val_vel_x = train_vel_xs[:train_num, ...], train_vel_xs[train_num:, ...]
        train_vel_y, val_vel_y = train_vel_ys[:train_num, ...], train_vel_ys[train_num:, ...]
        train_weight, val_weight = train_weights[:train_num, ...], train_weights[train_num:, ...]

        # current, test_vel_y and test_weight contain all known part,
        # test_vel_x is regulated to a specific rm ratio.
        # ideally, we only care about unknown part in test_vel_x, but available in test_vel_y
        test_vel_x = test_data_dict["velocity_x"].squeeze(-1)
        test_vel_y = test_data_dict["velocity_y"].squeeze(-1)
        test_weight = test_data_dict["weight_y"]

        # When testing, we ignore the known part (input),
        # but only test on the unknown part with ground truth.
        mask_w = test_weight == 0
        mask_x = test_vel_x != 0
        w = np.ones_like(test_weight)
        w[mask_x] = 0
        w[mask_w] = 0

        # n_temporal => number of historical matrix
        # we uniformly set the missing historical matrices to zero matrice, e.g., the historical matrices of the first matrix are not available.
        if n_temporal > 0:
            train_vel_x_his = []
            val_vel_x_his = []
            test_vel_x_his = []
            for train_idx in range(len(train_vel_x)):
                his = []
                for n_t in range(1, n_temporal + 1):
                    idx = train_idx - n_t
                    if idx < 0:
                        his.append(np.zeros_like(train_vel_x[train_idx, ...]))
                    else:
                        his.append(train_vel_x[idx, ...])
                his = np.array(his)
                train_vel_x_his.append(his)

            for val_idx in range(len(val_vel_x)):
                his = []
                for n_t in range(1, n_temporal + 1):
                    idx = val_idx - n_t
                    if idx < 0:
                        his.append(np.zeros_like(val_vel_x[val_idx, ...]))
                    else:
                        his.append(val_vel_x[idx, ...])
                his = np.array(his)
                val_vel_x_his.append(his)

            for test_idx in range(len(test_vel_x)):
                his = []
                for n_t in range(1, n_temporal + 1):
                    idx = test_idx - n_t
                    if idx < 0:
                        his.append(np.zeros_like(test_vel_x[test_idx, ...]))
                    else:
                        his.append(test_vel_x[idx, ...])
                his = np.array(his)
                test_vel_x_his.append(his)

            train_vel_x_his = np.array(train_vel_x_his)
            val_vel_x_his = np.array(val_vel_x_his)
            test_vel_x_his = np.array(test_vel_x_his)

            if return_m_his:
                train_m_his = np.zeros_like(train_vel_x_his)
                val_m_his = np.zeros_like(val_vel_x_his)
                test_m_his = np.zeros_like(test_vel_x_his)

                train_m_his[train_vel_x_his != 0] = 1
                val_m_his[val_vel_x_his != 0] = 1
                test_m_his[test_vel_x_his != 0] = 1

                return train_vel_x, train_vel_y, train_weight, val_vel_x, val_vel_y, val_weight,\
                       test_vel_x, test_vel_y, w, train_vel_x_his, train_m_his,\
                       val_vel_x_his, val_m_his, test_vel_x_his, test_m_his, edge_index, None
            else:
                return train_vel_x, train_vel_y, train_weight, val_vel_x, val_vel_y, val_weight, \
                       test_vel_x, test_vel_y, w, train_vel_x_his, val_vel_x_his, test_vel_x_his, edge_index, None

        return train_vel_x, train_vel_y, train_weight, val_vel_x, val_vel_y, val_weight, \
               test_vel_x, test_vel_y, w, edge_index, None
    elif dataset == 'newyork_full':
        base_dir = "../GraphCompletion/data/{}".format(dataset)

        if return_edge_index:
            adj_path = "{}/{}".format(base_dir, "newyork_adj.npy")
            adj = np.load(adj_path)
            edge_index = adj
            edge_index = torch.LongTensor(edge_index)
        else:
            edge_index = None

        data_dir = "{}/rm{}".format(base_dir, ratio)
        train_x_file = "{}/newyork_train_x.npy".format(data_dir)
        train_y_file = "{}/newyork_train_y.npy".format(data_dir)
        train_m_file = "{}/newyork_train_m.npy".format(data_dir)
        test_x_file = "{}/newyork_test_x.npy".format(data_dir)
        test_y_file = "{}/newyork_test_y.npy".format(data_dir)
        test_m_file = "{}/newyork_test_m.npy".format(data_dir)

        mph_to_mps = 0.44704
        train_xs = np.load(train_x_file) * mph_to_mps
        train_ys = np.load(train_y_file) * mph_to_mps
        train_ms = np.load(train_m_file)
        test_x = np.load(test_x_file) * mph_to_mps
        test_y = np.load(test_y_file) * mph_to_mps
        test_m = np.load(test_m_file)

        # scaler => 25m/s
        if not scale:
            max_val = 25
            train_xs = train_xs / max_val
            train_ys = train_ys / max_val
            test_x = test_x / max_val
            test_y = test_y / max_val

        train_len = len(train_xs)
        val_ratio = 0.2
        val_num = int(train_len * val_ratio)
        train_num = train_len - val_num
        train_x, val_x = train_xs[:train_num, ...], train_xs[train_num:, ...]
        train_y, val_y = train_ys[:train_num, ...], train_ys[train_num:, ...]
        train_m, val_m = train_ms[:train_num, ...], train_ms[train_num:, ...]

        if scale:
            x = np.concatenate([train_x, val_x, test_x], axis=0)

            # Standard Scaler
            scaler = StandardScaler()
            scaler.fit(x)

            train_x = scaler.transform(train_x)
            train_y = scaler.transform(train_y)

            val_x = scaler.transform(val_x)
            val_y = scaler.transform(val_y)

            test_x = scaler.transform(test_x)
            test_y = scaler.transform(test_y)
        else:
            scaler = None

        # n_temporal => number of historical matrix
        # we uniformly set the missing historical matrices to zero matrice, e.g., the historical matrices of the first matrix are not available.
        if n_temporal > 0:
            train_x_his = []
            val_x_his = []
            test_x_his = []
            for train_idx in range(len(train_x)):
                his = []
                for n_t in range(1, n_temporal + 1):
                    idx = train_idx - n_t
                    if idx < 0:
                        his.append(np.zeros_like(train_x[train_idx, ...]))
                    else:
                        his.append(train_x[idx, ...])
                his = np.array(his)
                train_x_his.append(his)

            for val_idx in range(len(val_x)):
                his = []
                for n_t in range(1, n_temporal + 1):
                    idx = val_idx - n_t
                    if idx < 0:
                        his.append(np.zeros_like(val_x[val_idx, ...]))
                    else:
                        his.append(val_x[idx, ...])
                his = np.array(his)
                val_x_his.append(his)

            for test_idx in range(len(test_x)):
                his = []
                for n_t in range(1, n_temporal + 1):
                    idx = test_idx - n_t
                    if idx < 0:
                        his.append(np.zeros_like(test_x[test_idx, ...]))
                    else:
                        his.append(test_x[idx, ...])
                his = np.array(his)
                test_x_his.append(his)

            train_x_his = np.array(train_x_his)
            val_x_his = np.array(val_x_his)
            test_x_his = np.array(test_x_his)

            if return_m_his:
                train_m_his = np.zeros_like(train_x_his)
                val_m_his = np.zeros_like(val_x_his)
                test_m_his = np.zeros_like(test_x_his)

                train_m_his[train_x_his != 0] = 1
                val_m_his[val_x_his != 0] = 1
                test_m_his[test_x_his != 0] = 1

                return train_x, train_y, train_m, val_x, val_y, val_m, test_x, test_y, test_m,\
                       train_x_his, train_m_his, val_x_his, val_m_his, test_x_his, test_m_his,\
                       edge_index, scaler

            else:
                return train_x, train_y, train_m, val_x, val_y, val_m, \
                       test_x, test_y, test_m, train_x_his, val_x_his, test_x_his, edge_index, scaler

        return train_x, train_y, train_m, val_x, val_y, val_m, \
                   test_x, test_y, test_m, edge_index, scaler
