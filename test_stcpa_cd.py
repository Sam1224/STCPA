import os
import argparse
import numpy as np
import torch
import copy
from torch import distributions
from data_loader.data_loaders import *
from data_loader.datasets import *
from model.loss import *
from model.metric import *
from model.model import *
from model.gcn import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from trainer import STCPGIterativeRefine2GTrainer
from utils import *
import logging
import nni
from nni.utils import merge_parameter

logger = Logger()


# ========================================
# spatio-temporal cycle-perceptual generator
# two stcpg
# ========================================
def main(args):
    # ========================================
    # DataLoader
    # ========================================
    get_his_his = True
    train_data_loader = SpeedDataLoader(mode="train", data=args["data"], ratio=args["ratio"], fold=args["fold"],
                                        batch_size=args["batch_size"], n_temporal=args["n_temporal"],
                                        get_his_his=get_his_his, return_m_his=True, shuffle=True, num_workers=8)

    # Calculate mean value of train_x, val_x, test_x
    train_x = train_data_loader.dataset.train_x.copy()
    train_x[train_x == 0] = np.nan
    train_mean = np.nanmean(train_x, axis=0)

    val_x = train_data_loader.dataset.val_x.copy()
    val_x[val_x == 0] = np.nan
    val_mean = np.nanmean(val_x, axis=0)

    test_x = train_data_loader.dataset.test_x.copy()
    test_x[test_x == 0] = np.nan
    test_mean = np.nanmean(test_x, axis=0)

    means = [train_mean, train_mean, train_mean]

    # also construct an auxiliary dataloader for torch_geometric to construct edge_index, etc.
    edge_index = train_data_loader.dataset.edge_index
    train_x = train_data_loader.dataset.train_x
    train_list = []
    for idx, x in enumerate(train_x):
        x = torch.FloatTensor(x)
        train_list.append(Data(x=x, edge_index=edge_index))

    # shuffle does not matter, because we only use edge_index and batch of Data (torch_geometric)
    gcn_data_loader = DataLoader(train_list, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    gcn_iterator = iter(gcn_data_loader)
    gcn_batch = next(gcn_iterator)

    # ========================================
    # Device
    # ========================================
    device = torch.device("cuda:{}".format(args["device"]))

    # ========================================
    # Model
    # ========================================
    # num_nodes => num of road segments
    num_nodes = int(train_data_loader.dataset.train_x[0].shape[0])

    # pretrained gcn => for perceptual loss
    gcn_model = ChebNet(num_nodes, 1, device).to(device)
    sample_rate = 15
    if args["data"] == "chengdu":
        sample_rate = 15  # 15, 30, 45, 60 - manually be consistent with load_matrix method
    elif "newyork" in args["data"]:
        sample_rate = 60
    pretrained_weights = "./pretrained_model/{}/{}_1/rm0.5/pretrained_gcn.pth".format(args["data"], sample_rate)
    gcn_model.load_state_dict(torch.load(pretrained_weights, map_location="cuda:{}".format(args["device"])))
    gcn_model.eval()

    # ========================================
    # Trainer
    # ========================================
    losses = {}
    losses["mse"] = mean_mse_loss

    metrics = mse_metric

    resume = None
    config = args

    fc_model = STGAIN_Att(num_nodes=num_nodes, n_blocks=args["n_blocks"], n_temporal=args["n_temporal"],
                          device=device).to(device)
    att_model = STGAIN_Att(num_nodes=num_nodes, n_blocks=args["n_blocks"], n_temporal=args["n_temporal"],
                           device=device).to(device)

    fc_optimizer = torch.optim.Adam(fc_model.parameters(), lr=args["fc_lr"])
    att_optimizer = torch.optim.Adam(att_model.parameters(), lr=args["att_lr"])
    fc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fc_optimizer,
                                                                 mode="min", factor=0.2, patience=10,
                                                                 verbose=True)
    att_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(att_optimizer,
                                                                  mode="min", factor=0.2, patience=10,
                                                                  verbose=True)

    models = {}
    models["fc"] = fc_model
    models["att"] = att_model
    models["gcn"] = gcn_model

    optimizers = {}
    optimizers["fc_optimizer"] = fc_optimizer
    optimizers["att_optimizer"] = att_optimizer

    lr_schedulers = {}
    lr_schedulers["fc_lr_scheduler"] = fc_lr_scheduler
    lr_schedulers["att_lr_scheduler"] = att_lr_scheduler

    # ========================================
    # Mean Imputation
    # ========================================
    mean_imputation(train_data_loader, means)

    # ========================================
    # Update historical data
    # ========================================
    update_his(train_data_loader, get_his_his, args)

    # ========================================
    # Update best model parameters
    # ========================================
    best_model_path = "./saved/STCPA_CD_best/model_best.pth".format(idx + 1)
    trainer = STCPGIterativeRefine2GTrainer(
        models=models,
        optimizers=optimizers,
        loss=losses,
        metrics=metrics,
        resume=best_model_path,
        config=config,
        train_data_loader=train_data_loader,
        lr_scheduler=lr_schedulers,
        train_logger=logger,
        gcn_batch=gcn_batch
    )
    test_rmse = trainer.test()["test_rmse_att"]
    print("Iteration 0, Best Test RMSE: {:.4f}".format(test_rmse))


def update_his(train_data_loader, get_his_his, args):
    n_temporal = args["n_temporal"]
    if n_temporal > 0 and get_his_his:
        n_temporal = 2 * n_temporal

    if n_temporal > 0:
        train_x = train_data_loader.dataset.train_x
        val_x = train_data_loader.dataset.val_x
        test_x = train_data_loader.dataset.test_x

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

        train_data_loader.dataset.train_x_his = train_x_his
        train_data_loader.dataset.val_x_his = val_x_his
        train_data_loader.dataset.test_x_his = test_x_his


def model_imputation(train_data_loader, fc_model, att_model, device, args):
    bs = 256

    # Train set
    inputs = train_data_loader.dataset.train_x
    masks = train_data_loader.dataset.train_w
    inputs_his = train_data_loader.dataset.train_x_his
    m_hiss = train_data_loader.dataset.train_w_his

    imputations = np.zeros_like(inputs)
    train_len = inputs.shape[0]
    for i in range(train_len // bs):
        input = inputs[bs * i: bs * (i + 1), ...]
        mask = masks[bs * i: bs * (i + 1), ...]
        m_his = m_hiss[bs * i: bs * (i + 1), ...]
        input_his = inputs_his[bs * i: bs * (i + 1), ...]
        imputations[bs * i: bs * (i + 1), ...] = update_dataset(inputs=input, masks=mask, m_hiss=m_his,
                                                                inputs_his=input_his,
                                                                fc_model=fc_model, att_model=att_model, device=device,
                                                                args=args)
    if bs * (i + 1) < train_len:
        input = inputs[bs * (i + 1):, ...]
        mask = masks[bs * (i + 1):, ...]
        m_his = m_hiss[bs * (i + 1):, ...]
        input_his = inputs_his[bs * (i + 1):, ...]
        imputations[bs * (i + 1):, ...] = update_dataset(inputs=input, masks=mask, m_hiss=m_his, inputs_his=input_his,
                                                         fc_model=fc_model, att_model=att_model, device=device,
                                                         args=args)

    train_x = inputs * masks + imputations * (1 - masks)
    train_data_loader.dataset.train_x = train_x
    train_data_loader.dataset.train_imputation = imputations

    # Valid set
    inputs = train_data_loader.dataset.val_x
    masks = train_data_loader.dataset.val_w
    inputs_his = train_data_loader.dataset.val_x_his
    m_hiss = train_data_loader.dataset.val_w_his
    imputation = update_dataset(inputs=inputs, masks=masks, m_hiss=m_hiss, inputs_his=inputs_his,
                                fc_model=fc_model, att_model=att_model, device=device, args=args)
    val_x = inputs * masks + imputation * (1 - masks)
    train_data_loader.dataset.val_x = val_x

    # Test set
    inputs = train_data_loader.dataset.test_x
    masks = 1 - train_data_loader.dataset.test_w
    inputs_his = train_data_loader.dataset.test_x_his
    m_hiss = train_data_loader.dataset.test_w_his
    imputation = update_dataset(inputs=inputs, masks=masks, m_hiss=m_hiss, inputs_his=inputs_his,
                                fc_model=fc_model, att_model=att_model, device=device, args=args)
    test_x = inputs * masks + imputation * (1 - masks)
    train_data_loader.dataset.test_x = test_x


def mean_imputation(train_data_loader, means):
    # Train set
    inputs = train_data_loader.dataset.train_x
    imputation = fill_in_by_mean(inputs, means, which_set=0)
    train_data_loader.dataset.train_x = imputation
    train_mean = means[0]
    train_mean = np.expand_dims(train_mean, axis=0).repeat(inputs.shape[0], axis=0)
    train_data_loader.dataset.train_imputation = train_mean

    # Valid set
    inputs = train_data_loader.dataset.val_x
    imputation = fill_in_by_mean(inputs, means, which_set=1)
    train_data_loader.dataset.val_x = imputation

    # Test set
    inputs = train_data_loader.dataset.test_x
    imputation = fill_in_by_mean(inputs, means, which_set=2)
    train_data_loader.dataset.test_x = imputation


def update_dataset(inputs, masks, m_hiss, inputs_his, fc_model, att_model, device, args):
    inputs = torch.from_numpy(inputs).to(device)
    masks = torch.from_numpy(masks).to(device)
    m_hiss = torch.from_numpy(m_hiss).to(device)
    inputs_his = torch.from_numpy(inputs_his).to(device)
    x_his = inputs_his[:, :args["n_temporal"], ...].float()
    x_his_his = inputs_his.float()

    x = inputs.float()
    m = masks.float()

    fc_prob = fc_model(x, x_his)

    x_his_2 = process_historical_data(x_his, x_his_his, m_hiss, fc_model, device, args)
    x_att = fc_prob
    x_att = m * x + (1 - m) * x_att
    att_prob = att_model(x_att, x_his_2)
    return att_prob.detach().cpu().numpy()


def process_historical_data(x_his, x_his_his, m_hiss, fc_model, device, args):
    x_hiss = []
    for idx in range(args["n_temporal"]):
        his = x_his[:, idx, ...]
        m_his = m_hiss[:, idx, ...]

        his_his = x_his_his[:, idx + 1: idx + 1 + args["n_temporal"], ...]

        his_out = fc_model(his, his_his)
        his_out = m_his * his + (1 - m_his) * his_out
        x_hiss.append(his_out.detach().cpu().numpy())
    if len(x_hiss) > 0:
        x_his = np.array(x_hiss)
        x_his = torch.from_numpy(x_his).to(device).permute(1, 0, 2).detach().to(torch.float32)
    return x_his


def fill_in_by_mean(x, means, which_set):
    # which_set:
    # 0 =>  train
    # 1 =>  valid
    # 2 =>  test
    x[x == 0] = np.nan
    inds = np.where(np.isnan(x))
    x[inds] = np.take(means[which_set], inds[-1])
    return x


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='STCPG AutoML')
    parser.add_argument("--device", default=0, type=int, help="which gpu to use")
    parser.add_argument("--data", default="chengdu", type=str,
                        choices=["chengdu", "newyork_full"],
                        help="which dataset to use")
    parser.add_argument("--ratio", default=0.5,
                        choices=[0.4, 0.5, 0.6, 0.7, 0.8], help="which remove ratio to use")
    parser.add_argument("--fold", default=4, type=int,
                        choices=[0, 1, 2, 3, 4], help="which data fold to use (split train/test)")
    parser.add_argument("--batch_size", default=64, type=int, help="train batch size")
    parser.add_argument("--fc_lr", default=1e-3, type=float, help="lr of gen model")
    parser.add_argument("--att_lr", default=1e-3, type=float, help="lr of gen model")
    parser.add_argument("--name", default="STCPG_Iterative_Refine_2G", type=str, help="model name")
    parser.add_argument("--early_stop", default=50, type=int, help="epochs of early stop")
    parser.add_argument("--epoch", default=2000, type=int, help="epochs")
    parser.add_argument("--alpha", default=5000, type=int, help="weights of mse loss")
    parser.add_argument("--n_blocks", default=3, type=int, help="number of attention blocks")
    parser.add_argument("--n_temporal", default=2, type=int, help="number of historical matrices")
    parser.add_argument("--beta", default=10000, help="weights of recon loss")
    parser.add_argument("--cycle_num", default=1, type=int, help="number of cycle consistency")  # optimal
    parser.add_argument("--cycle_weight", default=1, help="weight of cycle consistency")  # optimal
    parser.add_argument("--use_fake", default=True, type=bool,
                        help="whether the 2nd output is supervised on the 1st output, or partial input")
    parser.add_argument("--perceptual_weight", default=0.001, help="weights of perceptual loss")  # optimal
    parser.add_argument("--perceptual_layer", default=1, type=int, choices=[1, 2, 3],
                        help="which layer of gcn to be calculated")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    try:
        SEED = 123
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        print(exception)
        raise
