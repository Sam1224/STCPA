import os
import numpy as np
import sys
import torch
import torch.nn as nn
from base import BaseSTGTrainer
from matplotlib import pyplot as plt
from utils import *
from utils.util import ensure_dir

plt.switch_backend('agg')


# ========================================
# STCPGIterativeRefine2GTrainer
# Speed Completion
# M_t => \hat{M}_t
# ========================================
class STCPGIterativeRefine2GTrainer(BaseSTGTrainer):
    def __init__(self, models, optimizers, loss, metrics, resume, config,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None,
                 gcn_batch=None):

        super(STCPGIterativeRefine2GTrainer, self).__init__(
            models,
            optimizers,
            loss=loss,
            metrics=metrics,
            resume=resume,
            config=config,
            train_logger=train_logger)

        self.config = config
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(train_data_loader.batch_size)) * 5

        # lr_scheduler
        self.fc_lr_scheduler = lr_scheduler["fc_lr_scheduler"]
        self.att_lr_scheduler = lr_scheduler["att_lr_scheduler"]

        # cycle consistency parameters
        self.cycle_num = self.config["cycle_num"] if self.config["cycle_num"] is not None else 0
        self.ratio = self.config["ratio"]
        if self.ratio == "mix":
            self.ratio = 0.55  # the missing ratios of mix dataset ranges from 0.4 - 0.7

        # perceptual loss parameters
        # perceptual loss
        self.gcn_model = models["gcn"]
        self.perceptual_weight = self.config["perceptual_weight"]
        self.perceptual_layer = self.config["perceptual_layer"]

        # gcn_batch is used to get the formatted edge_index and batch of Data (torch_geometric)
        self.gcn_batch = gcn_batch

        self.scale = 0
        if self.config["data"] == "chengdu":
            self.scale = 40
        elif self.config["data"] == "newyork_full":
            self.scale = 25

    def process_historical_data(self, x_his, x_his_his, m_hiss):
        # process historical matrices
        x_hiss = []
        for idx in range(self.config["n_temporal"]):
            his = x_his[:, idx, ...]
            m_his = m_hiss[:, idx, ...]

            # process his_his matrices, the historical matrices of each historical matrix
            his_his = x_his_his[:, idx + 1: idx + 1 + self.config["n_temporal"], ...]

            his_out = self.fc_model(his, his_his)
            his_out = m_his * his + (1 - m_his) * his_out
            x_hiss.append(his_out.detach().cpu().numpy())
        if len(x_hiss) > 0:
            x_his = np.array(x_hiss)
            x_his = torch.from_numpy(x_his).to(self.device).permute(1, 0, 2).detach()  # bs, n_temporal, num_nodes
        return x_his

    def cal_per_loss(self, per_inputs, per_labels):
        bs = per_inputs.size(0)
        edge_index = self.gcn_batch.edge_index.to(self.device)
        batch = self.gcn_batch.batch.to(self.device)
        per_inputs = per_inputs.view(-1, 1)  # the dim of bs and num_nodes are fused in torch_geometric
        per_labels = per_labels.view(-1, 1)
        per_inputs_fms = self.gcn_model(per_inputs, edge_index=edge_index, batch=batch)
        per_labels_fms = self.gcn_model(per_labels, edge_index=edge_index, batch=batch)

        # select a specific layer for calculation
        per_inputs_fm = per_inputs_fms[self.perceptual_layer - 1]
        per_labels_fm = per_labels_fms[self.perceptual_layer - 1]

        # when calculating the perceptual loss, do not need partial mask any more
        per_inputs_fm = per_inputs_fm.view(bs, -1)
        per_labels_fm = per_labels_fm.view(bs, -1)
        full_mask = torch.ones_like(per_inputs_fm).to(self.device)
        gen_perceptual_loss = self.loss["mse"](per_inputs_fm, per_labels_fm, full_mask)
        return gen_perceptual_loss

    def _train_epoch(self, epoch):
        self.fc_model.train()
        self.att_model.train()
        total_loss = 0.
        total_gen_fc_mse_loss = 0.
        total_gen_fc_per_loss = 0.
        total_gen_att_mse_loss = 0.
        total_gen_att_per_loss = 0.
        total_cyc_fc_mse_loss = 0.
        total_cyc_fc_per_loss = 0.
        total_cyc_att_mse_loss = 0.
        total_cyc_att_per_loss = 0.
        for batch_idx, data in enumerate(self.train_data_loader):
            inputs, labels, masks = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
            train_imputation = data[5].to(self.device)
            if self.config["n_temporal"] > 0:
                inputs_his = data[3].to(self.device)
                m_hiss = data[4].to(self.device)

                # x_his => normal n_temporal historical matrices
                x_his = inputs_his[:, :self.config["n_temporal"], ...].float()
                x_his_his = inputs_his.float()
            else:
                m_hiss = None
                x_his = None
                x_his_his = None

            x = inputs.float()
            m = masks.float()

            self.fc_optimizer.zero_grad()
            self.att_optimizer.zero_grad()

            # ========================================
            # 1st model - fc
            # ========================================
            gen_fc_prob = self.fc_model(x, x_his)

            # mse loss
            gen_fc_mse_loss = self.loss["mse"](gen_fc_prob, labels, m)

            # perceptual loss
            per_inputs = gen_fc_prob * m
            per_labels = labels * m
            gen_fc_per_loss = self.cal_per_loss(per_inputs, per_labels)
            gen_fc_loss_sum = self.config["alpha"] * (gen_fc_mse_loss + self.perceptual_weight * gen_fc_per_loss)

            # ========================================
            # 2nd model - att
            # ========================================
            x_his_2 = self.process_historical_data(x_his, x_his_his, m_hiss)

            gen_x_att = gen_fc_prob.detach()
            gen_x_att = m * x + (1 - m) * gen_x_att

            gen_att_prob = self.att_model(gen_x_att, x_his_2)

            # mse loss
            gen_att_mse_loss = self.loss["mse"](gen_att_prob, labels, m)

            # perceptual loss
            per_inputs = gen_att_prob * m
            per_labels = labels * m
            gen_att_per_loss = self.cal_per_loss(per_inputs, per_labels)
            gen_att_loss_sum = self.config["beta"] * (gen_att_mse_loss + self.perceptual_weight * gen_att_per_loss)

            gen_loss_sum = gen_fc_loss_sum + gen_att_loss_sum

            # ========================================
            # cycle consistency
            # ========================================
            cyc_fc_loss_sum = 0.
            cyc_att_loss_sum = 0.
            temp_cyc_fc_mse_loss = 0.  # temp => for display
            temp_cyc_fc_per_loss = 0.
            temp_cyc_att_mse_loss = 0.
            temp_cyc_att_per_loss = 0.
            for _ in range(self.cycle_num):
                # construct mask with the same ratio as the current dataset (only adopted during training)
                cycle_mask = binary_sampler(self.ratio, bs=x.size(0), num_nodes=x.size(1))
                cycle_mask = torch.from_numpy(cycle_mask).to(self.device)
                # create cycle inputs with same format as inputs x
                cycle_x = cycle_mask * gen_att_prob.detach() + (1 - cycle_mask) * train_imputation

                # ========================================
                # 1st model - fc
                # ========================================
                cyc_fc_prob = self.fc_model(cycle_x, x_his)

                # mse + perceptual loss
                if self.config["use_fake"]:
                    full_mask = torch.ones_like(m).to(self.device)

                    # mse loss
                    cyc_fc_mse_loss = self.loss["mse"](cyc_fc_prob, gen_fc_prob, full_mask)

                    # perceptual loss
                    per_inputs = cyc_fc_prob
                    per_labels = gen_fc_prob
                    cyc_fc_per_loss = self.cal_per_loss(per_inputs, per_labels)
                else:
                    # mse loss
                    cyc_fc_mse_loss = self.loss["mse"](cyc_fc_prob, labels, m)

                    # perceptual loss
                    per_inputs = cyc_fc_prob * m
                    per_labels = labels * m
                    cyc_fc_per_loss = self.cal_per_loss(per_inputs, per_labels)

                cyc_fc_loss_sum += self.config["alpha"] * (cyc_fc_mse_loss + self.perceptual_weight * cyc_fc_per_loss)
                temp_cyc_fc_mse_loss += cyc_fc_mse_loss.item()
                temp_cyc_fc_per_loss += cyc_fc_per_loss.item()

                # ========================================
                # 2nd model - att
                # ========================================
                cyc_x_att = cyc_fc_prob.detach()

                # mse + perceptual loss
                if self.config["use_fake"]:
                    cyc_x_att = cycle_mask * gen_att_prob.detach() + (1 - cycle_mask) * cyc_x_att
                    cyc_att_prob = self.att_model(cyc_x_att, x_his_2)

                    full_mask = torch.ones_like(m).to(self.device)

                    # mse loss
                    cyc_att_mse_loss = self.loss["mse"](cyc_att_prob, gen_att_prob, full_mask)

                    # perceptual loss
                    per_inputs = cyc_att_prob
                    per_labels = gen_att_prob
                    cyc_att_per_loss = self.cal_per_loss(per_inputs, per_labels)
                else:
                    cyc_x_att = m * x + (1 - m) * cyc_x_att
                    cyc_att_prob = self.att_model(cyc_x_att, x_his_2)

                    # mse loss
                    cyc_att_mse_loss = self.loss["mse"](cyc_att_prob, labels, m)

                    # perceptual loss
                    per_inputs = cyc_att_prob * m
                    per_labels = labels * m
                    cyc_att_per_loss = self.cal_per_loss(per_inputs, per_labels)

                cyc_att_loss_sum += self.config["beta"] * (cyc_att_mse_loss + self.perceptual_weight * cyc_att_per_loss)
                temp_cyc_att_mse_loss += cyc_att_mse_loss.item()
                temp_cyc_att_per_loss += cyc_att_per_loss.item()
            cyc_fc_loss_sum /= self.cycle_num if self.cycle_num != 0 else 1
            cyc_att_loss_sum /= self.cycle_num if self.cycle_num != 0 else 1
            temp_cyc_fc_mse_loss /= self.cycle_num if self.cycle_num != 0 else 1
            temp_cyc_fc_per_loss /= self.cycle_num if self.cycle_num != 0 else 1
            temp_cyc_att_mse_loss /= self.cycle_num if self.cycle_num != 0 else 1
            temp_cyc_att_per_loss /= self.cycle_num if self.cycle_num != 0 else 1

            cyc_loss_sum = cyc_fc_loss_sum + cyc_att_loss_sum

            loss_sum = gen_loss_sum + self.config["cycle_weight"] * cyc_loss_sum
            loss_sum.backward()
            self.fc_optimizer.step()
            self.att_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('total_loss', loss_sum.item())
            self.writer.add_scalar('gen_fc_mse_loss', gen_fc_mse_loss.item())
            self.writer.add_scalar('gen_fc_per_loss', gen_fc_per_loss.item())
            self.writer.add_scalar('gen_att_mse_loss', gen_att_mse_loss.item())
            self.writer.add_scalar('gen_att_per_loss', gen_att_per_loss.item())
            self.writer.add_scalar('cyc_fc_mse_loss', temp_cyc_fc_mse_loss)
            self.writer.add_scalar('cyc_fc_per_loss', temp_cyc_fc_per_loss)
            self.writer.add_scalar('cyc_att_mse_loss', temp_cyc_att_mse_loss)
            self.writer.add_scalar('cyc_att_per_loss', temp_cyc_att_per_loss)
            total_loss += (gen_fc_loss_sum + gen_att_loss_sum).item()
            total_gen_fc_mse_loss += gen_fc_mse_loss.item()
            total_gen_fc_per_loss += gen_fc_per_loss.item()
            total_gen_att_mse_loss += gen_att_mse_loss.item()
            total_gen_att_per_loss += gen_att_per_loss.item()
            total_cyc_fc_mse_loss += temp_cyc_fc_mse_loss
            total_cyc_fc_per_loss += temp_cyc_fc_per_loss
            total_cyc_att_mse_loss += temp_cyc_att_mse_loss
            total_cyc_att_per_loss += temp_cyc_att_per_loss

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Total Loss: {:.4f} '
                    'Gen: [FC MSE Loss: {:.4f} FC Per Loss: {:.4f} '
                    'Att MSE Loss: {:.4f} Att Per Loss: {:.4f}] '
                    'Cyc: [FC MSE Loss: {:.4f} FC Per Loss: {:.4f} '
                    'Att MSE Loss: {:.4f} Att Per Loss: {:.4f}]'.format(
                        epoch,
                        batch_idx * self.train_data_loader.batch_size,
                        self.train_data_loader.n_samples,
                        100.0 * batch_idx / len(self.train_data_loader),
                        loss_sum.item(),
                        gen_fc_mse_loss.item(),
                        gen_fc_per_loss.item(),
                        gen_att_mse_loss.item(),
                        gen_att_per_loss.item(),
                        temp_cyc_fc_mse_loss,
                        temp_cyc_fc_per_loss,
                        temp_cyc_att_mse_loss,
                        temp_cyc_att_per_loss
                    ))

        log = {
            'total_loss': total_loss / len(self.train_data_loader),
            'gen_fc_mse_loss': total_gen_fc_mse_loss / len(self.train_data_loader),
            'gen_fc_per_loss': total_gen_fc_per_loss / len(self.train_data_loader),
            'gen_att_mse_loss': total_gen_att_mse_loss / len(self.train_data_loader),
            'gen_att_per_loss': total_gen_att_per_loss / len(self.train_data_loader),
            'cyc_fc_mse_loss': total_cyc_fc_mse_loss / len(self.train_data_loader),
            'cyc_fc_per_loss': total_cyc_fc_per_loss / len(self.train_data_loader),
            'cyc_att_mse_loss': total_cyc_att_mse_loss / len(self.train_data_loader),
            'cyc_att_per_loss': total_cyc_att_per_loss / len(self.train_data_loader),
        }

        val_log = self._valid_epoch(epoch)
        log = {**log, **val_log}

        test_log = self._test_epoch(epoch)
        log = {**log, **test_log}

        if self.fc_lr_scheduler is not None:
            self.fc_lr_scheduler.step(log["val_rmse_fc"])

        # if self.att_lr_scheduler is not None and epoch >= 20:
        if self.att_lr_scheduler is not None:
            self.att_lr_scheduler.step(log["val_rmse_att"])

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.fc_model.eval()
        self.att_model.eval()
        with torch.no_grad():
            inputs = self.train_data_loader.dataset.val_x
            labels = self.train_data_loader.dataset.val_y
            masks = self.train_data_loader.dataset.val_w
            if self.config["n_temporal"] > 0:
                inputs_his = self.train_data_loader.dataset.val_x_his
                inputs_his = torch.from_numpy(inputs_his)
                inputs_his = inputs_his.to(self.device)

                m_hiss = self.train_data_loader.dataset.val_w_his
                m_hiss = torch.from_numpy(m_hiss).float().to(self.device)

                x_his = inputs_his[:, :self.config["n_temporal"], ...].float()
                x_his_his = inputs_his.float()
            else:
                m_hiss = None
                x_his = None
                x_his_his = None

            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            masks = torch.from_numpy(masks)
            inputs, labels, masks = inputs.to(self.device), labels.to(self.device), masks.to(self.device)

            x = inputs.float()
            m = masks.float()

            fc_prob = self.fc_model(x, x_his)

            x_his_2 = self.process_historical_data(x_his, x_his_his, m_hiss)
            x_att = fc_prob
            x_att = m * x + (1 - m) * x_att
            att_prob = self.att_model(x_att, x_his_2)

            self.writer.set_step((epoch - 1))

            val_mse_fc = self.metrics(fc_prob, labels, masks, scale=None)
            val_rmse_fc = torch.sqrt(val_mse_fc)
            val_mse_att = self.metrics(att_prob, labels, masks, scale=None)
            val_rmse_att = torch.sqrt(val_mse_att)

        return {
            'val_mse_fc': val_mse_fc,
            'val_rmse_fc': val_rmse_fc,
            'val_mse_att': val_mse_att,
            'val_rmse_att': val_rmse_att
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch
        :return: A log that contains information about test
        Note:
            The validation metrics in log must have the key 'test_xxx'.
        """
        self.fc_model.eval()
        self.att_model.eval()
        with torch.no_grad():
            inputs = self.train_data_loader.dataset.test_x
            labels = self.train_data_loader.dataset.test_y
            masks = self.train_data_loader.dataset.test_w
            if self.config["n_temporal"] > 0:
                inputs_his = self.train_data_loader.dataset.test_x_his
                inputs_his = torch.from_numpy(inputs_his)
                inputs_his = inputs_his.to(self.device)

                m_hiss = self.train_data_loader.dataset.test_w_his
                m_hiss = torch.from_numpy(m_hiss).float().to(self.device)

                x_his = inputs_his[:, :self.config["n_temporal"], ...].float()
                x_his_his = inputs_his.float()
            else:
                m_hiss = None
                x_his = None
                x_his_his = None

            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            masks = torch.from_numpy(masks)
            inputs, labels, masks = inputs.to(self.device), labels.to(self.device), masks.to(self.device)

            x = inputs.float()
            m = 1 - masks.float()

            fc_prob = self.fc_model(x, x_his)

            x_his_2 = self.process_historical_data(x_his, x_his_his, m_hiss)
            x_att = fc_prob
            x_att = m * x + (1 - m) * x_att
            att_prob = self.att_model(x_att, x_his_2)

            self.writer.set_step((epoch - 1))

            test_mse_fc = self.metrics(fc_prob, labels, masks, scale=self.scale)
            test_rmse_fc = torch.sqrt(test_mse_fc)
            test_mse_att = self.metrics(att_prob, labels, masks, scale=self.scale)
            test_rmse_att = torch.sqrt(test_mse_att)

        return {
            'test_mse_fc': test_mse_fc,
            'test_rmse_fc': test_rmse_fc,
            'test_mse_att': test_mse_att,
            'test_rmse_att': test_rmse_att
        }

    def test(self):
        return self._test_epoch(0)
