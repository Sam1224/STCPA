import os
import numpy as np
import torch
from base import BaseNetTrainer
from matplotlib import pyplot as plt
from utils import *
from utils.util import ensure_dir


plt.switch_backend('agg')


# ========================================
# BasicTrainer
# Speed Completion
# M_t => \hat{M}_t
# ========================================
class BasicTrainer(BaseNetTrainer):
    def __init__(self, models, optimizers, loss, metrics, resume, config,
                 train_data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):

        super(BasicTrainer, self).__init__(
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
        self.lr_scheduler = lr_scheduler["lr_scheduler"]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0.
        for batch_idx, (inputs, labels, masks) in enumerate(self.train_data_loader):
            inputs, labels, masks = inputs.to(self.device), labels.to(self.device), masks.to(self.device)

            x = inputs.float()
            m = masks.float()
            z = uniform_sampler(0, 0.01, bs=x.size(0), num_nodes=x.size(1))
            z = torch.from_numpy(z).float().to(self.device)
            x = m * x + (1 - m) * z

            self.optimizer.zero_grad()
            gen_prob = self.model(x, m)
            mse_loss = self.loss["mse"](gen_prob, labels, m)
            loss_sum = mse_loss
            loss_sum.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('gen_mse_loss', loss_sum.item())
            total_loss += loss_sum.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] MSE Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.train_data_loader.batch_size,
                        self.train_data_loader.n_samples,
                        100.0 * batch_idx / len(self.train_data_loader),
                        loss_sum.item()
                    ))

        log = {
            'gen_mse_loss': total_loss / len(self.train_data_loader),
        }

        val_log = self._valid_epoch(epoch)
        log = {**log, **val_log}

        test_log = self._test_epoch(epoch)
        log = {**log, **test_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(log["val_rmse"])

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.train_data_loader.dataset.val_x
            labels = self.train_data_loader.dataset.val_y
            masks = self.train_data_loader.dataset.val_w

            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            masks = torch.from_numpy(masks)
            inputs, labels, masks = inputs.to(self.device), labels.to(self.device), masks.to(self.device)
            x = inputs.float()
            m = masks.float()
            z = uniform_sampler(0, 0.01, bs=x.size(0), num_nodes=x.size(1))
            z = torch.from_numpy(z).float().to(self.device)
            x = m * x + (1 - m) * z

            gen_prob = self.model(x, m)

            self.writer.set_step((epoch - 1))

            val_mse = self.metrics(gen_prob, labels, masks, scale=None)
            val_rmse = torch.sqrt(val_mse)

        return {
            'val_mse': val_mse,
            'val_rmse': val_rmse
        }

    def _test_epoch(self, epoch):
        """
        Test after training an epoch
        :return: A log that contains information about test
        Note:
            The validation metrics in log must have the key 'test_xxx'.
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.train_data_loader.dataset.test_x
            labels = self.train_data_loader.dataset.test_y
            masks = self.train_data_loader.dataset.test_w

            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            masks = torch.from_numpy(masks)
            inputs, labels, masks = inputs.to(self.device), labels.to(self.device), masks.to(self.device)
            x = inputs.float()
            m = np.ones_like(masks.detach().cpu().numpy())
            mask_x = inputs.detach().cpu().numpy() == 0
            m[mask_x] = 0
            m = torch.from_numpy(m).float().to(self.device)
            z = uniform_sampler(0, 0.01, bs=x.size(0), num_nodes=x.size(1))
            z = torch.from_numpy(z).float().to(self.device)
            x = m * x + (1 - m) * z

            gen_prob = self.model(x, m)

            self.writer.set_step((epoch - 1))

            test_mse = self.metrics(gen_prob, labels, masks, scale=40)
            test_rmse = torch.sqrt(test_mse)

        return {
            'test_mse': test_mse,
            'test_rmse': test_rmse
        }
