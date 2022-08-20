import os
import math
import json
import logging
import datetime
import torch
from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX


class BaseNetTrainer:
    """
    Base class for trainers
    """

    def __init__(self, models, optimizers, loss, metrics, resume, config, train_logger=None):
        """
        class initialization
        :param models: models dictionary contains net model
        :param optimizers: optimizers dictionary contains optimizers
        :param loss: loss dictionary contains the loss objectives
        :param metrics: other metrics except for the loss want to display during training
        :param resume: resume checkpoint
        :param config: config file
        :param train_logger: logger
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device = torch.device("cuda:{}".format(config["device"]))
        self.model = models["gen"].to(self.device)

        self.loss = loss
        self.metrics = metrics
        self.train_logger = train_logger
        self.optimizer = optimizers["gen_optimizer"]

        # read training settings
        self.epochs = config["epoch"]
        self.save_period = 10
        self.verbosity = 2
        self.monitor = "min val_rmse"

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = config["early_stop"]

        self.start_epoch = 1

        # setup directory for checkpoint saving
        self.checkpoint_dir = os.path.join("saved/", config["name"])
        # setup visualization writer instance
        writer_dir = os.path.join("saved/runs", config["name"])
        self.writer = WriterTensorboardX(writer_dir, self.logger, True)

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        self.config['checkpoint_dir'] = self.checkpoint_dir
        config_save_path = os.path.join(self.checkpoint_dir, '{}.json'.format(config['name']))
        with open(config_save_path, 'w') as handle:
            json.dump(config, handle, indent=4, sort_keys=False)

        if resume:
            self._resume_checkpoint(resume)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        print("have {} gpu".format(n_gpu))
        print("use {} gpu".format(n_gpu_use))
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):

            # save logged informations into log dict
            log = {'epoch': epoch}
            result = self._train_epoch(epoch)
            for key, value in result.items():
                log[key] = value

            # print logged informations to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    self._save_checkpoint(epoch, save_best=best)
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn\'t improve for {} epochs. Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        return float(self.mnt_best.cpu().numpy())

    def _train_net_epoch(self, epoch):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
