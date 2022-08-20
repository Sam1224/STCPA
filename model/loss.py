import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# ========================================
# mse_loss
# ========================================
def mse_loss(x, y, w):
    unmasked_mse = F.mse_loss(x, y, reduction="none")
    masked_mse = (unmasked_mse * w.float()).sum(1)
    masked_mse = masked_mse / w.sum(1)
    return masked_mse.mean()


# ========================================
# mean_mse_loss
# ========================================
def mean_mse_loss(x, y, w, lambd=None):
    unmasked_mse = F.mse_loss(x, y, reduction="none")
    if lambd is not None:
        masked_mse = (unmasked_mse * w.float() * lambd.float()).mean()
    else:
        masked_mse = (unmasked_mse * w.float()).mean()
    masked_mse = masked_mse / w.mean()
    return masked_mse
