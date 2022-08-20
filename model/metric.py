import numpy as np
import sys
import torch.nn.functional as F


def mse_metric(x, y, w, scale=None):
    # x => bs, num_nodes
    # y => bs, num_nodes
    # w => bs, num_nodes
    if scale is not None:
        x = x * scale
        y = y * scale
    unmasked_mse = F.mse_loss(x, y, reduction="none")
    masked_mse = (unmasked_mse * w.float()).sum(1)
    masked_mse = masked_mse / w.sum(1)
    return masked_mse.mean()
