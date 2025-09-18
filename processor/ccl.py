import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
# from collections import OrderedDict
# from graph_model import VisualGraph, TextualGraph


class CCL(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, tau=0.1, method='log', q=0.5, ratio=0):
        super(CCL, self).__init__()
        self.tau = tau
        self.method = method
        self.q = q
        self.ratio = ratio

    def forward(self, scores):
        eps = 1e-10
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        randn, eye = torch.rand_like(scores), torch.eye(scores.shape[0]).cuda()
        randn[eye > 0] = randn.min(dim=1)[0] - 1
        n = scores.shape[0]
        num = n - 1 if self.ratio <= 0 or self.ratio >= 1 else int(self.ratio * n)
        V, K = randn.topk(num, dim=1)
        mask = torch.zeros_like(scores)
        mask[torch.arange(n).reshape([-1, 1]).cuda(), K] = 1.

        if self.method == 'log':
            criterion = lambda x: -((1. - x + eps).log() * mask).sum(1).mean()
        elif self.method == 'tan':
            criterion = lambda x: (x.tan() * mask).sum(1).mean()
        elif self.method == 'abs':
            criterion = lambda x: (x * mask).sum(1).mean()
        elif self.method == 'exp':
            criterion = lambda x: ((-(1. - x)).exp() * mask).sum(1).mean()
        elif self.method == 'gce':
            criterion = lambda x: ((1. - (1. - x + eps) ** self.q) / self.q * mask).sum(1).mean()
        elif self.method == 'infoNCE':
            criterion = lambda x: -x.diag().log().mean()
        else:
            raise Exception('Unknown Loss Function!')
        return criterion(i2t) + criterion(t2i)