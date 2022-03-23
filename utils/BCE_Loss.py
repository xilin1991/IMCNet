import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# BCE Loss
class WeightedBCE2d(nn.Module):
    """
    Weighted Binary Cross Entropy loss for Video object segmentation
    """
    def __init__(self, size_average=True, batch_average=True):
        super(WeightedBCE2d, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, input, target, negative=None):
        if negative is None:
            negative = torch.zeros_like(input)
        target = torch.ge(target, 0.5).float()

        num_label_pos = torch.sum(target)
        num_label_neg = torch.sum(1.0 - target)
        num_total = num_label_pos + num_label_neg

        loss_val = -F.binary_cross_entropy(input=input, target=target, reduction='none')

        loss_pos = torch.sum(-torch.mul(target, loss_val))
        weight = torch.mul((1.0 - target), (1.0 + negative))
        loss_neg = torch.sum(-torch.mul(weight, loss_val))

        final_loss = num_label_pos / num_total * loss_pos + num_label_neg / num_total * loss_neg

        if self.size_average:
            final_loss /= np.prod(target.size())
            # final_loss /= torch.prod(target.size())
        elif self.batch_average:
            final_loss /= target.size()[0]

        return final_loss
