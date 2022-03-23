import torch.nn as nn
from utils import WeightedBCE2d, IOU, SSIM


class Loss(nn.Module):
    def __init__(self, size_average=True, batch_average=True):
        super(Loss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.bce_loss = WeightedBCE2d(size_average=self.size_average, batch_average=self.batch_average)
        self.iou_loss = IOU(size_average=self.size_average, batch_average=self.batch_average)
        self.ssim_loss = SSIM(window_size=11, size_average=self.size_average, batch_average=self.batch_average)

    def forward(self, pred, target):
        bce_out = self.bce_loss(pred, target)
        iou_out = self.iou_loss(pred, target)
        ssim_out = 1 - self.ssim_loss(pred, target)

        final_loss = bce_out + iou_out + ssim_out
        # final_loss = bce_out

        return final_loss


class Loss_BCE(nn.Module):
    def __init__(self, size_average=True, batch_average=True):
        super(Loss_BCE, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.bce_loss = WeightedBCE2d(size_average=self.size_average, batch_average=self.batch_average)

    def forward(self, pred, target):
        bce_out = self.bce_loss(pred, target)

        return bce_out
