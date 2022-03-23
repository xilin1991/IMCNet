import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            out_planes = in_planes
        if in_planes == out_planes:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class KeyGenerator(nn.Module):
    def __init__(self, indim, keydim):
        super(KeyGenerator, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, x):
        return self.Key(x)
