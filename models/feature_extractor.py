import torch
import torch.nn as nn
from torchvision.models import resnet


fext_arch = {'resnet18': resnet.resnet18,
             'resnet34': resnet.resnet34,
             'resnet50': resnet.resnet50,
             'resnet101': resnet.resnet101,
             'resnet152': resnet.resnet152}


class FeatureExtractor(nn.Module):
    """"""
    def __init__(self,
                 arch='resnet18',
                 frozen_layers=(),
                 pretrained=True,
                 ckpt_dir=None):
        super(FeatureExtractor, self).__init__()
        self.backbone = fext_arch[arch](pretrained=pretrained)
        del self.backbone.avgpool, self.backbone.fc
        if ckpt_dir is None:
            print('Model load PyTorch pretrained parameters!')
        else:
            print('Model load parameters which pretrained SOD task!')
            ckpt = torch.load(ckpt_dir, map_location=lambda storage, loc: storage)
            self.backbone.load_state_dict(ckpt)

        if isinstance(frozen_layers, str):
            if frozen_layers.lower() == 'none':
                frozen_layers = ()
            elif frozen_layers.lower() != 'all':
                raise ValueError('Unknown option for frozen layers \"{}\". Should be \"all\", \"none\" or list of layer names.')

        self.frozen_layers = frozen_layers
        self._is_frozen_nograd = False

    def train(self, mode=True):
        super().train(mode)
        if mode is True:
            self._set_frozen_to_eval()
        if not self._is_frozen_nograd:
            self._set_frozen_to_nograd()
            self._is_frozen_nograd = True
        return self

    def _set_frozen_to_eval(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            self.backbone.eval()
        else:
            for layer in self.frozen_layers:
                getattr(self.backbone, layer).eval()

    def _set_frozen_to_nograd(self):
        if isinstance(self.frozen_layers, str) and self.frozen_layers.lower() == 'all':
            for p in self.parameters():
                p.requires_grad_(False)
        else:
            for layer in self.frozen_layers:
                for p in getattr(self.backbone, layer).parameters():
                    p.requires_grad_(False)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        out = dict()
        # lyaer #0
        L0 = self.backbone.layer1(x)
        out['l0'] = L0

        L1 = self.backbone.layer2(L0)
        out['l1'] = L1

        L2 = self.backbone.layer3(L1)
        out['l2'] = L2

        L3 = self.backbone.layer4(L2)
        out['l3'] = L3

        return out
