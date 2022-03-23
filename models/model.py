import numpy as np
import torch

from .network import IMCNet


class ModelBuild(object):
    """
    """
    def __init__(self,
                 arch='resnet50',
                 frozen_layers=(),
                 pretrained=True,
                 ckpt_dir=None,
                 planes=64,
                 num_frames=3,
                 with_fusion=True,
                 num_classes=2,
                 device='cuda:0'):
        super(ModelBuild, self).__init__()
        self.arch = arch
        self.frozen_layers = frozen_layers
        self.pretrained = pretrained
        self.ckpt_dir = ckpt_dir
        self.planes = planes
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.with_fusion = with_fusion
        self.device = device

    def get_model(self, model_path=None):
        model = IMCNet(arch=self.arch,
                       frozen_layers=self.frozen_layers,
                       pretrained=self.pretrained,
                       ckpt_dir=self.ckpt_dir,
                       planes=self.planes,
                       num_frame=self.num_frames,
                       with_fusion=self.with_fusion,
                       num_classes=self.num_classes)

        total_parameters = self.get_params_num(model)
        print('Total network parameters: ' + str(total_parameters) + ' million')
        if model_path is not None:
            model = self.load_pretrained(model, model_path=model_path)
        model = model.to(self.device)

        return model

    def get_params_num(self, model):
        """
        Computing total network parameters
        Args:
            net: model
        return: total network parameters
        """
        total_parameters = 0
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        total_parameters = sum([np.prod(p.size()) for p in model_parameters])
        return (1.0 * total_parameters / (1000 * 1000))

    def get_params(self, model):
        num = 0
        print(len(list(model.parameters())))
        for name, module in model.named_children():
            print('children module:', name)
            num += len(list(module.parameters()))
        print(num)
        return

    def load_pretrained(self, model=None, model_path=None):
        """
        Use load pretrained model
        """
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
            if 'model' in ckpt.keys():
                model.load_state_dict(ckpt['model'])
        return model
