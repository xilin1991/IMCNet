import random
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as F


class FixedResize(object):
    def __init__(self, size):
        super(FixedResize, self).__init__()
        self.size = size

    def __call__(self, sample):
        for key, value in sample.items():
            if key not in ['num_frame', 'fname', 'd_type', 'img_size']:
                for idx in range(sample['num_frame']):
                    if key == 'imgs':
                        sample[key][idx] = value[idx].resize(self.size, Image.BILINEAR)
                    else:
                        sample[key][idx] = value[idx].resize(self.size, Image.NEAREST)
        return sample


class RandomHorizontalFlip(object):
    """
    Random horizontal flip augment
    """
    def __init__(self, prob=None):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def __call__(self, sample):
        if self.prob is None:
            self.prob = random.random()
        if self.prob >= 0.5:
            for key, value in sample.items():
                if key not in ['num_frame', 'fname', 'd_type', 'img_size']:
                    for idx in range(sample['num_frame']):
                        sample[key][idx] = value[idx].transpose(Image.FLIP_LEFT_RIGHT)

        return sample


class RandomRotation(object):
    """
    Random rotation augment
    """
    def __init__(self):
        super(RandomRotation, self).__init__()
        self.mode = Image.BICUBIC

    def __call__(self, sample):
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            for key, value in sample.items():
                if key not in ['num_frame', 'fname', 'd_type', 'img_size']:
                    for idx in range(sample['num_frame']):
                        sample[key][idx] = value[idx].rotate(random_angle, self.mode)

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W
    """
    def __call__(self, sample):
        for key, value in sample.items():
            if key not in ['num_frame', 'fname', 'd_type', 'img_size']:
                if key == 'imgs':
                    if sample['d_type'] == 'video_set':
                        tmp = list(
                            torch.from_numpy(
                                np.array(value[idx], dtype=np.float32).transpose(
                                    (2, 0, 1)) / 255.0).float()
                            for idx in range(sample['num_frame']))
                        sample[key] = torch.stack(tmp, dim=0)
                    else:
                        tmp = list(F.to_tensor(value[idx]) for idx in range(sample['num_frame']))
                        sample[key] = torch.stack(tmp, dim=0)
                else:
                    if sample['d_type'] == 'video_set':
                        tmp = list(np.array(value[idx], dtype=np.int32) for idx in range(sample['num_frame']))
                        for idx in range(len(tmp)):
                            tmp[idx][tmp[idx] != 0] = 1
                            tmp[idx] = torch.from_numpy(np.expand_dims(tmp[idx], axis=0)).float()
                        sample[key] = torch.stack(tmp, dim=0)
                    else:
                        tmp = list(F.to_tensor(value[idx]) for idx in range(sample['num_frame']))
                        sample[key] = torch.stack(tmp, dim=0)
        return sample


class Normalize(object):
    """
    Normalize ndarrays in samples
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        for key, value in sample.items():
            if key == 'imgs':
                t, c, h, w = value.size()
                for idx in range(t):
                    value[idx, :, :, :] = F.normalize(value[idx, :, :, :], mean=self.mean, std=self.std)
                sample[key] = value

        return sample
