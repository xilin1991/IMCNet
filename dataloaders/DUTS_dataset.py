from __future__ import division

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import dataloaders.helpers as helpers


class DUTS(Dataset):
    """
    DUTS dataset
    """
    def __init__(self,
                 split='train',
                 num_frame=1,
                 step=1,
                 db_root_dir=None,
                 transform=None,
                 seq_name=None):
        super(DUTS, self).__init__()
        self.split = split
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.seq_name = seq_name
        self.num_frame = num_frame
        self.step = step
        self.center_frame = self.num_frame // 2

        if db_root_dir is None:
            raise ValueError('Dataset dir not given!')

        if seq_name is None:
            seqs = ['{}/DUTS-{}-'.format(db_root_dir, 'TR' if split == 'train' else 'TE')]
        else:
            seqs = [self.seq_name]
        imgs_list = []
        gts_list = []
        for seq in seqs:
            # Images
            images = np.sort(os.listdir(os.path.join(seq + 'Image')))
            images_path = list(map(lambda x: os.path.join(seq + 'Image', x), images))
            for idx, sample in enumerate(helpers.generator_sample(images_path, num_frame)):
                sample_elem = sample[0]
                sample.extend([sample_elem, sample_elem])
                imgs_list.append(tuple(sample))

            # Ground-Truth
            gts = np.sort(os.listdir(os.path.join(seq + 'Mask')))
            gts_path = list(map(lambda x: os.path.join(seq + 'Mask', x), gts))
            for idx, sample in enumerate(helpers.generator_sample(gts_path, num_frame)):
                sample_elem = sample[0]
                sample.extend([sample_elem, sample_elem])
                gts_list.append(tuple(sample))
        assert len(imgs_list) == len(gts_list)

        self.imgs_list = imgs_list
        self.gts_list = gts_list
        print('Done Initializing ' + self.split + ' Dataset')

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        num_frame = 3
        imgs = list(map(lambda x: Image.open(x).convert('RGB'), self.imgs_list[index]))
        gts = list(map(lambda x: Image.open(x).convert('L'), self.gts_list[index]))

        if self.seq_name:
            fname = os.path.join(self.seq_name, '%05d' % index)
            sample = {'num_frame': num_frame, 'imgs': imgs, 'gts': gts, 'fname': fname}
        else:
            sample = {'num_frame': num_frame, 'imgs': imgs, 'gts': gts, 'd_type': 'image_set'}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
