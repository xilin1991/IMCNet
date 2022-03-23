from __future__ import division

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

import dataloaders.helpers as helpers


class DAVIS2016(Dataset):
    """
    DAVIS 2016 dataset
    """
    def __init__(self,
                 split='train',
                 num_frame=3,
                 step=4,
                 db_root_dir=None,
                 transform=None,
                 seq_name=None):
        super(DAVIS2016, self).__init__()
        self.split = split
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.seq_name = seq_name
        self.num_frame = num_frame
        self.center_frame = self.num_frame // 2
        self.step = step

        if db_root_dir is None:
            raise ValueError('Dataset dir not given!')

        if self.seq_name is None:
            with open(os.path.join(db_root_dir, 'ImageSets/2016', split + '.txt')) as f:
                seqs = f.readlines()
        else:
            seqs = [self.seq_name]
        imgs_list = []
        gts_list = []
        for seq in seqs:
            # Images
            images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p', seq.strip())))
            images_path = list(map(lambda x: os.path.join(db_root_dir, 'JPEGImages/480p', seq.strip(), x), images))
            for idx, sample in enumerate(helpers.generator_sample_step(images_path, num_frame, step)):
                imgs_list.append(tuple(sample))

            # Ground-Truth
            gts = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations_unsupervised/480p/', seq.strip())))
            gts_path = list(map(lambda x: os.path.join(db_root_dir, 'Annotations_unsupervised/480p/', seq.strip(), x), gts))
            for idx, sample in enumerate(helpers.generator_sample_step(gts_path, num_frame, step)):
                gts_list.append(tuple(sample))
        assert len(imgs_list) == len(gts_list)

        self.imgs_list = imgs_list
        self.gts_list = gts_list
        print('Done Initializing ' + self.split + ' Dataset')

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        num_frame = self.num_frame
        imgs = list(map(lambda x: Image.open(x).convert('RGB'), self.imgs_list[index]))
        gts = list(map(lambda x: Image.open(x).convert('L'), self.gts_list[index]))

        if self.seq_name:
            fname = os.path.join(self.seq_name, '%05d' % index)
            img_size = gts[num_frame // 2].size
            sample = {'num_frame': num_frame, 'imgs': imgs, 'gts': gts, 'fname': fname,
                      'img_size': img_size, 'd_type': 'video_set'}
        else:
            sample = {'num_frame': num_frame, 'imgs': imgs, 'gts': gts, 'd_type': 'video_set'}

        if self.transform is not None:
            sample = self.transform(sample)
        return sample
