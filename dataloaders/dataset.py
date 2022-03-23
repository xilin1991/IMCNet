from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from dataloaders import custom_transforms as ct
from dataloaders import custum_sampler as cs


class DatasetBuild(object):
    """
    """
    def __init__(self, dataset_list, dataset_conf, device='cuda:0'):
        super(DatasetBuild, self).__init__()
        self.dataset_list = dataset_list
        self.dataset_conf = dataset_conf
        self.device = device

    def get_single_dataset(self, type='davis2016', train=True, seq_name=None, multi=False):
        if type in self.dataset_list:
            if train:
                split = self.dataset_conf[type]['split'][0]
                batch_size = self.dataset_conf[type]['batch_size_tr']
                shuffle = True
                drop_last = True
                transform = transforms.Compose([ct.FixedResize(size=self.dataset_conf[type]['input_size']),
                                                ct.RandomHorizontalFlip(),
                                                ct.RandomRotation(),
                                                ct.ToTensor(),
                                                ct.Normalize(mean=self.dataset_conf[type]['mean'],
                                                             std=self.dataset_conf[type]['std'])])
            else:
                split = self.dataset_conf[type]['split'][1]
                if seq_name is None:
                    batch_size = self.dataset_conf[type]['batch_size_ts']
                else:
                    batch_size = self.dataset_conf[type]['batch_size_in']
                shuffle = False
                drop_last = False
                if multi:
                    transform = transforms.Compose([ct.ToTensor(),
                                                    ct.Normalize(mean=self.dataset_conf[type]['mean'],
                                                                 std=self.dataset_conf[type]['std'])])
                else:
                    transform = transforms.Compose([ct.FixedResize(size=self.dataset_conf[type]['input_size']),
                                                    ct.ToTensor(),
                                                    ct.Normalize(mean=self.dataset_conf[type]['mean'],
                                                                 std=self.dataset_conf[type]['std'])])
            db_root_dir = self.dataset_conf[type]['db_root_dir']
            num_frame = self.dataset_conf[type]['num_frame']
            step = self.dataset_conf[type]['step']
            db = self.dataset_conf[type]['dataset_cls_name'](split=split,
                                                             num_frame=num_frame,
                                                             step=step,
                                                             db_root_dir=db_root_dir,
                                                             transform=transform,
                                                             seq_name=seq_name)
            db_loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_last)
        else:
            print('Not complete this dataset code!')
        return db_loader

    def get_multi_dataset(self, dset_list=['davis2016'], train=True, seq_name=None):
        if len(dset_list) == 0:
            raise ValueError('Dataset must be provided!')
        elif len(dset_list) == 1:
            self.get_single_dataset(type=dset_list[0], train=train, seq_name=seq_name)
        else:
            batch_size = self.dataset_conf[dset_list[0]]['batch_size_tr']
            transform = transforms.Compose([ct.FixedResize(size=self.dataset_conf[dset_list[0]]['input_size']),
                                            ct.RandomHorizontalFlip(),
                                            ct.RandomRotation(),
                                            ct.ToTensor(),
                                            ct.Normalize(mean=self.dataset_conf[dset_list[0]]['mean'],
                                                         std=self.dataset_conf[dset_list[0]]['std'])])
            db_list = list(map(lambda x: self.dataset_conf[x]['dataset_cls_name'](split=self.dataset_conf[x]['split'][0],
                                                                                  num_frame=self.dataset_conf[x]['num_frame'],
                                                                                  step=self.dataset_conf[x]['step'],
                                                                                  db_root_dir=self.dataset_conf[x]['db_root_dir'],
                                                                                  transform=transform,
                                                                                  seq_name=seq_name), dset_list))
            db = ConcatDataset(db_list)
            if 'duts' in self.dataset_list:
                sampler = cs.CustomSampler(data_source=db, batch_size=batch_size)
                shuffle = False
            else:
                sampler = None
                shuffle = True
            drop_last = True
            db_loader = DataLoader(db, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=4, drop_last=drop_last)

            return db_loader
