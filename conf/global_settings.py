from datetime import datetime

from dataloaders.DAVIS16_dataset import DAVIS2016
from dataloaders.YouTubeVOS_dataset import YouTubeVOS2019
from dataloaders.DUTS_dataset import DUTS
from dataloaders.YouTubeObj_dataset import YouTebeObj


# global parameters
DEVICE_ID = 0

# model paramenters
ARCH = 'resnet101'
PLANES = 64
NUM_FRAMES = 3
STEP = 4
NUM_CLASSES = 2

# training parameters
TR_BATCH_SIZES = 8
TS_BATCH_SIZES = 8
IN_BATCH_SIZES = 1

RESNET_CKPT_DIRS = {
    'resnet34': './checkpoints/pre/resnet34-best.pth',
    'resnet50': './checkpoints/pre/resnet50-best.pth',
    'resnet101': './checkpoints/pre/resnet101-best.pth',
}


TRAINCONF = {
    's1': {
        'dataset_list': ['davis2016', 'youtubevos2019'],
        'resume': False,
        'load_epoch': 0,
        'num_epochs': 100,
        'lr': 1e-4,
        'wd': 0,
        'momentum': 0.9,
        'frozen_layers': 'none',  # 'none' or ['conv1', 'bn1', ...]
        'checkpoints_dir': './checkpoints/s1',
        'runs_dir': './runs/s1',
        'load_best': False,
        'tr_batch_sizes': TR_BATCH_SIZES,
        'ts_batch_sizes': TS_BATCH_SIZES,
        'snapshot': 5,
        'use_test': True,
        'test_session': 2,
        'pre_trained': None,
    },
    's2': {
        'dataset_list': ['davis2016', 'duts'],
        'resume': False,
        'load_epoch': 0,
        'num_epochs': 100,
        'lr': 1e-5,
        'wd': 0,
        'momentum': 0.9,
        'frozen_layers': 'none',  # 'none' or ['conv1', 'bn1', ...]
        'checkpoints_dir': './checkpoints/s2',
        'runs_dir': './runs/s2',
        'load_best': False,
        'tr_batch_sizes': TR_BATCH_SIZES,
        'ts_batch_sizes': TS_BATCH_SIZES,
        'snapshot': 10,
        'use_test': True,
        'test_session': 2,
        'pre_trained': './checkpoints/s1/IMCNET_99.pth',
    },
    's3': {
        'dataset_list': ['davis2016'],
        'resume': False,
        'load_epoch': 0,
        'num_epochs': 200,
        'lr': 1e-4,
        'wd': 0,
        'momentum': 0.9,
        'frozen_layers': 'none',
        'checkpoints_dir': './checkpoints/s3',
        'runs_dir': './runs/s3',
        'load_best': False,
        'tr_batch_sizes': TR_BATCH_SIZES,
        'ts_batch_sizes': TS_BATCH_SIZES,
        'snapshot': 10,
        'use_test': True,
        'test_session': 2,
        'pre_trained': './checkpoints/s1/IMCNET_99.pth',
    }
}

INFERCONF = {
    'infer': {
        'dataset_list': ['davis2016', 'youtube_object'],
        'eval_dataset': 'davis2016',
        'frozen_layers': 'all',
        'runs_dir': './runs/infer',
        'in_batch_sizes': IN_BATCH_SIZES,
        'multi_scale': False,
        'multi_scale_size': [(352, 352), (480, 480), (736, 736)],
        'results_dir': './results',
        'pre_trained': './checkpoints/s2/IMCNET_best.pth',
    }
}


# dataset parameters
IMG_SIZE = (854, 480)
DATASET_CONF = {
    'davis2016': {
        'dataset_cls_name': DAVIS2016,
        'split': ['train', 'val', 'test'],
        'batch_size_tr': TR_BATCH_SIZES,
        'batch_size_ts': TS_BATCH_SIZES,
        'batch_size_in': IN_BATCH_SIZES,
        'db_root_dir': '/data0/xilin/DAVIS2017/',
        'std': (0.229, 0.224, 0.225),
        'mean': (0.485, 0.456, 0.406),
        'meanval': (104.00699, 116.66877, 122.67892),
        'num_classes': NUM_CLASSES,
        'input_size': (480, 480),  # W, H
        'multi_scale_size': [(352, 352), (480, 480), (736, 736)],
        'num_frame': NUM_FRAMES,
        'step': STEP
    },
    'youtubevos2019': {
        'dataset_cls_name': YouTubeVOS2019,
        'split': ['train', 'valid', 'test'],
        'batch_size_tr': TR_BATCH_SIZES,
        'batch_size_ts': TS_BATCH_SIZES,
        'db_root_dir': '/data0/xilin/YouTubeVOS/2019/',
        'std': (0.229, 0.224, 0.225),
        'mean': (0.485, 0.456, 0.406),
        'meanval': (104.00699, 116.66877, 122.67892),
        'num_classes': NUM_CLASSES,
        'input_size': (480, 480),  # W, H
        'num_frame': NUM_FRAMES,
        'step': STEP
    },
    'duts': {
        'dataset_cls_name': DUTS,
        'split': ['train', 'valid', 'test'],
        'batch_size_tr': TR_BATCH_SIZES,
        'batch_size_ts': TS_BATCH_SIZES,
        'db_root_dir': '/data3/SOD/DUTS/DUTS-TR/',
        'std': (0.229, 0.224, 0.225),
        'mean': (0.485, 0.456, 0.406),
        'meanval': (104.00699, 116.66877, 122.67892),
        'num_classes': NUM_CLASSES,
        'input_size': (480, 480),  # W, H
        'num_frame': 1,
        'step': STEP
    },
    'youtube_object': {
        'dataset_cls_name': YouTebeObj,
        'split': ['train', 'val', 'test'],
        'batch_size_tr': TR_BATCH_SIZES,
        'batch_size_ts': TS_BATCH_SIZES,
        'batch_size_in': IN_BATCH_SIZES,
        'db_root_dir': '/data0/xilin/YouTube-Objects/',
        'std': (0.229, 0.224, 0.225),
        'mean': (0.485, 0.456, 0.406),
        'meanval': (104.00699, 116.66877, 122.67892),
        'num_classes': NUM_CLASSES,
        'input_size': (288, 288),  # W, H
        'multi_scale_size': [(224, 224), (288, 288), (448, 448)],
        'num_frame': NUM_FRAMES,
        'step': STEP
    }
}


# Time parameters
# DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
DATE_FORMAT = '%b%d_%H-%M-%S'
# time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)
