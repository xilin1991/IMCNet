import os
# import socket

import argparse
import torch

from conf import settings
from utils import Solver
from dataloaders import DatasetBuild
from models import ModelBuild


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def get_arguments():
    """
    Parse all the arguments provided from the CLI
    Return:
        a list of parsed arguments
    """
    description = "IMC UVOS PyTorch Implementation ---- Evaluating Phase"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--cuda', default=True, help='Run on CPU or GPU')
    parser.add_argument('--gpus',
                        type=int,
                        default=settings.DEVICE_ID,
                        help='Choose gpu device')
    parser.add_argument('--arch',
                        '-n',
                        type=str,
                        default=settings.ARCH,
                        help='Feature extracted net type')
    parser.add_argument('--planes',
                        type=int,
                        default=settings.PLANES,
                        help='Deformable Conv planes')
    parser.add_argument('--num-frames',
                        type=int,
                        default=settings.NUM_FRAMES,
                        help='Reference frame number')
    parser.add_argument('--num-classes',
                        type=int,
                        default=settings.NUM_CLASSES,
                        help='Mask layers')
    # parser.add_argument('--runs-dir', type=str, default=settings.RUNS_DIR, help='Runs dir')
    parser.add_argument('--results-dir',
                        type=str,
                        default=settings.INFERCONF['infer']['results_dir'],
                        help='Results dir')
    parser.add_argument('--infer-batch',
                        type=int,
                        default=settings.INFERCONF['infer']['in_batch_sizes'],
                        help='Testing batch size')
    parser.add_argument('--dset',
                        type=str,
                        default=settings.INFERCONF['infer']['eval_dataset'],
                        help='Evaluate dataset')
    parser.add_argument('--multi',
                        action='store_true',
                        help='Use multi-scale inference')

    return parser.parse_args()


args = get_arguments()


def main():
    if 'SEQ_NAME' not in os.environ.keys():
        seq_name = 'dance-twirl'
    else:
        seq_name = str(os.environ['SEQ_NAME'])

    print('=====> Set GPU for evaluate <=====')
    if args.cuda:
        device = torch.device("cuda:" + str(args.gpus) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('Using GPU:{}'.format(args.gpus))

    print('=====> Configure path <=====')
    if not os.path.exists(args.results_dir):
        os.makedirs(os.path.join(args.results_dir))
    results_dir_seq = os.path.join(args.results_dir, seq_name)
    if not os.path.exists(results_dir_seq):
        os.makedirs(results_dir_seq)

    print('=====> Constructing model architecture and resuming from a checkpoint, for continuing training <=====')
    model_name = 'IMCNET'
    model_init = ModelBuild(arch=args.arch, planes=args.planes, num_frames=args.num_frames,
                            with_fusion=True, num_classes=args.num_classes,
                            frozen_layers=settings.INFERCONF['infer']['frozen_layers'], device=device)
    model = model_init.get_model(model_path=settings.INFERCONF['infer']['pre_trained'])

    # print('=====> Configure Tensorboard setting <=====')
    # log_dir = os.path.join(args.runs_dir, settings.TIME_NOW + '_' + socket.gethostname() + '_eval')
    # writer = SummaryWriter(log_dir=log_dir, comment='')

    print('=====> Dataset Loading <=====')
    data_init = DatasetBuild(dataset_list=settings.INFERCONF['infer']['dataset_list'], dataset_conf=settings.DATASET_CONF)
    print('=====> 1.Loading inference dataset <=====')
    testloader = data_init.get_single_dataset(type=args.dset, train=False, seq_name=seq_name, multi=args.multi)
    solver = Solver(model, model_name=model_name, num_frames=args.num_frames, infer_batch=args.infer_batch,
                    device=device, results_dir=args.results_dir)
    print('=====> Begin to evaluating <=====')
    if args.multi:
        multi_scale_size = settings.DATASET_CONF[args.dset]['multi_scale_size']
        solver.multi_scale_test(testloader, multi_scale_size=multi_scale_size)
    else:
        solver.test(testloader)


if __name__ == '__main__':
    main()
