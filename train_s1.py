import os
import socket

import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import Loss, Solver
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
    description = 'IMC UVOS PyTorch Implementation ---- Training Phase'
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
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default=settings.TRAINCONF['s1']['checkpoints_dir'],
                        help='Checkpoints dir')
    parser.add_argument('--runs-dir',
                        type=str,
                        default=settings.TRAINCONF['s1']['runs_dir'],
                        help='Runs dir')
    parser.add_argument('--resume',
                        type=bool,
                        default=settings.TRAINCONF['s1']['resume'],
                        help='Default is False, change if want to resume')
    parser.add_argument('--load-best',
                        type=bool,
                        default=settings.TRAINCONF['s1']['load_best'],
                        help='Default is False, load checkpoint based best IOU')
    parser.add_argument('--load-epoch',
                        type=int,
                        default=settings.TRAINCONF['s1']['load_epoch'],
                        help='load checkpoint based epoch')
    parser.add_argument('--num-epochs',
                        type=int,
                        default=settings.TRAINCONF['s1']['num_epochs'],
                        help='Number of epochs for training')
    parser.add_argument('--train-batch',
                        type=int,
                        default=settings.TRAINCONF['s1']['tr_batch_sizes'],
                        help='Training batch size')
    parser.add_argument('--test-batch',
                        type=int,
                        default=settings.TRAINCONF['s1']['ts_batch_sizes'],
                        help='Testing batch size')
    parser.add_argument('--lr',
                        type=float,
                        default=settings.TRAINCONF['s1']['lr'],
                        help='Learning rate')
    parser.add_argument('--wd',
                        type=float,
                        default=settings.TRAINCONF['s1']['wd'],
                        help='Weights decay')
    parser.add_argument('--momentum',
                        type=float,
                        default=settings.TRAINCONF['s1']['momentum'],
                        help='Momentum')
    parser.add_argument('--snapshot',
                        type=int,
                        default=settings.TRAINCONF['s1']['snapshot'],
                        help='Store a model every snapshot epochs')
    parser.add_argument('--use-test',
                        action='store_true',
                        default=settings.TRAINCONF['s1']['use_test'],
                        help='See evolution of the test set when training?')
    parser.add_argument('--test-session',
                        type=int,
                        default=settings.TRAINCONF['s1']['test_session'],
                        help='Run on test set every nTestInterval epochs')

    return parser.parse_args()


args = get_arguments()


def main():
    print('=====> Set GPU for training <=====')
    if args.cuda:
        device = torch.device("cuda:" + str(args.gpus) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('Using GPU:{}'.format(args.gpus))

    print('=====> Configure path <=====')
    if not os.path.exists(os.path.join(args.checkpoint_dir)):
        os.makedirs(os.path.join(args.checkpoint_dir))
    if not os.path.exists(os.path.join(args.runs_dir)):
        os.makedirs(os.path.join(args.runs_dir))

    print('=====> Constructing model architecture and resuming from a checkpoint, for continuing training <=====')
    model_name = 'IMCNET'
    model_init = ModelBuild(arch=args.arch, frozen_layers=settings.TRAINCONF['s1']['frozen_layers'],
                            pretrained=False, ckpt_dir=settings.RESNET_CKPT_DIRS[args.arch],
                            planes=args.planes, num_frames=args.num_frames, with_fusion=True,
                            num_classes=args.num_classes, device=device)
    model = model_init.get_model()
    params = [{'params': model.feature_extractor.parameters(), 'lr': args.lr * 0.01},
              {'params': model.K_r3.parameters(), 'lr': args.lr * 0.01},
              {'params': model.co_att_x3.parameters(), 'lr': args.lr * 0.1},
              {'params': model.decoder.parameters(), 'lr': args.lr * 0.1},
              {'params': model.align.parameters(), 'lr': args.lr},
              {'params': model.fusion.parameters(), 'lr': args.lr * 2},
              {'params': model.segmentation.parameters(), 'lr': args.lr * 2}]

    print('=====> Configure Tensorboard setting <=====')
    log_dir = os.path.join(args.runs_dir, settings.TIME_NOW + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir, comment='')

    print('=====> Dataset Loading <=====')
    data_init = DatasetBuild(dataset_list=settings.TRAINCONF['s1']['dataset_list'],
                             dataset_conf=settings.DATASET_CONF)
    print('=====> 1.Loading training dataset <=====')
    trainloader = data_init.get_multi_dataset(dset_list=settings.TRAINCONF['s1']['dataset_list'], train=True)
    print('=====> 2.Loading testing dataset <=====')
    testloader = data_init.get_single_dataset(type='davis2016', train=False)

    criterion = Loss(size_average=True, batch_average=False)
    optimizer = optim.Adam(params=params, lr=args.lr, betas=[0.9, 0.99], weight_decay=args.wd, eps=1e-08)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 50], gamma=0.5)
    solver = Solver(model, model_name=model_name, num_frames=args.num_frames,
                    criterion=criterion, optimizer=optimizer, scheduler=scheduler, writer=writer,
                    train_batch=args.train_batch, test_batch=args.test_batch, num_epochs=args.num_epochs,
                    resume=args.resume, snapshot=args.snapshot, device=device, checkpoint_dir=args.checkpoint_dir,
                    load_best=args.load_best, load_epoch=args.load_epoch, use_test=args.use_test, test_session=args.test_session)
    print('=====> Begin to train <=====')
    solver.train(trainloader, testloader)


if __name__ == '__main__':
    main()
