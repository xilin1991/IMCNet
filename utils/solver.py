import os
import timeit

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F

from measures import db_eval_iou_multi


class Solver(object):
    """
    """
    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.model_name = kwargs.pop('model_name', False)
        self.num_frames = kwargs.pop('num_frames', 3)
        self.criterion = kwargs.pop('criterion', None)
        self.optimizer = kwargs.pop('optimizer', None)
        self.scheduler = kwargs.pop('scheduler', None)
        self.writer = kwargs.pop('writer', None)

        self.train_batch = kwargs.pop('train_batch', 2)
        self.test_batch = kwargs.pop('test_batch', 1)
        self.infer_batch = kwargs.pop('infer_batch', 1)
        self.num_epochs = kwargs.pop('num_epochs', 100)
        self.resume = kwargs.pop('resume', False)

        self.snapshot = kwargs.pop('snapshot', 40)
        self.device = kwargs.pop('device', 'cuda:0')
        self.test_session = kwargs.pop('test_session', 4)
        self.load_best = kwargs.pop('load_best', False)
        self.load_epoch = kwargs.pop('load_epoch', 0)
        self.checkpoint_dir = kwargs.pop('checkpoint_dir', './checkpoints')
        self.use_test = kwargs.pop('use_test', True)
        self.results_dir = kwargs.pop('results_dir', './results')

        self.epoch = 0
        self.center_frame_idx = self.num_frames // 2
        self.stats = dict(best_iou=0, loss_tr=[], loss_ts=[])
        # self.iters_to_accumulate = 16

        self._set()

    def _set(self):
        """
        Load checkpoint and others
        """
        if self.resume:
            print('=====> Loading checkpoint <=====')
            if self.load_best:
                checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_best.pth')
            else:
                checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name + '_' + str(self.load_epoch) + '.pth')
            print('Updating weights from: {}'.format(checkpoint_file))
            self.load_checkpoint(checkpoint_file)

    def load_checkpoint(self, checkpoint_file, fields=None, ignore_fields=None):
        ckpt = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        if fields is None:
            fields = ckpt.keys()
        if ignore_fields is None:
            ignore_fields = ['model_name']
        ignore_fields.extend(['type', 'schedular'])
        self.epoch = ckpt['epoch']
        self.epoch = self.epoch + 1
        print('Starting epoch: {}'.format(self.epoch + 1))
        self.stats = ckpt['stats']
        for key in fields:
            if key in ignore_fields:
                continue
            elif key == 'model':
                self.model.load_state_dict(ckpt[key])
            elif key == 'optimizer':
                if self.optimizer is not None:
                    self.optimizer.load_state_dict(ckpt[key])
        if self.scheduler is not None:
            self.scheduler.last_epoch = self.epoch

    def save_checkpoint(self, type='latest'):
        ckpt = dict(model_name=self.model_name,
                    type=type,
                    epoch=self.epoch,
                    stats=self.stats,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict())
        if type == 'latest':
            torch.save(ckpt, os.path.join(self.checkpoint_dir, self.model_name + '_' + str(self.epoch) + '.pth'))
        else:
            torch.save(ckpt, os.path.join(self.checkpoint_dir, self.model_name + '_' + type + '.pth'))

    def _epoch_step(self, trainloader, testloader, epoch):
        self.model.train()
        num_img_tr = len(trainloader)
        running_loss_tr = 0
        start_time = timeit.default_timer()
        pbar_desc = 'Training progress ==> [Epoch: {}]'.format(epoch)
        pbar = tqdm(enumerate(trainloader), desc=pbar_desc, total=len(trainloader), ncols=120)
        for it, sample_batched in pbar:
            inputs = sample_batched['imgs']
            center_gt = sample_batched['gts'][:, self.center_frame_idx, :, :, :]
            gts = sample_batched['gts']
            d_type = sample_batched['d_type']
            assert len(set(d_type)) == 1, 'd_type is not only one value!'
            d_type = d_type[0]

            # Forward-Backward of the mini-batch
            inputs.requires_grad_()
            self.optimizer.zero_grad()
            inputs = inputs.to(self.device)
            center_gt = center_gt.to(self.device)
            gts = gts.to(self.device)

            output, _, saliency_map = self.model(inputs, d_type)
            mask_loss = self.criterion(output, center_gt)
            side_loss = 0
            for idx_list in range(len(saliency_map)):
                for idx_tensor in range(saliency_map[idx_list].size(1)):
                    side_loss += (self.criterion(saliency_map[idx_list][:, idx_tensor, :, :, :], gts[:, idx_list, :, :, :]) / 12)
            loss = mask_loss + side_loss
            running_loss_tr += loss.item()

            # Backward + update
            self.writer.add_scalar('train/total_loss_iter', loss.item(), it + num_img_tr * epoch)
            self.writer.add_scalar('train/mask_loss_iter', mask_loss.item(), it + num_img_tr * epoch)
            self.writer.add_scalar('train/side_loss_iter', side_loss.item(), it + num_img_tr * epoch)
            loss.backward()
            self.optimizer.step()

            # print stuff
            if it % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                self.stats['loss_tr'].append(running_loss_tr)
                self.writer.add_scalar('train/total_loss_epoch', running_loss_tr, epoch)
                postfix_str = {'Training Loss': '%.5f' % running_loss_tr}
                pbar.set_postfix(postfix_str)
                running_loss_tr = 0

        # Save the model
        if (epoch % self.snapshot) == self.snapshot - 1 and epoch != 0:
            print('=====> Saving model <=====')
            self.save_checkpoint(type='latest')

        # One testing epoch
        if self.use_test and epoch % self.test_session == (self.test_session - 1):
            print('=====> Validatate model <=====')
            self.validation(testloader, epoch)
        # execution time compute
        stop_time = timeit.default_timer()
        exec_time = stop_time - start_time
        exec_min = int(exec_time // 60)
        exec_sec = int(exec_time % 60)
        print('Execution time: ' + str(exec_min) + ' minutes, ' + str(exec_sec) + ' seconds.')

    def train(self, trainloader, testloader):
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            self._epoch_step(trainloader, testloader, epoch)
            self.scheduler.step()
        self.writer.close()

    def validation(self, testloader, epoch):
        self.model.eval()
        num_img_ts = len(testloader)
        running_loss_ts = 0
        miou_ts = 0
        with torch.no_grad():
            pbar_desc = 'Validation progress ==> [Epoch: {}]'.format(epoch)
            pbar = tqdm(enumerate(testloader), desc=pbar_desc, total=len(testloader), ncols=120)
            for it, sample_batched in pbar:
                inputs = sample_batched['imgs']
                center_gt = sample_batched['gts'][:, self.center_frame_idx, :, :, :]
                gts = sample_batched['gts']
                d_type = sample_batched['d_type']
                assert len(set(d_type)) == 1, 'd_type is not only one value!'
                d_type = d_type[0]

                # Forward-Backward of the mini-batch
                inputs = inputs.to(self.device)
                center_gt = center_gt.to(self.device)
                gts = gts.to(self.device)

                output, _, saliency_map = self.model(inputs, d_type)

                # Compute the loss
                mask_loss = self.criterion(output, center_gt)
                side_loss = 0
                for idx_list in range(len(saliency_map)):
                    for idx_tensor in range(saliency_map[idx_list].size(1)):
                        side_loss += (self.criterion(saliency_map[idx_list][:, idx_tensor, :, :, :], gts[:, idx_list, :, :, :]) / 12)
                loss = mask_loss + side_loss
                running_loss_ts += loss.item()
                iou = db_eval_iou_multi(center_gt.cpu().detach().numpy(), output.cpu().detach().numpy())
                miou_ts += iou
                if it % num_img_ts == num_img_ts - 1:
                    running_loss_ts = running_loss_ts / num_img_ts
                    miou_ts = miou_ts / num_img_ts
                    self.stats['loss_ts'].extend([running_loss_ts] * self.test_session)
                    self.writer.add_scalar('test/test_loss_epoch', running_loss_ts, epoch)
                    self.writer.add_scalar('test/mIoU_epoch', miou_ts, epoch)
                    postfix_str = {'Testing Loss': '%.5f' % running_loss_ts, 'Testing mIoU': '%.2f' % miou_ts}
                    pbar.set_postfix(postfix_str)
                    if miou_ts > self.stats['best_iou']:
                        self.stats['best_iou'] = miou_ts
                        if epoch >= 5:
                            print('=====> Saving best model <=====')
                            self.save_checkpoint(type='best')

    def test(self, testloader):
        self.model.eval()
        # num_img_infer = len(testloader)
        with torch.no_grad():
            pbar_desc = 'Inference progress ==>'
            pbar = tqdm(enumerate(testloader), desc=pbar_desc, total=len(testloader), ncols=120, unit='images')
            for it, sample_batched in pbar:
                inputs = sample_batched['imgs']
                fname = sample_batched['fname']
                img_size = sample_batched['img_size']
                d_type = sample_batched['d_type']
                seq_name = os.path.split(fname[0])[0]
                pbar.set_description('Inference progress ==> [ ' + seq_name + ' ]')
                assert len(set(d_type)) == 1, 'd_type is not only one value!'
                d_type = d_type[0]

                inputs = inputs.to(self.device)
                # Inference
                _, output, _ = self.model(inputs, d_type)

                for idx in range(inputs.size()[0]):
                    pred = output[idx, 0, :, :].sigmoid().cpu().detach().numpy() * 255
                    mask = Image.fromarray(pred).convert('L')
                    mask = mask.resize(img_size, Image.NEAREST)
                    mask.save(os.path.join(self.results_dir, fname[idx] + '.png'))

    def multi_scale_test(self, testloader, multi_scale_size=[(352, 352), (480, 480), (736, 736)]):
        self.model.eval()
        # DAVIS
        # multi_scale_size = [(352, 352), (480, 480), (736, 736)]
        # YouTube Objects
        # multi_scale_size = [(224, 224), (288, 288), (448, 448)]
        # num_img_infer = len(testloader)
        with torch.no_grad():
            pbar_desc = 'Inference progress ==>'
            pbar = tqdm(enumerate(testloader), desc=pbar_desc, total=len(testloader), ncols=120, unit='images')
            for it, sample_batched in pbar:
                inputs = sample_batched['imgs']
                fname = sample_batched['fname']
                img_size = sample_batched['img_size']
                d_type = sample_batched['d_type']
                seq_name = os.path.split(fname[0])[0]
                pbar.set_description('Inference progress ==> [ ' + seq_name + ' ]')
                assert len(set(d_type)) == 1, 'd_type is not only one value!'
                d_type = d_type[0]

                inputs = inputs.to(self.device)
                inputs_list = []
                for s in multi_scale_size:
                    inputs_list.append(F.interpolate(inputs[0], size=s, mode='bilinear', align_corners=True))
                    inputs_list.append(torch.flip(F.interpolate(inputs[0], size=s, mode='bilinear', align_corners=True), dims=[2]))
                    inputs_list.append(torch.flip(F.interpolate(inputs[0], size=s, mode='bilinear', align_corners=True), dims=[3]))
                # Inference
                inputs_0 = torch.stack(inputs_list[:3], dim=0)
                inputs_1 = torch.stack(inputs_list[3:6], dim=0)
                inputs_2 = torch.stack(inputs_list[6:], dim=0)
                output_0, *_ = self.model(inputs_0, d_type)
                output_1, *_ = self.model(inputs_1, d_type)
                output_2, *_ = self.model(inputs_2, d_type)
                output_0[1] = torch.flip(output_0[1], dims=[1])
                output_0[2] = torch.flip(output_0[2], dims=[2])
                output_1[1] = torch.flip(output_1[1], dims=[1])
                output_1[2] = torch.flip(output_1[2], dims=[2])
                output_2[1] = torch.flip(output_2[1], dims=[1])
                output_2[2] = torch.flip(output_2[2], dims=[2])
                output_0 = F.interpolate(output_0, size=[img_size[1], img_size[0]], mode='bilinear', align_corners=True)
                output_1 = F.interpolate(output_1, size=[img_size[1], img_size[0]], mode='bilinear', align_corners=True)
                output_2 = F.interpolate(output_2, size=[img_size[1], img_size[0]], mode='bilinear', align_corners=True)
                output = torch.cat([output_0, output_1, output_2], dim=0)
                # output = torch.sigmoid(torch.mean(output, dim=0, keepdim=True))
                output = torch.mean(output, dim=0, keepdim=True)

                for idx in range(inputs.size()[0]):
                    pred = output[idx, 0, :, :].cpu().detach().numpy() * 255
                    mask = Image.fromarray(pred).convert('L')
                    # mask = mask.resize(img_size, Image.NEAREST)
                    mask.save(os.path.join(self.results_dir, fname[idx] + '.png'))
