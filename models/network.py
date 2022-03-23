import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractor import FeatureExtractor
from models.libs.DCNv2 import ConvOffset2d
from models.modules import ResBlock, KeyGenerator


class CoAttention(nn.Module):
    """
    Compute Co_attention of reference frame and neighbour frames
    Args:
        planes (int):
        num_frame (int):
    """
    def __init__(self, planes=64, num_frame=3):
        super(CoAttention, self).__init__()
        self.left_transform = nn.Conv2d(planes * num_frame, planes * num_frame, kernel_size=1, stride=1, padding=0)
        self.right_transform = nn.Conv2d(planes * num_frame, planes * num_frame, kernel_size=1, stride=1, padding=0)
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, key, value):
        B, T, Ck, H3, W3 = key.size()
        key = key.view(B, T * Ck, H3, W3)
        key_l_flat = self.left_transform(key)
        key_r_flat = self.right_transform(key)
        key_l = key_l_flat.view(B, -1, H3 * W3)
        key_r = key_r_flat.view(B, -1, H3 * W3)
        key_l_t = torch.transpose(key_l, 1, 2).contiguous()

        S = torch.bmm(key_l_t, key_r)
        Sc = F.softmax(S, dim=1)
        value = (value.view(B, -1, H3, W3)).view(B, -1, H3 * W3)
        att = torch.bmm(value, Sc).view(B, -1, H3, W3)
        att = att.view(B, T, -1, H3, W3)

        return att


class Decoder(nn.Module):
    def __init__(self, in_planes, planes=64):
        super(Decoder, self).__init__()
        self.head = nn.Conv2d(in_planes[3], planes, kernel_size=3, stride=1, padding=1)
        self.l3 = ResBlock(planes, planes)

        self.l2_head = nn.Conv2d(in_planes[2], planes, kernel_size=3, stride=1, padding=1)
        self.l2 = ResBlock(planes, planes)
        self.e2 = ResBlock(planes, planes)
        self.pd2 = nn.Sequential(
            ResBlock(planes, planes),
            nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

        self.l1_head = nn.Conv2d(in_planes[1], planes, kernel_size=3, stride=1, padding=1)
        self.l1 = ResBlock(planes, planes)
        self.e1 = ResBlock(planes, planes)
        self.pd1 = nn.Sequential(
            ResBlock(planes, planes),
            nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

        self.l0_head = nn.Conv2d(in_planes[0], planes, kernel_size=3, stride=1, padding=1)
        self.l0 = ResBlock(planes, planes)
        self.e0 = ResBlock(planes, planes)
        self.pd0 = nn.Sequential(
            ResBlock(planes, planes),
            nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid())

    def _upsample_add(self, x, y):
        [_, _, H_y, W_y] = y.size()
        return F.interpolate(x, size=(H_y, W_y), mode='bilinear', align_corners=True) + y

    def _fg_att(self, feat, pred):
        [_, _, H, W] = feat.size()
        pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
        out = feat * pred
        return out

    def forward(self, att, r3, r2, r1, r0):
        B, C, H, W = r0.size()
        pred = []
        mask = torch.mean(att, dim=[2, 3], keepdim=True)
        weighted_r3 = r3 * att * mask
        cam = torch.mean(weighted_r3, dim=1).unsqueeze(1)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-6)
        cam = torch.clamp(cam, 0, 1)
        _pred = cam
        pred.append(F.interpolate(_pred, size=[H * 4, W * 4], mode='bilinear', align_corners=True))

        p3 = self.l3(self.head(weighted_r3))

        weighted_r2 = self._fg_att(r2, _pred)
        p2 = self._upsample_add(p3, self.l2(self.l2_head(weighted_r2)))
        p2 = self.e2(p2)
        _pred = self.pd2(p2)
        pred.append(F.interpolate(_pred, size=[H * 4, W * 4], mode='bilinear', align_corners=True))

        weighted_r1 = self._fg_att(r1, _pred)
        p1 = self._upsample_add(p2, self.l1(self.l1_head(weighted_r1)))
        p1 = self.e1(p1)
        _pred = self.pd1(p1)
        pred.append(F.interpolate(_pred, size=[H * 4, W * 4], mode='bilinear', align_corners=True))

        weighted_r0 = self._fg_att(r0, _pred)
        p0 = self._upsample_add(p1, self.l0(self.l0_head(weighted_r0)))
        p0 = self.e0(p0)
        _pred = self.pd0(p0)
        pred.append(F.interpolate(_pred, size=[H * 4, W * 4], mode='bilinear', align_corners=True))
        pred = torch.stack(pred, dim=1)

        return p0, pred


class Align(nn.Module):
    """
    Alignment module using Deformable convolution.
    Args:
    """
    def __init__(self, planes=64, deformable_groups=8):
        super(Align, self).__init__()
        # #0
        self.cr0 = nn.Conv2d(planes * 2, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.off2d_0 = nn.Conv2d(planes, 3 * 3 * 2 * deformable_groups,
                                 kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv_0 = ConvOffset2d(planes, planes, kernel_size=3, stride=1,
                                    padding=1, deformable_groups=deformable_groups)
        # #1
        self.cr1 = nn.Conv2d(planes * 2, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.off2d_1 = nn.Conv2d(planes, 3 * 3 * 2 * deformable_groups,
                                 kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv_1 = ConvOffset2d(planes, planes, kernel_size=3, stride=1,
                                    padding=1, deformable_groups=deformable_groups)
        # #2
        self.cr2 = nn.Conv2d(planes * 2, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.off2d_2 = nn.Conv2d(planes, 3 * 3 * 2 * deformable_groups,
                                 kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv_2 = ConvOffset2d(planes, planes, kernel_size=3, stride=1,
                                    padding=1, deformable_groups=deformable_groups)
        # #last
        self.cr = nn.Conv2d(planes * 2, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.off2d = nn.Conv2d(planes, 3 * 3 * 2 * deformable_groups,
                               kernel_size=3, stride=1, padding=1, bias=True)
        self.dconv = ConvOffset2d(planes, planes, kernel_size=3, stride=1,
                                  padding=1, deformable_groups=deformable_groups)

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ref_feat, nbr_feat):
        c_feat = torch.cat([ref_feat, nbr_feat], dim=1)
        feat_off = self.cr0(c_feat)
        offset0 = self.off2d_0(feat_off)
        feat = self.dconv_0(nbr_feat, offset0)
        c_feat = torch.cat([ref_feat, feat], dim=1)
        feat_off = self.cr1(c_feat)
        offset1 = self.off2d_1(feat_off)
        feat = self.dconv_1(feat, offset1)
        c_feat = torch.cat([ref_feat, feat], dim=1)
        feat_off = self.cr2(c_feat)
        offset2 = self.off2d_2(feat_off)
        feat = self.dconv_2(feat, offset2)

        c_feat = torch.cat([ref_feat, feat], dim=1)
        feat_off = self.cr(c_feat)
        offset = self.off2d(feat_off)
        feat = self.dconv(feat, offset)

        return feat


class Fusion(nn.Module):
    """
    coordinate attention module
    """
    def __init__(self, planes=64, num_frame=3):
        super(Fusion, self).__init__()
        self.center_frame_idx = num_frame // 2
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.temporal_attn2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.feat_fusion = nn.Conv2d(planes * 3, planes, kernel_size=1, stride=1, padding=0)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(planes * num_frame, planes, kernel_size=1, stride=1, padding=0)
        self.spatial_attn2 = nn.Conv2d(planes * 2, planes, kernel_size=1, stride=1, padding=0)
        self.spatial_attn3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.spatial_attn4 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0)
        self.spatial_attn5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.spatial_attn_l1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0)
        self.spatial_attn_l2 = nn.Conv2d(planes * 2, planes, kernel_size=3, stride=1, padding=1)
        self.spatial_attn_l3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.spatial_attn_add1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0)
        self.spatial_attn_add2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        B, T, C, H, W = x.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            x[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(x.view(-1, C, H, W))
        embedding = embedding.view(B, T, -1, H, W)

        corr_l = []  # correlation list
        for idx in range(T):
            emb_neighbor = embedding[:, idx, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, dim=1)
            corr_l.append(corr.unsqueeze(1))
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))
        corr_prob = corr_prob.unsqueeze(2).expand(B, T, C, H, W)
        corr_prob = corr_prob.contiguous().view(B, -1, H, W)
        x = x.view(B, -1, H, W) * corr_prob

        # fusion
        feat = self.relu(self.feat_fusion(x))

        # spatial attention
        attn = self.relu(self.spatial_attn1(x))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.relu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.relu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.relu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.relu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.relu(self.spatial_attn3(attn)) + attn_level
        attn = self.relu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.relu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        feat = feat * attn * 2 + attn_add
        return feat


class Segment(nn.Module):
    def __init__(self, in_planes, planes, num_classes=2):
        super(Segment, self).__init__()
        self.reslayer = ResBlock(in_planes, planes)
        self.predlayer = nn.Conv2d(planes, num_classes - 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.reslayer(x)
        x = self.predlayer(x)
        return x


class IMCNet(nn.Module):
    """
    """
    def __init__(self,
                 arch='resnet18',
                 frozen_layers='none',
                 pretrained=True,
                 ckpt_dir=None,
                 planes=64,
                 num_frame=3,
                 deformable_groups=8,
                 with_fusion=True,
                 num_classes=2):
        super(IMCNet, self).__init__()
        self.arch = arch
        self.frozen_layers = frozen_layers
        self.pretrained = pretrained
        self.ckpt_dir = ckpt_dir
        self.planes = planes
        self.num_frame = num_frame
        self.deformable_groups = deformable_groups
        self.with_fusion = with_fusion
        self.num_classes = num_classes
        self.center_frame_idx = num_frame // 2
        is_light = arch in ['resnet18', 'resnet34']
        num_feat = [64, 128, 256, 512] if is_light else [256, 512, 1024, 2048]

        # feature extractor
        self.feature_extractor = FeatureExtractor(self.arch, self.frozen_layers,
                                                  self.pretrained, self.ckpt_dir)
        self.K_r3 = KeyGenerator(indim=num_feat[-1], keydim=planes)
        self.co_att_x3 = CoAttention(planes=planes, num_frame=num_frame)
        self.decoder = Decoder(num_feat, planes)

        self.align = Align(planes=planes, deformable_groups=deformable_groups)
        if with_fusion:
            self.fusion = Fusion(planes=planes, num_frame=num_frame)
        else:
            self.fusion = nn.Conv2d(num_frame * planes, planes, kernel_size=3, stride=1, padding=1)
        self.segmentation = Segment(planes, planes, num_classes)

    def forward(self, x, d_type=None):
        B, T, C, H, W = x.size()

        r0, r1, r2, r3 = [[] for idx in range(4)]
        for idx in range(T):
            res_out = self.feature_extractor(x[:, idx, :, :, :])
            r0.append(res_out['l0'])
            r1.append(res_out['l1'])
            r2.append(res_out['l2'])
            r3.append(res_out['l3'])
        r0 = torch.stack(r0, dim=1)
        r1 = torch.stack(r1, dim=1)
        r2 = torch.stack(r2, dim=1)
        r3 = torch.stack(r3, dim=1)

        if d_type == 'image_set':
            B3, T3, C3, H3, W3 = r3.size()
            r3_att = r3.new_ones([B3, T3, C3, H3, W3], requires_grad=False)
        else:
            r3_key = []
            for idx in range(T):
                r3_key.append(self.K_r3(r3[:, idx, :, :, :]))
            r3_key = torch.stack(r3_key, dim=1)
            r3_att = self.co_att_x3(r3_key, r3)

        feat = []
        saliency_map = []
        for idx in range(T):
            p0, _pred = self.decoder(r3_att[:, idx, :, :, :], r3[:, idx, :, :, :],
                                     r2[:, idx, :, :, :], r1[:, idx, :, :, :],
                                     r0[:, idx, :, :, :])
            feat.append(p0)
            saliency_map.append(_pred)

        if d_type == 'image_set':
            feat_fusion = feat[1]
        else:
            ref_feat = feat[self.center_frame_idx]
            aligned_feat = []
            for idx in range(T):
                if idx == self.center_frame_idx:
                    aligned_feat.append(feat[idx])
                    continue
                nbr_feat = feat[idx]
                aligned_feat.append(self.align(ref_feat, nbr_feat))
            aligned_feat = torch.stack(aligned_feat, dim=1)
            if not self.with_fusion:
                aligned_feat = aligned_feat.view(B, -1, H // 4, W // 4)
            feat_fusion = self.fusion(aligned_feat)
        pred_logits = self.segmentation(feat_fusion)
        pred = pred_logits.sigmoid()

        pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)
        pred_logits = F.interpolate(pred_logits, size=(H, W), mode='bilinear', align_corners=True)

        return pred, pred_logits, saliency_map
