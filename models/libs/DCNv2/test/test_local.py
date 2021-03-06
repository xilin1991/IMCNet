#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from modules.deform_conv import DeformConv, _DeformConv, DeformConvPack
from modules.modulated_deform_conv import ModulatedDeformConv, _ModulatedDeformConv, ModulatedDeformConvPack


deformable_groups = 1
N, inC, inH, inW = 2, 4, 4, 4
outC = 4
kH, kW = 3, 3

torch.manual_seed(3)


def example_dconv():
    """example dconv"""
    input = torch.randn(2, 64, 480, 480).cuda()
    dcn = DeformConvPack(64, 128, kernel_size=(3, 3), stride=1, padding=1, groups=1, deformable_groups=1).cuda()
    # print(dcn.weight.shape, dcn.bias.shape)
    output = dcn(input)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print(output.shape)


def example_mdconv():
    """example modulated dconv"""
    input = torch.randn(2, 64, 480, 480).cuda()
    dcn = ModulatedDeformConvPack(64, 128, kernel_size=(3, 3), stride=1, padding=1, groups=1, deformable_groups=1).cuda()
    output = dcn(input)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print(output.shape)


def check_dconv_im2col_step_forward():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    groups = 2

    dcn1 = DeformConv(inC, outC, (kH, kW),
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=groups,
                      deformable_groups=deformable_groups,
                      im2col_step=1).cuda()

    dcn2 = DeformConv(inC, outC, (kH, kW),
                      stride=1,
                      padding=1,
                      dilation=1,
                      groups=groups,
                      deformable_groups=deformable_groups,
                      im2col_step=2).cuda()
    dcn1.weight = dcn2.weight
    dcn1.bias = dcn2.bias
    output1 = dcn1(input, offset)
    output2 = dcn2(input, offset)

    d = (output1 - output2).abs().max()
    if d < 1e-10:
        print('dconv im2col_step forward passed with {}'.format(d))
    else:
        print('dconv im2col_step forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())


def check_mdconv_im2col_step_forward():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()
    conv_mask = nn.Conv2d(inC, deformable_groups * 1 * kH * kW,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    groups = 2

    dcn1 = ModulatedDeformConv(inC, outC, (kH, kW),
                               stride=1,
                               padding=1,
                               dilation=1,
                               groups=groups,
                               deformable_groups=deformable_groups,
                               im2col_step=1).cuda()

    dcn2 = ModulatedDeformConv(inC, outC, (kH, kW),
                               stride=1,
                               padding=1,
                               dilation=1,
                               groups=groups,
                               deformable_groups=deformable_groups,
                               im2col_step=2).cuda()
    dcn1.weight = dcn2.weight
    dcn1.bias = dcn2.bias
    output1 = dcn1(input, offset, mask)
    output2 = dcn2(input, offset, mask)

    d = (output1 - output2).abs().max()
    if d < 1e-10:
        print('mdconv im2col_step forward passed with {}'.format(d))
    else:
        print('mdconv im2col_step forward failed with {}'.format(d))
        print(output1)
        print(output2)
        print((output1 - output2).abs())


def check_dconv_im2col_step_backward():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kH * kW, inH, inW).cuda() * 2
    offset.requires_grad = True

    weight = torch.randn(outC, int(inC // groups), kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    output1 = _DeformConv(input, offset, weight, bias, stride, padding,
                          dilation, groups, deformable_groups, 2)
    target = torch.rand(*output1.size()).cuda()
    error = (target - output1).mean()
    error.backward(retain_graph=True)
    input_grad = input.grad.clone()
    offset_grad = offset.grad.clone()
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()
    output2 = _DeformConv(input, offset, weight, bias, stride, padding,
                          dilation, groups, deformable_groups, 1)
    error2 = (target - output2).mean()
    error2.backward()
    print((output1 - output2).abs().max())
    input_grad_err = (input.grad - 2 * input_grad).abs().max()
    offset_grad_err = (offset.grad - 2 * offset_grad).abs().max()
    weight_grad_err = (weight.grad - 2 * weight_grad).abs().max()
    bias_grad_err = (bias.grad - 2 * bias_grad).abs().max()
    grad_err = input_grad_err + offset_grad_err + weight_grad_err + bias_grad_err
    if grad_err:
        print("dconv im2col_step backward passed with {} = {}+{}+{}+{}".format(
              grad_err, input_grad_err, offset_grad_err, weight_grad_err,
              bias_grad_err))
    else:
        print("dconv im2col_step backward failed with {} = {}+{}+{}+{}".format(
              grad_err, input_grad_err, offset_grad_err, weight_grad_err,
              bias_grad_err))


def check_mdconv_im2col_step_backward():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH,
                         inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.sigmoid(
        torch.randn(N, deformable_groups * 1 * kW * kH, inH, inW).cuda())
    mask.requires_grad = True

    weight = torch.randn(outC, int(inC // groups), kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    output1 = _ModulatedDeformConv(input, offset, mask, weight, bias, stride,
                                   padding, dilation, groups,
                                   deformable_groups, 2)
    targert = torch.rand(*output1.size()).cuda()
    error = (targert - output1).mean()
    error.backward(retain_graph=True)
    input_grad = input.grad.clone()
    offset_grad = offset.grad.clone()
    mask_grad = mask.grad.clone()
    weight_grad = weight.grad.clone()
    bias_grad = bias.grad.clone()
    output2 = _ModulatedDeformConv(input, offset, mask, weight, bias, stride,
                                   padding, dilation, groups,
                                   deformable_groups, 1)
    error2 = (targert - output2).mean()
    error2.backward()
    print((output1 - output2).abs().max())
    input_grad_err = (input.grad - 2 * input_grad).abs().max()
    offset_grad_err = (offset.grad - 2 * offset_grad).abs().max()
    mask_grad_err = (mask.grad - 2 * mask_grad).abs().max()
    weight_grad_err = (weight.grad - 2 * weight_grad).abs().max()
    bias_grad_err = (bias.grad - 2 * bias_grad).abs().max()
    grad_err = input_grad_err + offset_grad_err + mask_grad_err + weight_grad_err + bias_grad_err
    if grad_err < 1e-7:
        print("mdconv im2col_step backward passed with {}".format(grad_err))
    else:
        print("mdconv im2col_step backward failed with {}".format(grad_err))


def conv_identify(weight, bias, groups=1):
    weight.data.zero_()
    bias.data.zero_()
    o, i, h, w = weight.shape
    y = h // 2
    x = w // 2
    oc = o // groups
    for p in range(i):
        for q in range(o):
            if (p) == (q % oc):
                weight.data[q, p, y, x] = 1.0


def check_dconv_zero_offset():
    conv_offset = nn.Conv2d(inC, deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    dcn = DeformConv(inC, outC, (kH, kW),
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=2,
                     deformable_groups=deformable_groups,
                     im2col_step=1).cuda()
    pcn = nn.Conv2d(inC, outC, (kH, kW),
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=2).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    output_d = dcn(input, offset)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('dconv zero offset passed with {}'.format(d))
    else:
        print('dconv zero offset failed with {}'.format(d))
        print((output_d - output_p).abs())


def check_mdconv_zero_offset():
    conv_offset = nn.Conv2d(inC,
                            deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC,
                          deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    dcn = ModulatedDeformConv(inC,
                              outC, (kH, kW),
                              stride=1,
                              padding=1,
                              dilation=1,
                              groups=2,
                              deformable_groups=deformable_groups,
                              im2col_step=1).cuda()
    pcn = nn.Conv2d(inC,
                    outC, (kH, kW),
                    stride=1,
                    padding=1,
                    dilation=1,
                    groups=2).cuda()
    pcn.weight = dcn.weight
    pcn.bias = dcn.bias
    print((pcn.weight.data - dcn.weight.data).abs().max())

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    mask *= 2
    output_d = dcn(input, offset, mask)
    output_p = pcn(input)
    d = (output_d - output_p).abs().max()
    if d < 1e-5:
        print('mdconv zero offset passed with {}'.format(d))
    else:
        print('mdconv zero offset failed with {}'.format(d))
        # print(output_p)
        # print(output_d)
        print((output_d - output_p).abs())


def check_dconv_zero_offset_identify():
    conv_offset = nn.Conv2d(inC,
                            deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    groups = 2
    dcn = DeformConv(inC, outC, (kH, kW),
                     stride=1,
                     padding=1,
                     dilation=1,
                     groups=groups,
                     deformable_groups=deformable_groups,
                     im2col_step=1).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_identify(dcn.weight, dcn.bias, groups)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    output = dcn(input, offset)
    d = (input - output).abs().max()
    if d < 1e-10:
        print('dconv zero offset identify passed with {}'.format(d))
    else:
        print('dconv zero offset identify failed with {}'.format(d))
        print((input - output).abs())


def check_mdconv_zero_offset_identify():
    conv_offset = nn.Conv2d(inC,
                            deformable_groups * 2 * kH * kW,
                            kernel_size=(kH, kW),
                            stride=(1, 1),
                            padding=(1, 1),
                            bias=True).cuda()

    conv_mask = nn.Conv2d(inC,
                          deformable_groups * 1 * kH * kW,
                          kernel_size=(kH, kW),
                          stride=(1, 1),
                          padding=(1, 1),
                          bias=True).cuda()

    groups = 2
    dcn = ModulatedDeformConv(inC,
                              outC, (kH, kW),
                              stride=1,
                              padding=1,
                              dilation=1,
                              groups=groups,
                              deformable_groups=deformable_groups,
                              im2col_step=1).cuda()

    conv_offset.weight.data.zero_()
    conv_offset.bias.data.zero_()
    conv_mask.weight.data.zero_()
    conv_mask.bias.data.zero_()
    conv_identify(dcn.weight, dcn.bias, groups)

    input = torch.randn(N, inC, inH, inW).cuda()
    offset = conv_offset(input)
    mask = conv_mask(input)
    mask = torch.sigmoid(mask)
    output = dcn(input, offset, mask)
    output *= 2
    d = (input - output).abs().max()
    if d < 1e-10:
        print('mdconv zero offset identify passed with {}'.format(d))
    else:
        print('mdconv zero offset identify failed with {}'.format(d))
        # print(input)
        # print(output)
        print((input - output).abs())


def check_gradient_conv():
    input = torch.rand(N, inC, inH, inW).double().cuda() * 0.01
    input.requires_grad = True
    from torch.nn.functional import conv2d

    weight = torch.randn(outC, inC, kH, kW).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).double().cuda()
    bias.requires_grad = True

    stride = 1
    padding = 1
    dilation = 1
    print('check_gradient_conv: ',
          gradcheck(conv2d, (input, weight, bias, stride, padding, dilation, deformable_groups)))


def check_gradient_dconv():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1
    im2col_step = 1

    input = torch.rand(N, inC, inH, inW).double().cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kH * kW, inH, inW).double().cuda() * 2
    offset.required_grad = True

    weight = torch.randn(outC, int(inC // groups), kH, kW).double().cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).double().cuda()
    bias.requires_grad = True

    print('check_gradient_dconv:',
          gradcheck(_DeformConv,
                    (input, offset, weight, bias, stride, padding, dilation, groups, deformable_groups, im2col_step),
                    eps=1e-3,
                    atol=1e-3,
                    rtol=1e-2,
                    raise_exception=True))


def check_gradient_mdconv():
    stride = 1
    padding = 1
    groups = 2
    dilation = 1
    im2col_step = 1

    input = torch.rand(N, inC, inH, inW).cuda() * 0.01
    input.requires_grad = True

    offset = torch.randn(N, deformable_groups * 2 * kW * kH, inH,
                         inW).cuda() * 2
    # offset.data.zero_()
    # offset.data -= 0.5
    offset.requires_grad = True

    mask = torch.rand(N, deformable_groups * 1 * kW * kH, inH, inW).cuda()
    # mask.data.zero_()
    mask.requires_grad = True
    mask = torch.sigmoid(mask)

    weight = torch.randn(outC, int(inC // groups), kH, kW).cuda()
    weight.requires_grad = True

    bias = torch.rand(outC).cuda()
    bias.requires_grad = True

    print('check_gradient_mdconv: ',
          gradcheck(_ModulatedDeformConv,
                    (input, offset, mask, weight, bias, stride, padding, dilation, groups, deformable_groups, im2col_step),
                    eps=1e-3,
                    atol=1e-3,
                    rtol=1e-2,
                    raise_exception=True))


if __name__ == '__main__':
    example_dconv()
    example_mdconv()

    print('Checking...')
    check_dconv_im2col_step_forward()
    check_dconv_im2col_step_backward()
    check_mdconv_im2col_step_forward()
    check_mdconv_im2col_step_backward()

    if inC == outC:
        check_dconv_zero_offset()
        check_dconv_zero_offset_identify()
        check_mdconv_zero_offset()
        check_mdconv_zero_offset_identify()

    check_gradient_conv()
    # check_gradient_dconv()
    check_gradient_mdconv()
