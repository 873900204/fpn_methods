from math import pi
from math import log
import mmcv
import torch
import torch.nn as nn
import numpy as np

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss




@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def biou_loss(encode_decode_preds, encode_decode_targets, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6):
    encode_pred, decode_pred = encode_decode_preds
    encode_target, decode_target = encode_decode_targets
    ious = bbox_overlaps(decode_pred, decode_target, is_aligned=True, eps=eps)
    #smooth l1 loss
    diff = torch.abs(encode_pred - encode_target)
    smooth_l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                                 diff - 0.5 * beta).sum(1)
    assert beta > 0
    assert encode_pred.size() == encode_target.size() and encode_target.numel() > 0
    diff = torch.abs(encode_pred - encode_target)

    # b = np.e**(gamma / alpha) - 1
    # balanced_l1_loss = torch.where(
    #     diff < beta, alpha / b *
    #     (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
    #     gamma * diff + gamma / b - alpha * beta).sum(1) 
    #主要在这里修改
    #分多个步骤
    #第一个是IOU*
    #第二个是IOU
    #第三个是直线的IOU
    #第四个是曲线的IOU
    balanced_iou_loss=torch.where(ious<0.5,1-ious,1+0.25*ious-torch.pow(ious-0.5,3)/3)

    #loss = balanced_iou_loss + balanced_l1_loss
    loss = balanced_iou_loss

    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def b1iou_loss(encode_decode_preds, encode_decode_targets, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6):
    encode_pred, decode_pred = encode_decode_preds
    encode_target, decode_target = encode_decode_targets
    ious = bbox_overlaps(decode_pred, decode_target, is_aligned=True, eps=eps)
    #smooth l1 loss
    # diff = torch.abs(encode_pred - encode_target)
    # smooth_l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
    #                              diff - 0.5 * beta).sum(1)
    # assert beta > 0
    # assert encode_pred.size() == encode_target.size() and encode_target.numel() > 0
    # diff = torch.abs(encode_pred - encode_target)

    # b = np.e**(gamma / alpha) - 1
    # balanced_l1_loss = torch.where(
    #     diff < beta, alpha / b *
    #     (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
    #     gamma * diff + gamma / b - alpha * beta).sum(1) 
    #主要在这里修改
    #分多个步骤
    #第一个是IOU*
    #第二个是IOU
    #第三个是直线的IOU
    #第四个是曲线的IOU
    b1_iou_loss=torch.where(ious<0.5,1-ious,ious*ious-2*ious+1)
    #loss = balanced_iou_loss + balanced_l1_loss
    loss = b1_iou_loss
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def b2iou_loss(encode_decode_preds, encode_decode_targets, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6):
    encode_pred, decode_pred = encode_decode_preds
    encode_target, decode_target = encode_decode_targets
    ious = bbox_overlaps(decode_pred, decode_target, is_aligned=True, eps=eps)
    #smooth l1 loss
    # diff = torch.abs(encode_pred - encode_target)
    # smooth_l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
    #                              diff - 0.5 * beta).sum(1)
    # assert beta > 0
    # assert encode_pred.size() == encode_target.size() and encode_target.numel() > 0
    # diff = torch.abs(encode_pred - encode_target)

    # b = np.e**(gamma / alpha) - 1
    # balanced_l1_loss = torch.where(
    #     diff < beta, alpha / b *
    #     (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
    #     gamma * diff + gamma / b - alpha * beta).sum(1) 
    #主要在这里修改
    #分多个步骤
    #第一个是IOU*
    #第二个是IOU
    #第三个是直线的IOU
    #第四个是曲线的IOU
    b2_iou_loss=torch.where(ious<0,1-ious,1/log(2)*torch.pow(2,ious)-2*ious+2-2/log(2))
    #loss = balanced_iou_loss + balanced_l1_loss
    loss = b2_iou_loss
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def b3iou_loss(encode_decode_preds, encode_decode_targets, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6):
    encode_pred, decode_pred = encode_decode_preds
    encode_target, decode_target = encode_decode_targets
    ious = bbox_overlaps(decode_pred, decode_target, is_aligned=True, eps=eps)
    #smooth l1 loss
    # diff = torch.abs(encode_pred - encode_target)
    # smooth_l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
    #                              diff - 0.5 * beta).sum(1)
    # assert beta > 0
    # assert encode_pred.size() == encode_target.size() and encode_target.numel() > 0
    # diff = torch.abs(encode_pred - encode_target)

    # b = np.e**(gamma / alpha) - 1
    # balanced_l1_loss = torch.where(
    #     diff < beta, alpha / b *
    #     (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
    #     gamma * diff + gamma / b - alpha * beta).sum(1) 
    #主要在这里修改
    #分多个步骤
    #第一个是IOU*
    #第二个是IOU
    #第三个是直线的IOU
    #第四个是曲线的IOU
    b3_iou_loss=torch.where(ious<0.5,1-ious,1/2/log(4)*torch.pow(4,ious)-2*ious+2-2/log(4))
    #loss = balanced_iou_loss + balanced_l1_loss
    loss = b3_iou_loss
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def b4iou_loss(encode_decode_preds, encode_decode_targets, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6):
    encode_pred, decode_pred = encode_decode_preds
    encode_target, decode_target = encode_decode_targets
    ious = bbox_overlaps(decode_pred, decode_target, is_aligned=True, eps=eps)
    #smooth l1 loss
    # diff = torch.abs(encode_pred - encode_target)
    # smooth_l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
    #                              diff - 0.5 * beta).sum(1)
    # assert beta > 0
    # assert encode_pred.size() == encode_target.size() and encode_target.numel() > 0
    # diff = torch.abs(encode_pred - encode_target)

    # b = np.e**(gamma / alpha) - 1
    # balanced_l1_loss = torch.where(
    #     diff < beta, alpha / b *
    #     (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
    #     gamma * diff + gamma / b - alpha * beta).sum(1) 
    #主要在这里修改
    #分多个步骤
    #第一个是IOU*
    #第二个是IOU
    #第三个是直线的IOU
    #第四个是曲线的IOU
    b4_iou_loss=torch.where(ious<0.25,1-ious,3/4/log(2)/pow(2,1/3)*torch.pow(pow(2,4/3),ious)-2*ious+2-3/2/log(2))
    #loss = balanced_iou_loss + balanced_l1_loss
    loss = b4_iou_loss
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def b5iou_loss(encode_decode_preds, encode_decode_targets, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6):
    encode_pred, decode_pred = encode_decode_preds
    encode_target, decode_target = encode_decode_targets
    ious = bbox_overlaps(decode_pred, decode_target, is_aligned=True, eps=eps)
    #smooth l1 loss
    # diff = torch.abs(encode_pred - encode_target)
    # smooth_l1_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
    #                              diff - 0.5 * beta).sum(1)
    # assert beta > 0
    # assert encode_pred.size() == encode_target.size() and encode_target.numel() > 0
    # diff = torch.abs(encode_pred - encode_target)

    # b = np.e**(gamma / alpha) - 1
    # balanced_l1_loss = torch.where(
    #     diff < beta, alpha / b *
    #     (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
    #     gamma * diff + gamma / b - alpha * beta).sum(1) 
    #主要在这里修改
    #分多个步骤
    #第一个是IOU*
    #第二个是IOU
    #第三个是直线的IOU
    #第四个是曲线的IOU
    b5_iou_loss=torch.where(ious<0.75,1-ious,1/log(16)/pow(16,0.75)*torch.pow(16,ious)-2*ious+2-2/log(16))
    #loss = balanced_iou_loss + balanced_l1_loss
    loss = b5_iou_loss
    return loss



@LOSSES.register_module()
class BIoULoss(nn.Module):

    def __init__(self, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(BIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                encode_decode_preds,
                encode_decode_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(encode_decode_preds) == len(encode_decode_targets) == 2
        if weight is not None and not torch.any(weight > 0):
            return (encode_decode_preds[0] * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == encode_decode_preds[0].shape
            weight = weight.mean(-1)
            # print(torch.cat((pred[weight > 0], target[weight > 0]), axis=1))
        loss = self.loss_weight * biou_loss(
            encode_decode_preds,
            encode_decode_targets,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class B1IoULoss(nn.Module):

    def __init__(self, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(B1IoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                encode_decode_preds,
                encode_decode_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(encode_decode_preds) == len(encode_decode_targets) == 2
        if weight is not None and not torch.any(weight > 0):
            return (encode_decode_preds[0] * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == encode_decode_preds[0].shape
            weight = weight.mean(-1)
            # print(torch.cat((pred[weight > 0], target[weight > 0]), axis=1))
        loss = self.loss_weight * b1iou_loss(
            encode_decode_preds,
            encode_decode_targets,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss



@LOSSES.register_module()
class B2IoULoss(nn.Module):

    def __init__(self, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(B2IoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                encode_decode_preds,
                encode_decode_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(encode_decode_preds) == len(encode_decode_targets) == 2
        if weight is not None and not torch.any(weight > 0):
            return (encode_decode_preds[0] * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == encode_decode_preds[0].shape
            weight = weight.mean(-1)
            # print(torch.cat((pred[weight > 0], target[weight > 0]), axis=1))
        loss = self.loss_weight * b2iou_loss(
            encode_decode_preds,
            encode_decode_targets,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class B3IoULoss(nn.Module):

    def __init__(self, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(B3IoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                encode_decode_preds,
                encode_decode_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(encode_decode_preds) == len(encode_decode_targets) == 2
        if weight is not None and not torch.any(weight > 0):
            return (encode_decode_preds[0] * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == encode_decode_preds[0].shape
            weight = weight.mean(-1)
            # print(torch.cat((pred[weight > 0], target[weight > 0]), axis=1))
        loss = self.loss_weight * b3iou_loss(
            encode_decode_preds,
            encode_decode_targets,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class B4IoULoss(nn.Module):

    def __init__(self, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(B4IoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                encode_decode_preds,
                encode_decode_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(encode_decode_preds) == len(encode_decode_targets) == 2
        if weight is not None and not torch.any(weight > 0):
            return (encode_decode_preds[0] * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == encode_decode_preds[0].shape
            weight = weight.mean(-1)
            # print(torch.cat((pred[weight > 0], target[weight > 0]), axis=1))
        loss = self.loss_weight * b4iou_loss(
            encode_decode_preds,
            encode_decode_targets,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class B5IoULoss(nn.Module):

    def __init__(self, alpha=0.5,gamma=1.5,beta=1.0, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(B5IoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                encode_decode_preds,
                encode_decode_targets,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert len(encode_decode_preds) == len(encode_decode_targets) == 2
        if weight is not None and not torch.any(weight > 0):
            return (encode_decode_preds[0] * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == encode_decode_preds[0].shape
            weight = weight.mean(-1)
            # print(torch.cat((pred[weight > 0], target[weight > 0]), axis=1))
        loss = self.loss_weight * b5iou_loss(
            encode_decode_preds,
            encode_decode_targets,
            weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
