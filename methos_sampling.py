import numpy as np
import torch
import math
import mmcv

from ..builder import BBOX_SAMPLERS
from .random_sampler import RandomSampler
from ..transforms import bbox2roi
from mmdet.core.bbox import bbox_overlaps
import torch.nn.functional as F  

def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    gt_widths = torch.abs(target[:, 2] - target[:, 0])
    gt_heights = torch.abs(target[:, 3] - target[:, 1])
    gt_areas = gt_widths * gt_heights
    # large
    large_index = gt_areas > 96*96
    #print("large_index:",(large_index==True).sum())
    # small
    small_index = gt_areas < 32*32
    #print("small_index:",(small_index==True).sum())
    # 这个操作比较迷？ middle
    middle_index=gt_areas>=32*32
    middle_index=gt_areas<=96*96
    #print((middle_index==True).sum())   
    return loss


@BBOX_SAMPLERS.register_module()
class IoUBalancedNegSampler(RandomSampler):
    """IoU Balanced Sampling.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Sampling proposals according to their IoU. `floor_fraction` of needed RoIs
    are sampled from proposals whose IoU are lower than `floor_thr` randomly.
    The others are sampled from proposals whose IoU are higher than
    `floor_thr`. These proposals are sampled from some bins evenly, which are
    split by `num_bins` via IoU evenly.

    Args:
        num (int): number of proposals.
        pos_fraction (float): fraction of positive proposals.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 **kwargs):
        super(IoUBalancedNegSampler, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins

    def sample_via_interval(self, max_overlaps, full_set, num_expected):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []
        #在一个给定的iou的区间内进行采样
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected negative samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                # for sampling interval calculation
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=np.int)
            
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            #r如果样本数不够的情况下，则全采样
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            #print(self.floor_thr)
            return sampled_inds


@BBOX_SAMPLERS.register_module()
class IoUBalancedNegSampler_lms(RandomSampler):
    """IoU Balanced Sampling.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Sampling proposals according to their IoU. `floor_fraction` of needed RoIs
    are sampled from proposals whose IoU are lower than `floor_thr` randomly.
    The others are sampled from proposals whose IoU are higher than
    `floor_thr`. These proposals are sampled from some bins evenly, which are
    split by `num_bins` via IoU evenly.

    Args:
        num (int): number of proposals.
        pos_fraction (float): fraction of positive proposals.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 **kwargs):
        super(IoUBalancedNegSampler_lms, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins

    def sample_via_interval(self, max_overlaps, full_set, num_expected,large_ratio=0.33,small_ratio=0.33,
                            large_inds=None,small_inds=None,middle_inds=None):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        #这里需要求出大中小物体的iou的最大值进行转换~,对应的索引已经有了，应该可以求出来
        #这里需要加一个判断，就是inds是否为0，为0的话就不考虑了~
        max_iou = max_overlaps.max()
        max_iou_large=0
        max_iou_small=0
        max_iou_middle=0

        if len(large_inds)!=0:
            max_iou_large=max_overlaps[large_inds].max()
        if len(small_inds)!=0:
            max_iou_small=max_overlaps[small_inds].max()
        if len(middle_inds)!=0:
            max_iou_middle=max_overlaps[middle_inds].max()
        ##这里结束

        iou_interval = (max_iou - self.floor_thr) / self.num_bins

        iou_interval_large=(max_iou_large - self.floor_thr) / self.num_bins
        iou_interval_small=(max_iou_small - self.floor_thr) / self.num_bins
        iou_interval_middle=(max_iou_middle - self.floor_thr) / self.num_bins

        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []
        #在一个给定的iou的区间内进行采样
        #这里我们统计大、中、小物体,根据gt应该获得的数量，在给定的区间内，选择对应的样本进行统计~
        large_num=int(large_ratio*num_expected)
        per_num_expected_large=int(large_num / self.num_bins)
        small_num=int(small_ratio*num_expected)
        per_num_expected_small=int(small_num / self.num_bins)
        middle_num=num_expected-small_num-large_num
        per_num_expected_middle=int(middle_num / self.num_bins)

        ##这里进行三个循环，大物体
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval_large
            end_iou = self.floor_thr + (i + 1) * iou_interval_large
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps[large_inds] >= start_iou,
                                   max_overlaps[large_inds] < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected_large:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected_large)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)
        #小物体
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval_small
            end_iou = self.floor_thr + (i + 1) * iou_interval_small
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps[small_inds] >= start_iou,
                                   max_overlaps[small_inds] < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected_small:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected_small)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)
        #中物体
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval_middle
            end_iou = self.floor_thr + (i + 1) * iou_interval_middle
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps[middle_inds] >= start_iou,
                                   max_overlaps[middle_inds] < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected_middle:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected_middle)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)
        #结束
        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, assign_result, num_expected, bboxes=None,large_ratio=0.33,small_ratio=0.33,**kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected negative samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                # for sampling interval calculation
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))

            #这进行大中小物体比例的统计，主要是采样边界框相关的内容
            #输入的是采样的边界框，输出的是大中小物体的索引
            gt_widths = torch.abs(bboxes[:, 2] - bboxes[:, 0])
            gt_heights = torch.abs(bboxes[:, 3] - bboxes[:, 1])
            gt_areas = gt_widths * gt_heights
            #middle_ratio=0.5

            #这个可能不太需要的样子~
            large_num=int(large_ratio*num_expected)
            small_num=int(small_ratio*num_expected)
            middle_num=num_expected-small_num-large_num


            # large
            large_area = gt_areas > 96*96
            large_index=torch.nonzero(large_area).view(-1).tolist()
            #print("sample_large_area:",(large_area==True).sum())
            #large_inds=large_index[:large_num]

            small_area = gt_areas < 32*32
            small_index=torch.nonzero(small_area).view(-1).tolist()
            #print("sample_small_area:",(small_area==True).sum())
            #small_inds=small_index[:small_num]

            middle_area=gt_areas>=32*32
            middle_area=gt_areas<=96*96
            middle_area=torch.where(large_area<small_area,small_area,large_area)
            middle_area=~middle_area
            middle_index=torch.nonzero(middle_area).view(-1).tolist()
            ##到这里为止

            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling,large_ratio,small_ratio,
                        large_inds=large_index,small_inds=small_index,middle_inds=middle_index)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=np.int)
            
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            #r如果样本数不够的情况下，则全采样
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            return sampled_inds


@BBOX_SAMPLERS.register_module()
class IoUBalancedNegSampler_IoU(RandomSampler):
    """IoU Balanced Sampling.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Sampling proposals according to their IoU. `floor_fraction` of needed RoIs
    are sampled from proposals whose IoU are lower than `floor_thr` randomly.
    The others are sampled from proposals whose IoU are higher than
    `floor_thr`. These proposals are sampled from some bins evenly, which are
    split by `num_bins` via IoU evenly.

    Args:
        num (int): number of proposals.
        pos_fraction (float): fraction of positive proposals.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 **kwargs):
        super(IoUBalancedNegSampler_IoU, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins

    def sample_via_interval(self, max_overlaps, full_set, num_expected):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)
        #对max_overlaps从大到小排序，选择的肯定是从大到小的内容，然后在进行选取即可
        max_overlaps = -np.sort(-max_overlaps)
        sampled_inds = []
        #在一个给定的iou的区间内进行采样
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = tmp_inds[:per_num_expected]
                # tmp_sampled_set = self.random_choice(tmp_inds,
                #                                      per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected negative samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                # for sampling interval calculation
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=np.int)
            
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            #r如果样本数不够的情况下，则全采样
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            #print(self.floor_thr)
            return sampled_inds


@BBOX_SAMPLERS.register_module()
class IoUBalancedNegSampler_GIoU(RandomSampler):
    """IoU Balanced Sampling.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Sampling proposals according to their IoU. `floor_fraction` of needed RoIs
    are sampled from proposals whose IoU are lower than `floor_thr` randomly.
    The others are sampled from proposals whose IoU are higher than
    `floor_thr`. These proposals are sampled from some bins evenly, which are
    split by `num_bins` via IoU evenly.

    Args:
        num (int): number of proposals.
        pos_fraction (float): fraction of positive proposals.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 xiou_thr = 0.15,
                 mode = 'iou',
                 **kwargs):
        super(IoUBalancedNegSampler_GIoU, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins
        self.xiou_thr = xiou_thr
        self.mode = mode 

    def sample_via_interval(self, max_overlaps, full_set, num_expected,bboxes,gt_bboxes):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []
        #在一个给定的iou的区间内进行采样
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                #在这里进行GIoU的loss计算
                #做个循环处理函数就可以了。
                #先计算每个的GIoU，取最大的

                gt_bboxes_num = len(gt_bboxes)
                bboxes_num = len(bboxes)
                eps=1e-7

                valid_giou_cnt = 0

                valid_indices = max_overlaps >= self.xiou_thr    
                selected_bboxes = bboxes[valid_indices]    
                selected_bboxes = torch.tensor(selected_bboxes).cuda()  
                bboxes_num = len(selected_bboxes)  

                gious = torch.zeros((bboxes_num,))

                # gious_cmp = torch.full((bboxes_num,), eps)

                # gious_cmp_2 = torch.full((bboxes_num,), eps)
                # for bboxes_i in range(0,bboxes_num):
                #     for gt_bboxes_j in range(0,gt_bboxes_num):
                #         gious_cmp[bboxes_i] = max(gious_cmp[bboxes_i],bbox_overlaps(selected_bboxes[bboxes_i].unsqueeze(0), gt_bboxes[gt_bboxes_j].unsqueeze(0), mode='giou', is_aligned=True, eps=eps))
                # # #这里我们忽略了一个问题，所以计算结果相对较低。


                # 首先，组合所有的selected_bboxes和gt_bboxes  
                # 这样我们可以一次性计算所有的GIoU  
                all_selected_bboxes = selected_bboxes.unsqueeze(1).repeat(1, gt_bboxes_num, 1)  
                all_gt_bboxes = gt_bboxes.unsqueeze(0).repeat(bboxes_num, 1, 1)  
                
                # 一次性计算所有的GIoU  
                gious_2 = bbox_overlaps(all_selected_bboxes, all_gt_bboxes, mode=self.mode, is_aligned=True, eps=eps)  
                
                # 现在，gious包含所有bbox对之间的GIoU，我们只需要按需求获取最大值  


                # gious_cmp_2 = gious_2.max(dim=1)[0]

                gious = gious_2.max(dim=1)[0]

                # # Pad selected_bboxes and gt_bboxes if needed to make their dimensions match  
                # max_num = max(bboxes_num, gt_bboxes_num)  
                # selected_bboxes_padded = F.pad(selected_bboxes, (0, 0, 0, max_num - bboxes_num))  
                # gt_bboxes_padded = F.pad(gt_bboxes, (0, 0, 0, max_num - gt_bboxes_num))  
                
                # # Calculate giou for all combinations  
                # gious_matrix = bbox_overlaps(selected_bboxes_padded.unsqueeze(0), gt_bboxes_padded.unsqueeze(0), mode='giou', is_aligned=True, eps=eps)  
                
                # # For each bbox in selected_bboxes, find the max giou with gt_bboxes  
                # gious = gious_matrix.max(dim=0)[0]
             
  
                
                # # 我们使用向量化操作来计算 gious    
                # gious = bbox_overlaps(selected_bboxes, gt_bboxes, mode='giou', is_aligned=True, eps=eps)    
                # gious = gious.max(dim=1)[0]  # 取每行的最大值 
                
                #就是我们一开始选择的内容
                valid_giou_cnt = len(selected_bboxes)  

                # gious = bbox_overlaps(bboxes[0].unsqueeze(0), gt_bboxes[0].unsqueeze(0), mode='giou', is_aligned=True, eps=eps)

                # print("per_num_expected=",per_num_expected)
                # print("valid_giou_cnt=",valid_giou_cnt)
                if valid_giou_cnt >= per_num_expected:
                    _, indices = torch.topk(gious, per_num_expected)
                    tmp_sampled_set = indices.cpu().numpy()
                else:
                    _, indices = torch.topk(gious, valid_giou_cnt)
                    tmp_sampled_set = indices.cpu().numpy()


                # if len(tmp_sampled_set) >= per_num_expected: 
                #     _, indices = torch.topk(gious, per_num_expected)
                #     tmp_sampled_set = indices.numpy()
                # else:
                #     tmp_sampled_set = self.random_choice(tmp_inds,
                #                                      per_num_expected)
                

            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)

            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            #这里具体具体也可以添加一些内容。即相对较小的情况下的采样方法，我们想在这里添加PISA的一些想法的，所以先不在这里改。
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:#在这里进行比较

                extra_inds = self.random_choice(extra_inds, num_extra)



            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, assign_result,num_expected,bboxes,gt_bboxes,**kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected negative samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                # for sampling interval calculation
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling,bboxes,gt_bboxes)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=np.int)
            
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            #r如果样本数不够的情况下，则全采样
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            #print(self.floor_thr)
            return sampled_inds


@BBOX_SAMPLERS.register_module()
class IoUBalancedNegSampler_Ratio(RandomSampler):
    """IoU Balanced Sampling.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Sampling proposals according to their IoU. `floor_fraction` of needed RoIs
    are sampled from proposals whose IoU are lower than `floor_thr` randomly.
    The others are sampled from proposals whose IoU are higher than
    `floor_thr`. These proposals are sampled from some bins evenly, which are
    split by `num_bins` via IoU evenly.

    Args:
        num (int): number of proposals.
        pos_fraction (float): fraction of positive proposals.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 **kwargs):
        super(IoUBalancedNegSampler_Ratio, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins

    def sample_via_interval(self, max_overlaps, full_set, num_expected):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []

        random_sample_ratio = 0

        random_iou1_sample = 0
        random_iou1_sample_ratio = 0

        random_iou2_sample = 0
        random_iou2_sample_ratio = 0

        random_iou3_sample = 0
        random_iou3_sample_ratio = 0

        #在一个给定的iou的区间内进行采样
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)

            random_sample_ratio = num_extra / num_expected
            
            f_count = self.num_bins
            # fname = "/home/zbliu/ustcthesis-chap3-exper/mmdet/core/bbox/samplers/random_sample_ratio_num_bins_" + str(f_count) +".txt"
            fname = "/ghome/liuzb/mmdetection/random_sample_ratio_num_bins_" + str(f_count) +".txt"

            f = open(fname,"a")
            f.write(str(random_sample_ratio)+'\n')
            f.close()            

            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected negative samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                # for sampling interval calculation
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=np.int)
            
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            #r如果样本数不够的情况下，则全采样，实际上我们已经进行全采样了的，不是在这里进行处理的。
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            #print(self.floor_thr)
            return sampled_inds
        


@BBOX_SAMPLERS.register_module()
class IoUBalancedNegSampler_OHEM(RandomSampler):
    """IoU Balanced Sampling.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Sampling proposals according to their IoU. `floor_fraction` of needed RoIs
    are sampled from proposals whose IoU are lower than `floor_thr` randomly.
    The others are sampled from proposals whose IoU are higher than
    `floor_thr`. These proposals are sampled from some bins evenly, which are
    split by `num_bins` via IoU evenly.

    Args:
        num (int): number of proposals.
        pos_fraction (float): fraction of positive proposals.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 context,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 ohem_ratio = 1,
                 **kwargs):
        super(IoUBalancedNegSampler_OHEM, self).__init__(num, pos_fraction,
                                                    **kwargs)
        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins
        self.ohem_ratio = ohem_ratio

        self.context = context
        if not hasattr(self.context, 'num_stages'):
            self.bbox_head = self.context.bbox_head
        else:
            self.bbox_head = self.context.bbox_head[self.context.current_stage]

    def sample_via_interval(self, max_overlaps, full_set, num_expected,
                            assign_result,                    
                            bboxes,
                            feats=None,):
        """Sample according to the iou interval.

        Args:
            max_overlaps (torch.Tensor): IoU between bounding boxes and ground
                truth boxes.
            full_set (set(int)): A full set of indices of boxes。
            num_expected (int): Number of expected samples。

        Returns:
            np.ndarray: Indices  of samples
        """
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []

        #OHEM相关
        # neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        # neg_labels = assign_result.labels.new_empty(
        #         neg_inds.size(0)).fill_(self.bbox_head.num_classes)
        
        #在一个给定的iou的区间内进行采样
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            # 随机后的结果进行 ohem采样
            num_extra = num_expected - len(sampled_inds)
            ohem_num = math.ceil(num_extra * self.ohem_ratio)
            rand_num = num_extra - ohem_num
            #首先进行ohem采样，之后再随机采样
            #有个问题，怎么剔除已经采样的内容，在剩下可以选择的里面进行操作？这里需要调试一下
            #用这个方法
            # extra_inds = np.array(list(full_set - set(sampled_inds))) #获得额外的信息
            #获得对应标签
            neg_inds = torch.nonzero(assign_result.gt_inds == 0).flatten()

            # neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)

            neg_labels = assign_result.labels.new_empty(
                neg_inds.size(0)).fill_(self.bbox_head.num_classes)
            
            ohem_inds = self.hard_mining(neg_inds, ohem_num, bboxes[neg_inds],
                                    neg_labels, feats)
            ohem_inds = ohem_inds.cpu().numpy()
            #获得剩下的标签
            sampled_inds = np.concatenate((sampled_inds, ohem_inds), axis=0)
            
            # sampled_inds.append(ohem_inds)
            #随机采样
            extra_inds = np.array(list(full_set - set(sampled_inds))) #获得额外的信息
            if len(extra_inds) > rand_num:
                extra_inds = self.random_choice(extra_inds, rand_num)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def hard_mining(self, inds, num_expected, bboxes, labels, feats):
        with torch.no_grad():
            rois = bbox2roi([bboxes])
            if not hasattr(self.context, 'num_stages'):
                bbox_results = self.context._bbox_forward(feats, rois)
            else:
                bbox_results = self.context._bbox_forward(
                    self.context.current_stage, feats, rois)
            cls_score = bbox_results['cls_score']
            loss = self.bbox_head.loss(
                cls_score=cls_score,
                bbox_pred=None,
                rois=rois,
                labels=labels,
                label_weights=cls_score.new_ones(cls_score.size(0)),
                bbox_targets=None,
                bbox_weights=None,
                reduction_override='none')['loss_cls']
            _, topk_loss_inds = loss.topk(num_expected)
        return inds[topk_loss_inds]


    def _sample_neg(self, assign_result, num_expected,                     
                    bboxes=None,
                    feats=None,**kwargs):
        """Sample negative boxes.

        Args:
            assign_result (:obj:`AssignResult`): The assigned results of boxes.
            num_expected (int): The number of expected negative samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0]) s
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                # for sampling interval calculation
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling,assign_result,bboxes,feats)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=np.int)
            
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            #r如果样本数不够的情况下，则全采样,我们统计的应该是这里的，之前我们写错了应该，那就两个实验一起做了。

            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            #print(self.floor_thr)
            return sampled_inds
