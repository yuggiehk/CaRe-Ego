# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable
import os
from mmseg.registry import METRICS


@METRICS.register_module()
class NewSeperateIou(BaseMetric):
    """
    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta sc(int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = '',
                 output_vis_dir:Optional[str]='',
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.beta = beta
        self.output_dir = output_dir
        self.output_dir_gt = output_dir + '/gt'
        self.output_dir_hand = self.output_dir+'/hand'
        self.output_dir_left_obj = self.output_dir+'/left_obj'
        self.output_dir_right_obj = self.output_dir+'/right_obj'
        self.output_dir_two_obj = self.output_dir+'/two_obj'
        self.output_dir_vis = output_vis_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
            mkdir_or_exist(self.output_dir_hand)
            mkdir_or_exist(self.output_dir_left_obj)
            mkdir_or_exist(self.output_dir_right_obj)
            mkdir_or_exist(self.output_dir_two_obj)
            mkdir_or_exist(self.output_dir_gt)
            mkdir_or_exist(self.output_dir_vis)
        self.format_only = format_only
        self.results = []


    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        
        num_classes_hand = len(self.dataset_meta['class_hand'])
        num_classes_left_obj = len(self.dataset_meta['class_left_obj'])
        num_classes_right_obj = len(self.dataset_meta['class_right_obj'])
        num_classes_two_obj = len(self.dataset_meta['class_two_obj'])

        for i in range(len(data_batch)):
            img = data_batch['inputs'][i] # CHW

            img_path = data_batch['data_samples'][i].img_path
            filename = osp.splitext(osp.basename(
                    img_path))[0]
            img = img.clone().detach()
            img = img.to(torch.device('cpu'))


        for data_sample in data_samples:
            pred_label_hand = data_sample['pred_sem_seg_hand']['data'].squeeze()
            pred_label_left_obj = data_sample['pred_sem_seg_left_obj']['data'].squeeze()
            pred_label_right_obj = data_sample['pred_sem_seg_right_obj']['data'].squeeze()
            pred_label_two_obj = data_sample['pred_sem_seg_two_obj']['data'].squeeze()
            gt_img = data_sample['gt_sem_seg']['data'].squeeze()

        
            if not self.format_only:
                label_hand = data_sample['gt_sem_seg_hand']['data'].squeeze().to(pred_label_hand)
                label_left_obj = data_sample['gt_sem_seg_left_obj']['data'].squeeze().to(pred_label_left_obj)
                label_right_obj = data_sample['gt_sem_seg_right_obj']['data'].squeeze().to(pred_label_right_obj)
                label_two_obj = data_sample['gt_sem_seg_two_obj']['data'].squeeze().to(pred_label_two_obj)
                
                # self.results include(result_hand(intersection, union, histgram, histgram), result_obj(intersection, union, histgram, histgram))
                result_hand = self.intersect_and_union(pred_label_hand, label_hand, num_classes_hand, self.ignore_index)
                result_left_obj = self.intersect_and_union(pred_label_left_obj, label_left_obj, num_classes_left_obj, self.ignore_index)
                result_right_obj = self.intersect_and_union(pred_label_right_obj, label_right_obj, num_classes_right_obj, self.ignore_index)
                result_two_obj = self.intersect_and_union(pred_label_two_obj, label_two_obj, num_classes_two_obj, self.ignore_index)
              
                # this is to merge the hand output and obj output, we should delete 0,3,5,7 
                result_batch_intersection = torch.cat((result_hand[0], result_left_obj[0],result_right_obj[0], result_two_obj[0]),dim=0)
                result_batch_intersection = np.delete(np.array(result_batch_intersection), [0,3,5,7])
                result_batch_intersection = torch.from_numpy(result_batch_intersection)
    
                result_batch_union = torch.cat((result_hand[1], result_left_obj[1],result_right_obj[1], result_two_obj[1]),dim=0)
                result_batch_union = np.delete(np.array(result_batch_union), [0,3,5,7])
                result_batch_union = torch.from_numpy(result_batch_union)
        
                result_batch_pred = torch.cat((result_hand[2], result_left_obj[2],result_right_obj[2], result_two_obj[2]),dim=0)
                result_batch_pred = np.delete(np.array(result_batch_pred), [0,3,5,7])
                result_batch_pred = torch.from_numpy(result_batch_pred)
                result_batch_label = torch.cat((result_hand[3], result_left_obj[3],result_right_obj[3], result_two_obj[3]),dim=0)
                result_batch_label = np.delete(np.array(result_batch_label), [0,3,5,7])
                result_batch_label = torch.from_numpy(result_batch_label)

                result_batch = result_batch_intersection, result_batch_union, result_batch_pred, result_batch_label
                self.results.append(result_batch)      

            # save the visualization result
     
            if self.output_dir is not None:
                basename = osp.splitext(osp.basename(
                    data_sample['img_path']))[0]
                png_filename_hand = osp.abspath(
                    osp.join(self.output_dir_hand, f'{basename}.png'))
                png_filename_left_obj = osp.abspath(
                    osp.join(self.output_dir_left_obj, f'{basename}.png'))
                png_filename_right_obj = osp.abspath(
                    osp.join(self.output_dir_right_obj, f'{basename}.png'))
                png_filename_two_obj = osp.abspath(
                    osp.join(self.output_dir_two_obj, f'{basename}.png'))
                png_filename_gt = osp.abspath(
                    osp.join(self.output_dir_gt, f'{basename}.png')
                )
                output_mask_hand = np.array(pred_label_hand.cpu())
                output_mask_left_obj = np.array(pred_label_left_obj.cpu())
                output_mask_right_obj = np.array(pred_label_right_obj.cpu())
                output_mask_two_obj = np.array(pred_label_two_obj.cpu())

                # save the numpy prediction and GT
                gt_img = np.array(gt_img.cpu())
                output_hand = Image.fromarray((output_mask_hand).astype(np.uint8))
                output_left_obj = Image.fromarray((output_mask_left_obj).astype(np.uint8))
                output_right_obj = Image.fromarray((output_mask_right_obj).astype(np.uint8))
                output_two_obj = Image.fromarray((output_mask_two_obj).astype(np.uint8))
                gt_img = Image.fromarray((gt_img).astype(np.uint8))
                
                output_hand.save(png_filename_hand)
                output_left_obj.save(png_filename_left_obj)
                output_right_obj.save(png_filename_right_obj)
                output_two_obj.save(png_filename_two_obj)
                gt_img.save(png_filename_gt)

                # save the visualization results
                vis_pngname_all = osp.abspath(
                    osp.join(self.output_dir_vis, f'{basename}.png')
                )
                output_hand = np.array(output_hand)
                output_left_obj = np.array(output_left_obj)
                output_right_obj = np.array(output_right_obj)
                output_two_obj = np.array(output_two_obj)

                img_pngname = osp.abspath(
                    osp.join(self.output_dir, f'{basename}.jpg')
                )
                


    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        results = tuple(zip(*results))

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
       
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        class_names = self.dataset_meta['classes']
        
        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()

    
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / total_area_pred_label
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['Fscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall

        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })
        return ret_metrics
