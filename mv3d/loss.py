import torch
import torch.nn as nn
import torch.nn.functional as F


class MAELoss(nn.Module):
    def forward(self, pred_depth_image, gt_depth_image, depth_interval):
        """
        non zero mean absolute loss for one batch
        COPIED FROM: https://github.com/callmeray/PointMVSNet/blob/master/pointmvsnet/utils/feature_fetcher.py
        """
        if gt_depth_image.shape != pred_depth_image.shape:  # resize gt depth image to same shape as prediction
            gt_depth_image = F.interpolate(gt_depth_image.unsqueeze(1), pred_depth_image.shape[-2:], mode='nearest')\
                .squeeze(1)
        mask_valid = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        denom = torch.sum(mask_valid, dim=(1, 2)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2))
        masked_mae = torch.mean((masked_mae / depth_interval) / denom)
        return masked_mae
