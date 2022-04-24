import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import ltr.models.layers.filter as filter_layer
from pytracking import TensorList


class LinearFilterSeg(nn.Module):
    def __init__(self, label_encoder=None, init_filter_reg=1e-2, upsample_residuals=False, score_act=None):
        super().__init__()
        self.label_encoder = label_encoder
        self.filter_reg = nn.Parameter(init_filter_reg * torch.ones(1))
        self.upsample_residuals = upsample_residuals

        if score_act is None:
            self.score_activation = None
        elif score_act == 'relu':
            self.score_activation = nn.ReLU()


    def forward(self, meta_parameter: TensorList, feat, mask, sample_weight=None,
                filter_dilation_factors=None):
        # Assumes multiple filters, i.e.  (sequences, filters, feat_dim, fH, fW)
        filter = meta_parameter[0]

        num_images = feat.shape[0]
        num_sequences = feat.shape[1] if feat.dim() == 5 else 1

        # Compute scores
        scores = filter_layer.apply_filter(feat, filter, dilation_factors=filter_dilation_factors)

        # Compute label map masks and weight
        if self.label_encoder is not None:
            raise NotImplementedError
            label_map = self.label_encoder(mask, feat)
        else:
            label_map = mask

        if sample_weight is None:
            sample_weight = math.sqrt(1.0 / num_images)
        elif isinstance(sample_weight, torch.Tensor):
            if sample_weight.numel() == scores.numel():
                sample_weight = sample_weight.view(scores.shape)
            elif sample_weight.dim() == 1:
                sample_weight = sample_weight.view(-1, 1, 1, 1, 1)

        if self.upsample_residuals:
            h,w = label_map.shape[-2:]
            scores_shape = scores.shape
            scores = scores.view(-1, 1, *scores.shape[-2:])
            scores = F.interpolate(scores, (h,w), mode='bilinear', align_corners=False)

            if sample_weight is not None and isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.view(-1, 1, *sample_weight.shape[-2:])
                sample_weight = F.interpolate(sample_weight, (h, w), mode='bilinear', align_corners=False)
                sample_weight = sample_weight.view(*scores_shape[:3], *sample_weight.shape[-2:])

            scores = scores.view(*scores_shape[:3], *scores.shape[-2:])
        label_map = label_map.view(scores.shape)

        if self.score_activation is not None:
            scores_act = self.score_activation(scores)
        else:
            scores_act = scores

        data_residual = sample_weight * (scores_act - label_map)

        # Compute regularization residual. Put batch in second dimension
        reg_residual = self.filter_reg*filter.view(1, num_sequences, -1)

        return TensorList([data_residual, reg_residual])
