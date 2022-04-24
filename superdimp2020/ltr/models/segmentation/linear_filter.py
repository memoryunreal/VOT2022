import torch.nn as nn
import ltr.models.layers.filter as filter_layer
import math
from pytracking import TensorList


class LinearFilter(nn.Module):
    def __init__(self, filter_size, filter_initializer, filter_optimizer=None, feature_extractor=None,
                 filter_dilation_factors=None):
        super().__init__()

        self.filter_size = filter_size

        # Modules
        self.filter_initializer = filter_initializer
        self.filter_optimizer = filter_optimizer

        self.feature_extractor = feature_extractor

        self.filter_dilation_factors = filter_dilation_factors

        # Init weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, train_feat, test_feat, train_mask, *args, **kwargs):
        """ the mask should be 5d"""
        assert train_mask.dim() == 5

        num_sequences = train_mask.shape[1]

        if train_feat.dim() == 5:
            train_feat = train_feat.view(-1, *train_feat.shape[-3:])
        if test_feat.dim() == 5:
            test_feat = test_feat.view(-1, *test_feat.shape[-3:])

        # Extract features
        train_feat = self.extract_classification_feat(train_feat, num_sequences)
        test_feat = self.extract_classification_feat(test_feat, num_sequences)

        # Train filter
        filter, filter_iter, _ = self.get_filter(train_feat, train_mask,
                                                 *args, **kwargs)

        # Classify samples
        test_scores = [self.classify(f, test_feat) for f in filter_iter]

        return test_scores

    def extract_classification_feat(self, feat, num_sequences=None):
        if self.feature_extractor is None:
            return feat
        if num_sequences is None:
            return self.feature_extractor(feat)

        output = self.feature_extractor(feat)
        return output.view(-1, num_sequences, *output.shape[-3:])

    def classify(self, weights, feat):
        """Run classifier (filter) on the features (feat)."""

        scores = filter_layer.apply_filter(feat, weights, dilation_factors=self.filter_dilation_factors)

        return scores

    def get_filter(self, feat, train_mask_info, num_objects=None, *args, **kwargs):
        if isinstance(train_mask_info, (tuple, list)):
            train_mask = train_mask_info[0]
            sample_weight = train_mask_info[1]
        else:
            train_mask = train_mask_info
            sample_weight = None

        if num_objects is None:
            weights = self.filter_initializer(feat, train_mask)
        else:
            weights = self.filter_initializer(feat, train_mask)
            weights = weights.repeat(1, num_objects, 1, 1, 1)

        if self.filter_optimizer is not None:
            weights, weights_iter, losses = self.filter_optimizer(TensorList([weights]), feat=feat, mask=train_mask,
                                                                  sample_weight=sample_weight,
                                                                  *args, **kwargs)
            weights = weights[0]
            weights_iter = [w[0] for w in weights_iter]
        else:
            weights_iter = [weights]
            losses = None

        return weights, weights_iter, losses
