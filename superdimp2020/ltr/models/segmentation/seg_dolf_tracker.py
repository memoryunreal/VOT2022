import math
import torch
import torch.nn as nn
from collections import OrderedDict
import ltr.models.segmentation.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.segmentation.initializer as seg_initializer
import ltr.models.segmentation.label_encoder as seg_label_encoder
import ltr.models.segmentation.loss_residual_modules as loss_residual_modules
import ltr.models.segmentation.dolf_decoder as dolf_decoder
import ltr.models.backbone as backbones
import ltr.models.backbone.resnet_mrcnn as mrcnn_backbones
import ltr.models.meta.steepestdescent as steepestdescent
from ltr import model_constructor
from pytracking import TensorList


class SegDolfTracker(nn.Module):
    def __init__(self, feature_extractor, classifier, decoder, classification_layer, refinement_layers,
                 label_encoder=None, aux_layers=None, bb_regressor=None, bbreg_decoder_layer=None):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.decoder = decoder

        # self.output_layers = ['layer1', 'layer2', 'layer3']

        self.classification_layer = (classification_layer,) if isinstance(classification_layer,
                                                                         str) else classification_layer
        self.refinement_layers = refinement_layers
        self.output_layers = sorted(list(set(self.classification_layer + self.refinement_layers)))
        # self.classification_layer = ['layer3']
        self.label_encoder = label_encoder

        if aux_layers is None:
            self.aux_layers = nn.ModuleDict()
        else:
            self.aux_layers = aux_layers

        self.bb_regressor = bb_regressor
        self.bbreg_decoder_layer = bbreg_decoder_layer

    def forward(self, train_imgs, test_imgs, train_masks, test_masks, num_refinement_iter=2):
        pass

    def segment_target(self, target_filter, test_feat_clf, test_feat):
        # Classification features
        assert target_filter.dim() == 5     # seq, filters, ch, h, w
        test_feat_clf = test_feat_clf.view(1, 1, *test_feat_clf.shape[-3:])

        target_scores = self.classifier.classify(target_filter, test_feat_clf)

        mask_pred, decoder_feat = self.decoder(target_scores, test_feat,
                                               (test_feat_clf.shape[-2]*16, test_feat_clf.shape[-1]*16),
                                               (self.bbreg_decoder_layer, ))

        bb_pred = None
        if self.bb_regressor is not None:
            bb_pred = self.bb_regressor(decoder_feat[self.bbreg_decoder_layer])
            bb_pred[:, :2] *= test_feat_clf.shape[-2] * 16
            bb_pred[:, 2:] *= test_feat_clf.shape[-1] * 16
            bb_pred = torch.stack((bb_pred[:, 2], bb_pred[:, 0],
                                   bb_pred[:, 3] - bb_pred[:, 2],
                                   bb_pred[:, 1] - bb_pred[:, 0]), dim=1)

        decoder_feat['mask_enc'] = target_scores.view(-1, *target_scores.shape[-3:])
        aux_mask_pred = {}
        if 'mask_enc_iter' in self.aux_layers.keys():
            aux_mask_pred['mask_enc_iter'] = \
                self.aux_layers['mask_enc_iter'](target_scores.view(-1, *target_scores.shape[-3:]), (test_feat_clf.shape[-2]*16,
                                                                               test_feat_clf.shape[-1]*16))
        # Output is 1, 1, h, w
        return mask_pred, bb_pred, aux_mask_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]
        return feat

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        if layers is None:
            layers = self.output_layers + ['classification']
        if 'classification' not in layers:
            return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})


@model_constructor
def steepest_descent_resnet50(filter_size=1, num_filters=1, optim_iter=3, optim_init_reg=0.01,
                              backbone_pretrained=False, clf_feat_blocks=1,
                              clf_feat_norm=True, final_conv=False,
                              out_feature_dim=512,
                              classification_layer='layer3',
                              refinement_layers=("layer4", "layer3", "layer2", "layer1",),
                              detach_length=float('Inf'),
                              label_encoder_dims=(1, 1),
                              frozen_backbone_layers=(),
                              label_encoder_type='identity',
                              decoder_mdim=64, filter_groups=1,
                              upsample_residuals=True,
                              use_bn_in_label_enc=True,
                              cls_feat_extractor='ppm',
                              decoder_type='rofl',
                              dilation_factors=None,
                              use_final_relu=True,
                              bb_regressor_type=None,
                              backbone_type='imagenet'):
    # backbone
    if backbone_type == 'imagenet':
        backbone_net = backbones.resnet50(pretrained=backbone_pretrained, frozen_layers=frozen_backbone_layers)
    elif backbone_type == 'mrcnn':
        backbone_net = mrcnn_backbones.resnet50(pretrained=False, frozen_layers=frozen_backbone_layers)
    else:
        raise Exception

    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # classifier
    layer_channels = backbone_net.out_feature_channels()

    if cls_feat_extractor == 'res_block':
        clf_feature_extractor = clf_features.residual_basic_block(feature_dim=layer_channels[classification_layer],
                                                                  num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                                  final_conv=final_conv, norm_scale=norm_scale,
                                                                  out_dim=out_feature_dim)
    else:
        raise Exception

    initializer = seg_initializer.FilterInitializerZero(filter_size=filter_size, num_filters=num_filters,
                                                        feature_dim=out_feature_dim, filter_groups=filter_groups)

    if label_encoder_type == 'res_ds16_sw':
        label_encoder = seg_label_encoder.ResidualDS16SW(layer_dims=label_encoder_dims + (num_filters, ),
                                                         use_bn=use_bn_in_label_enc,
                                                         use_final_relu=use_final_relu)
    else:
        raise Exception

    residual_module = loss_residual_modules.LinearFilterSeg(init_filter_reg=optim_init_reg, upsample_residuals=upsample_residuals)

    optimizer = steepestdescent.GNSteepestDescent(residual_module=residual_module, num_iter=optim_iter, detach_length=detach_length,
                                                  residual_batch_dim=1, compute_losses=True,
                                                  filter_dilation_factors=dilation_factors)

    if dilation_factors is not None:
        assert num_filters == sum(dilation_factors.values())

    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor,
                                         filter_dilation_factors=dilation_factors)

    refinement_layers_channels = {L: layer_channels[L] for L in refinement_layers}

    if decoder_type == 'rofl':
        decoder = dolf_decoder.RefelixNetwork2(num_filters, decoder_mdim, refinement_layers_channels,
                                               new_upsampler=True, use_bn=True)
    else:
        raise Exception

    aux_layers = None

    bbreg_decoder_layer = None
    if bb_regressor_type is None:
        bb_regressor = None
    else:
        raise Exception

    net = SegDolfTracker(feature_extractor=backbone_net, classifier=classifier, decoder=decoder,
                         label_encoder=label_encoder,
                         bb_regressor=bb_regressor,
                         classification_layer=classification_layer, refinement_layers=refinement_layers,
                         aux_layers=aux_layers,
                         bbreg_decoder_layer=bbreg_decoder_layer)
    return net
