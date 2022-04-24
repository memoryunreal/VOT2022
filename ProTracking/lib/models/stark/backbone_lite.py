"""
Backbone modules without positional encoding and mask
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
from lib.utils.misc import NestedTensor, is_main_process
from lib.models.stark import resnet as resnet_module
from .deit import deit
from .swin_transformer import build_swint


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool,
                 output_layer: str):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !
        self.body = backbone
        self.num_channels = num_channels

    def forward(self, inp: torch.Tensor):
        return self.body(inp)


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 freeze_bn: bool,
                 output_layer: str):
        norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        # here is different from the original DETR because we use feature from block3
        if output_layer == "layer3":
            backbone = getattr(resnet_module, name)(
                replace_stride_with_dilation=[False, dilation, False],
                pretrained=is_main_process(), norm_layer=norm_layer, last_layer='layer3')
            num_channels = 256 if name in ('resnet18', 'resnet34') else 1024
        elif output_layer == "layer4":
            backbone = getattr(resnet_module, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=norm_layer, last_layer='layer4')
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        else:
            raise ValueError("Invalid output layer")
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, output_layer)


class Transformer_Backbone(nn.Module):
    """Transformer Backbone"""

    def __init__(self, img_sz, model_name, train_backbone, ckpt_path=None, pretrained=True):
        super().__init__()
        if model_name == "vit_deit_base_distilled_patch16_384":
            ckpt_name = "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth"
            model = deit(img_sz, pretrained=is_main_process(), model_name=model_name, ckpt_name=ckpt_name)
            self.num_channels = model.num_features
        elif model_name == "swin_base_patch4_window12_384_S16":
            model = build_swint(img_sz, model_name, ckpt_path=ckpt_path, pretrained=pretrained)
            self.num_channels = model.num_features[-1]
        else:
            raise ValueError("Unsupported model name")

        if not train_backbone:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(False)
        self.body = model

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]
        out["output"] = NestedTensor(xs, mask)
        return out


def build_backbone(cfg, pretrained=True):
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    return_interm_layers = cfg.MODEL.PREDICT_MASK
    if 'resnet' in cfg.MODEL.BACKBONE.TYPE:
        backbone = Backbone(cfg.MODEL.BACKBONE.TYPE, train_backbone, return_interm_layers,
                            cfg.MODEL.BACKBONE.DILATION, cfg.TRAIN.FREEZE_BACKBONE_BN,
                            cfg.MODEL.BACKBONE.OUTPUT_LAYERS[0])
    elif 'vit' in cfg.MODEL.BACKBONE.TYPE:
        backbone = Transformer_Backbone(cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.TYPE, train_backbone)
    elif 'swin' in cfg.MODEL.BACKBONE.TYPE:
        if cfg.MODEL.BACKBONE.TYPE == "swin_base_patch4_window12_384_S16":
            ckpt_name = "swin_base_patch4_window12_384_22k.pth"
        else:
            raise ValueError("Unsupported model_name")
        if pretrained:
            ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
            backbone = Transformer_Backbone(cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.TYPE, train_backbone,
                                            ckpt_path=ckpt_path, pretrained=True)
        else:
            backbone = Transformer_Backbone(cfg.DATA.SEARCH.SIZE, cfg.MODEL.BACKBONE.TYPE, train_backbone,
                                            ckpt_path=None, pretrained=False)
    else:
        raise ValueError("Unsupported backbone type")
    model = backbone
    model.num_channels = backbone.num_channels
    return model
