import torch.nn as nn
import math
from ltr.models.backbone.resnet import BasicBlock
from ltr.models.layers.blocks import conv_block


class ResidualDS16SW(nn.Module):
    def __init__(self, layer_dims, use_final_relu=True, use_bn=True):
        super().__init__()
        self.conv_block = conv_block(1, layer_dims[0], kernel_size=3, stride=2, padding=1, batch_norm=use_bn)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ds1 = nn.Conv2d(layer_dims[0], layer_dims[1], kernel_size=3, padding=1, stride=2)
        self.res1 = BasicBlock(layer_dims[0], layer_dims[1], stride=2, downsample=ds1, use_bn=use_bn)

        ds2 = nn.Conv2d(layer_dims[1], layer_dims[2], kernel_size=3, padding=1, stride=2)
        self.res2 = BasicBlock(layer_dims[1], layer_dims[2], stride=2, downsample=ds2, use_bn=use_bn)

        self.label_pred = conv_block(layer_dims[2], layer_dims[3], kernel_size=3, stride=1, padding=1,
                                     relu=use_final_relu, batch_norm=use_bn)

        self.samp_w_pred = nn.Conv2d(layer_dims[2], layer_dims[3], kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.samp_w_pred.weight.data.fill_(0)
        self.samp_w_pred.bias.data.fill_(1)

    def forward(self, label_mask, feature=None):
        # label_mask: frames, seq, h, w
        assert label_mask.dim() == 4

        label_shape = label_mask.shape
        label_mask = label_mask.view(-1, 1, *label_mask.shape[-2:])

        out = self.pool(self.conv_block(label_mask))
        out = self.res2(self.res1(out))

        label_enc = self.label_pred(out)
        sample_w = self.samp_w_pred(out)

        label_enc = label_enc.view(label_shape[0], label_shape[1], *label_enc.shape[-3:])
        sample_w = sample_w.view(label_shape[0], label_shape[1], *sample_w.shape[-3:])

        # Out dim is (num_seq, num_frames, layer_dims[-1], h, w)
        return label_enc, sample_w
