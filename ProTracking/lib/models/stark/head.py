import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.stark.backbone import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def FT_conv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
    """Fully-transformer convolution"""
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map):
        """ get soft-argmax coordinate for a given heatmap """
        prob_vec = nn.functional.softmax(
            score_map.view((-1, self.feat_sz * self.feat_sz)), dim=1)  # (batch, feat_sz * feat_sz)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


class FT_Corner_Predictor(nn.Module):
    """ Fully-transformer Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, stack_layers=4):
        super(FT_Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''stacked conv layers (FFNs)'''
        conv_tower_tl, conv_tower_br = [], []
        c_in, c_out = inplanes, channel
        for i in range(stack_layers):
            conv_tower_tl.append(FT_conv(c_in, c_out))
            conv_tower_br.append(FT_conv(c_in, c_out))
            c_in = c_out
            c_out = c_in
        conv_tower_tl.append(nn.Conv2d(c_in, 1, kernel_size=1))
        conv_tower_br.append(nn.Conv2d(c_in, 1, kernel_size=1))
        self.conv_tower_tl = nn.Sequential(*conv_tower_tl)
        self.conv_tower_br = nn.Sequential(*conv_tower_br)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        score_map_tl, score_map_br = self.conv_tower_tl(x), self.conv_tower_br(x)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map):
        """ get soft-argmax coordinate for a given heatmap """
        prob_vec = nn.functional.softmax(
            score_map.view((-1, self.feat_sz * self.feat_sz)), dim=1)  # (batch, feat_sz * feat_sz)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MLP_BN(nn.Module):
    """MLP with Batch Normalization"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k),
                                                  nn.BatchNorm1d(k)) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == "MLP":
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif cfg.MODEL.HEAD_TYPE in ["CORNER", "CENTER", "CORNER_FT"]:
        if cfg.MODEL.BACKBONE.OUTPUT_LAYERS[0] == "layer3":
            stride = 16
        elif cfg.MODEL.BACKBONE.OUTPUT_LAYERS[0] == "layer4":
            stride = 32
        else:
            raise ValueError("Invalid output layer")
        if cfg.MODEL.BACKBONE.DILATION:
            stride = stride // 2
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        # compute input dim
        input_dim_base = int((cfg.DATA.TEMPLATE.SIZE / stride) ** 2)
        fuse_method = getattr(cfg.MODEL, "FUSE_METHOD", None)
        if fuse_method == "POINTWISE":
            input_dim = input_dim_base
        elif fuse_method == "SINSOFTMAX":
            input_dim = input_dim_base
        elif fuse_method == "BISOFTMAX":
            input_dim = input_dim_base * 2
        elif fuse_method == "ORIGIN":
            input_dim = cfg.MODEL.HIDDEN_DIM
        elif fuse_method == "BISOFTMAX_ORIGIN":
            input_dim = input_dim_base * 2 + cfg.MODEL.HIDDEN_DIM
        elif fuse_method is None:
            input_dim = cfg.MODEL.HIDDEN_DIM
        else:
            raise ValueError("Invalid fuse method")
        # Build the head
        if cfg.MODEL.HEAD_TYPE == "CORNER":
            return Corner_Predictor(inplanes=input_dim, channel=256, feat_sz=feat_sz, stride=stride)
        elif cfg.MODEL.HEAD_TYPE == "CENTER":
            return Centernet_head(inplanes=input_dim, feat_sz=feat_sz)
        elif cfg.MODEL.HEAD_TYPE == "CORNER_FT":
            return FT_Corner_Predictor(inplanes=input_dim, channel=256, feat_sz=feat_sz, stride=stride)
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)


class Centernet_head(nn.Module):
    def __init__(self, inplanes, feat_sz, heads={'hm': 1, 'wh': 2, 'reg': 2}, head_conv=64, **kwargs):
        self.inplanes = inplanes
        self.feat_sz = feat_sz
        self.inplanes = 256
        self.deconv_with_bias = False
        self.heads = heads
        super(Centernet_head, self).__init__()
        # used for deconv layers
        """Here we use two deconv layers to adjust the stride from 16 to 4"""
        self.deconv_layers = self._make_deconv_layer(
            2,
            [256, 256],
            [4, 4],
        )
        # self.final_layer = []

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output,
                              kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        BN_MOMENTUM = 0.1
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def init_weights(self, num_layers, pretrained=True):
        # print('=> init resnet deconv weights from normal distribution')
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print('=> init {}.weight as 1'.format(name))
                # print('=> init {}.bias as 0'.format(name))
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # print('=> init final conv weights from normal distribution')
        for head in self.heads:
            final_layer = self.__getattr__(head)
            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    if m.weight.shape[0] == self.heads[head]:
                        if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]
