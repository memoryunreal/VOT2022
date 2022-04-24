"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn

from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .transformer_enc import build_transformer_enc  # encoder only
from .head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh


class STARKSPLUS(nn.Module):
    """ This is the base class for Transformer Tracking """
    def __init__(self, backbone, transformer, box_head, num_queries,
                 aux_loss=False, head_type="CORNER", fuse_method="POINTWISE"):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.box_head = box_head
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
        self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.fuse_method = fuse_method
        self.multipler = 1 / hidden_dim**0.5
        if "CORNER" in head_type:
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    def forward(self, img=None, seq_dict=None, mode="backbone", run_box_head=True, run_cls_head=False):
        if mode == "backbone":
            return self.forward_backbone(img)
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back, pos = self.backbone(input)  # features & masks, position embedding for the search
        # Adjust the shapes
        return self.adjust(output_back, pos)

    def forward_transformer(self, seq_dict, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], seq_dict["pos"])
        # Forward the corner head
        out, outputs_coord = self.forward_box_head(enc_mem)
        return out, outputs_coord, None

    def forward_box_head(self, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if "CORNER" in self.head_type:
            fx = memory[-self.feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, H_x*W_x, C)
            fz = memory[:-self.feat_len_s].transpose(0, 1)  # encoder output for the template (B, H_z*W_z, C)
            # feature fusion
            fx_t = fx.transpose(1, 2)
            vol = torch.matmul(fz, fx_t)  # (B, H_z*W_z, H_x*W_x)
            if self.fuse_method == "POINTWISE":
                opt_feat = vol.view(*vol.shape[:2], self.feat_sz_s, self.feat_sz_s)
            elif self.fuse_method == "SINSOFTMAX":
                vol = vol * self.multipler
                vol = torch.nn.functional.softmax(vol, dim=2)
                opt_feat = vol.view(*vol.shape[:2], self.feat_sz_s, self.feat_sz_s)
            elif self.fuse_method == "BISOFTMAX":
                # bidirectional softmax
                vol = vol * self.multipler
                vol2 = torch.nn.functional.softmax(vol, dim=2)
                vol1 = torch.nn.functional.softmax(vol, dim=1)
                vol = torch.cat([vol1, vol2], dim=1)  # (B, 2*H_z*W_z, H_x*W_x)
                opt_feat = vol.view(*vol.shape[:2], self.feat_sz_s, self.feat_sz_s)
            elif self.fuse_method == "ORIGIN":
                # search region feature output by the encoder
                opt_feat = fx_t.view(*fx_t.shape[:2], self.feat_sz_s, self.feat_sz_s)
            elif self.fuse_method == "BISOFTMAX_ORIGIN":
                # bidirectional softmax + search region feature output by the encoder
                vol = vol * self.multipler
                vol2 = torch.nn.functional.softmax(vol, dim=2)
                vol1 = torch.nn.functional.softmax(vol, dim=1)
                vol = torch.cat([vol1, vol2], dim=1)  # (B, 2*H_z*W_z, H_x*W_x)
                vol_map = vol.view(*vol.shape[:2], self.feat_sz_s, self.feat_sz_s)
                fx_map = fx_t.view(*fx_t.shape[:2], self.feat_sz_s, self.feat_sz_s)
                opt_feat = torch.cat([vol_map, fx_map], dim=1)  # ((B, 2*H_z*W_z+C, H_x*W_x))
            else:
                raise ValueError("Invalid fuse method")
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(opt_feat))
            out = {'pred_boxes': outputs_coord}
            return out, outputs_coord

    def adjust(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_starksplus(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer_enc(cfg)
    box_head = build_box_head(cfg)
    model = STARKSPLUS(
        backbone,
        transformer,
        box_head,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE,
        fuse_method=cfg.MODEL.FUSE_METHOD
    )

    return model
