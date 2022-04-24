import torch
from lib.timm.models import create_model


def deit(img_sz: int, pretrained: bool, model_name="vit_deit_base_distilled_patch16_384",
         ckpt_name="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth"):
    if ckpt_name.startswith('https'):

        ckpt_name = '/home/VOT2022/vot2022/votrgbd/ProTracking/checkpoints/deit_base_distilled_patch16_384-d0272ac0.pth'
    model = create_model(model_name, pretrained=False, img_size=img_sz)
    if pretrained:
        if ckpt_name.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(ckpt_name, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(ckpt_name, map_location='cpu')

        checkpoint_model = checkpoint['model']

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=True)
    return model
