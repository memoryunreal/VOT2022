class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/VOT2022/vot2022/votrgbd/MixFormer/pretrained_networks'
        self.lasot_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/lasot'
        self.got10k_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/got10k'
        self.lasot_lmdb_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/got10k_lmdb'
        self.trackingnet_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/trackingnet_lmdb'
        self.coco_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/coco'
        self.coco_lmdb_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/vid'
        self.imagenet_lmdb_dir = '/home/VOT2022/vot2022/votrgbd/MixFormer/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

