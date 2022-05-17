class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/gaoshang/pytracking/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        # self.tensorboard_dir = '/home/gaoshang/pytracking' + '/tensorboard/'    # Directory for tensorboard files.
        
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lasot_dir = ''
        self.got10k_dir = '/data1/gaoshang/GOT-10k/full_data/train_data'
        self.trackingnet_dir = ''
        self.coco_dir = '/data1/gaoshang/coco'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = '' 
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        # self.davis_dir = '/data1/yjy/VOS/DepthVOST_train'
        #self.davis16_dir = '/data1/yjy/VOS/Youtube-VOS/2019/DepthVOST_train'
        self.davis16_dir = '/home/gaoshang/pytracking/DAVIS/'#'/home/yangjinyu/STM/DAVIS'
        # self.youtubevos_dir = '/data1/yjy/VOS/Youtube-VOS/2019/DepthVOST_train'
