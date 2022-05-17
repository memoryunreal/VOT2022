from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import vot2022 as vot
import sys
import time
import os
import numpy as np
sys.path.append("/home/VOT2022/vot2022/votrgbd/MixFormer/")
sys.path.append("/home/VOT2022/vot2022/votrgbd/MixFormer/external/AR/")
from lib.test.tracker.mixformer_online import MixFormerOnline
from pytracking.ARcm_seg import ARcm_seg
from pytracking.vot20_utils import *
import argparse
import lib.test.parameter.mixformer_online as vot_params


from rgbd_blend import rgbd_blend

class MIXFORMER_ALPHA_SEG(object):
    def __init__(self, tracker,
                 refine_model_name='ARcm_coco_seg', threshold=0.6):
        self.THRES = threshold
        self.tracker = tracker
        '''create tracker'''
        '''Alpha-Refine'''
        project_path = os.path.join(os.path.dirname(__file__), '..', '..')
        refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/ARcm_seg/')
        refine_path = os.path.join(refine_root, refine_model_name)
        '''2020.4.25 input size: 384x384'''
        self.alpha = ARcm_seg(refine_path, input_sz=384)

    def initialize(self, image, mask):
        # mask
        # region = rect_from_mask(mask)
        # init_info = {'init_bbox': region}
        # self.tracker.initialize(image, init_info)
        region = mask
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize STARK for specific video'''
        init_info = {'init_bbox': list(gt_bbox_np)}
        self.tracker.initialize(image, init_info)
        '''initilize refinement module for specific video'''
        self.alpha.initialize(image, np.array(gt_bbox_np))

    def track(self, img_RGB):
        '''TRACK'''
        '''base tracker'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        '''Step2: Mask report'''
        pred_mask, search, search_mask = self.alpha.get_mask(img_RGB, np.array(pred_bbox), vis=True)
        final_mask = (pred_mask > self.THRES).astype(np.uint8)
        return final_mask, 1


def make_full_size(x, output_sz):
    '''
    zero-pad input x (right and down) to match output_sz
    x: numpy array e.g., binary mask
    output_sz: size of the output [width, height]
    '''
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        # padding has to be set to zero, otherwise pad function fails
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        # padding has to be set to zero, otherwise pad function fails
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)

parser = argparse.ArgumentParser(description='GPU selection and SRE selection', prog='tracker')
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--depthvalue", default=5000, type=int)
parser.add_argument("--blend", default=0.05, type=float)
parser.add_argument("--colormap", default="JET", type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


refine_model_name = 'ARcm_coco_seg_only_mask_384'
# params = vot_params.parameters("baseline_large")
params = vot_params.parameters("baseline_large", model="mixformerL_online_22k.pth.tar")
mixformer = MixFormerOnline(params, "VOT20")
tracker = MIXFORMER_ALPHA_SEG(tracker=mixformer, refine_model_name=refine_model_name)
# handle = vot.VOT("mask")
handle = vot.VOT("rectangle", channels='rgbd')
selection = handle.region()
# imagefile = handle.frame()

imagefile, depthfile = handle.frame()

if not imagefile:
    sys.exit(0)
'''
blend
'''
depthth = args.depthvalue
blend = args.blend
style = args.colormap
if not depthth == 0 and not blend == 0.0:
    image = rgbd_blend(imagefile, depthfile, depthth, blend, style)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
else:
    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
'''
    blend
'''
# image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
# mask given by the toolkit ends with the target (zero-padding to the right and down is needed)
# mask = make_full_size(selection, (image.shape[1], image.shape[0]))

tracker.H = image.shape[0]
tracker.W = image.shape[1]

# tracker.initialize(image, mask)
tracker.initialize(image, selection)

while True:
    # imagefile = handle.frame()
    imagefile, depthfile = handle.frame()
    if not imagefile:
        break
    # image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    '''
    blend
    '''
    if not depthth == 0 and not blend == 0.0:
        image = rgbd_blend(imagefile, depthfile, depthth, blend, style)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    '''
        blend
    '''
    region, confidence = tracker.track(image)
    try:
        tmp = rect_from_mask(region)
        tmp1 = tmp
    except:
        tmp = tmp1
    region = vot.Rectangle(tmp[0],tmp[1],tmp[2],tmp[3])
    handle.report(region, confidence)
