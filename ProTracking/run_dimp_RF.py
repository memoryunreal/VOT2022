# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import glob
import cv2
import torch
import numpy as np

from pytracking.refine_modules.refine_module import RefineModule
from pytracking.RF_utils import bbox_clip
from pytracking.evaluation import Tracker


parser = argparse.ArgumentParser(description='Pytracking-RF tracking')
parser.add_argument('--vis', action='store_true',default=False,
        help='whether to visualzie result')
parser.add_argument('--debug', action='store_true',default=False,
        help='whether to debug'),
parser.add_argument('--run_id',type=int, default=1)


args = parser.parse_args()
torch.set_num_threads(1)


class DBLoader(object):
    """ Debug Data Loader """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gt_file = os.path.join(self.data_dir, 'groundtruth.txt')
        self.curr_idx = 0
        self.im_paths = glob.glob(os.path.join(self.data_dir, 'color/*.jpg'))
        self.im_paths.sort()

    def region(self):
        return np.loadtxt(self.gt_file, dtype=np.float32, delimiter=',')[0]

    def frame(self):
        im_path = self.im_paths[self.curr_idx] if self.curr_idx < len(self.im_paths) else None
        print('pumping {}'.format(im_path))
        self.curr_idx += 1
        return im_path, None


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def get_dimp(img, init_box):
    # create tracker
    tracker_info = Tracker('dimp', 'super_dimp', None)
    params = tracker_info.get_parameters()
    params.visualization = args.vis
    params.debug = False
    params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
    tracker = tracker_info.tracker_class(params)

    H, W, _ = img.shape
    cx, cy, w, h = get_axis_aligned_bbox(np.array(init_box))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    '''Initialize'''
    gt_bbox_np = np.array(gt_bbox_)
    gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
    init_info = {}
    init_info['init_bbox'] = gt_bbox_torch
    tracker.initialize(img, init_info)

    return tracker


def main():
    debug_loader = DBLoader(data_dir='/home/zxy/Desktop/VOT_RGBD/data/RGBD19/box_room_occ_1')

    handle = debug_loader
    init_box = handle.region()
    imagefile, _ = handle.frame()
    img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    H, W, _ = img.shape

    tracker = get_dimp(img, init_box)

    """ Refinement module """
    refine_path = '/home/zxy/Desktop/VOT_RGBD/STARK_RGBD/checkpoints/ARcm_r34/SEcmnet_ep0040-a.pth.tar'
    selector_path = 0
    sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(refine_path, selector_path, search_factor=sr, input_sz=input_sz)
    RF_module.initialize(img, np.array(init_box))

    debug_loader = DBLoader(data_dir='/home/zxy/Desktop/VOT_RGBD/data/RGBD19/box_room_occ_1')
    handle = debug_loader

    # OPE tracking
    while True:
        imagefile, _ = handle.frame()
        if not imagefile:
            break
        img = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right

        """ Track """
        outputs = tracker.track(img)
        pred_bbox = outputs['target_bbox']

        # refine tracking results
        pred_bbox = RF_module.refine(img, np.array(pred_bbox))

        x1, y1, w, h = pred_bbox.tolist()
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (H, W))
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / tracker.base_target_sz.prod())

        # update
        tracker.pos = new_pos.clone()
        tracker.target_sz = new_target_sz
        tracker.target_scale = new_scale

        if args.vis:
            pred_bbox = list(map(int, pred_bbox))
            _img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.rectangle(_img, (pred_bbox[0], pred_bbox[1]),
                          (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
            cv2.imshow('', _img)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
