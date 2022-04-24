from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import torch
import vot
import sys
import time
from lib.test.evaluation import Tracker


class STARK_REF_LT(object):
    """STARK base tracker + STARK refinement"""
    def __init__(self, base_tracker='stark_st', base_param='baseline',
                 ref_tracker='stark_ref', ref_param='baseline', use_new_box=True):
        """use_new_box: whether to use the refined box as the new state"""
        self.use_new_box = use_new_box
        # create base tracker
        tracker_info = Tracker(base_tracker, base_param, "lasot", None)
        base_params = tracker_info.get_parameters()
        base_params.visualization, base_params.debug = False, False
        self.tracker = tracker_info.create_tracker(base_params)
        # create refinement module
        ref_info = Tracker(ref_tracker, ref_param, "lasot", None)
        ref_params = ref_info.get_parameters()
        ref_params.visualization, ref_params.debug = False, False
        self.ref = ref_info.create_tracker(ref_params)

    def initialize(self, img_rgb, box):
        """box: list"""
        # init on the 1st frame
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': box}
        # init base tracker
        _ = self.tracker.initialize(img_rgb, init_info)
        # init refinement module
        self.ref.initialize(img_rgb, init_info)

    def track(self, img_rgb):
        # track with the base tracker
        outputs, update_flag = self.tracker.track(img_rgb, return_update_flag=True)
        pred_bbox = outputs['target_bbox']
        conf_score = outputs["conf_score"]
        # refine with the refinement module
        outputs_ref = self.ref.track(img_rgb, pred_bbox, update_flag)
        pred_bbox_ref = outputs_ref["target_bbox"]
        if self.use_new_box:
            self.tracker.state = pred_bbox_ref
        return pred_bbox_ref, conf_score


def run_vot_exp(base_tracker, base_param, ref_tracker, ref_param, use_new_box, vis=False):
    torch.set_num_threads(1)
    dir_name = base_tracker + '_' + base_param + '___' + ref_tracker + '_' + ref_param
    save_root = os.path.join('/data/sda/v-yanbi/iccv21/STARK/vot21_lt_debug', dir_name)
    if vis and (not os.path.exists(save_root)):
        os.makedirs(save_root)
    tracker = STARK_REF_LT(base_tracker=base_tracker, base_param=base_param,
                           ref_tracker=ref_tracker, ref_param=ref_param, use_new_box=use_new_box)
    handle = vot.VOT("rectangle")
    selection = handle.region()
    imagefile = handle.frame()
    init_box = [selection.x, selection.y, selection.width, selection.height]
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root, seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    tracker.initialize(image, init_box)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, conf = tracker.track(image)
        x1, y1, w, h = b1
        handle.report(vot.Rectangle(x1, y1, w, h), conf)
        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:, :, ::-1].copy()  # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, image_ori)
            # tracker box
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg', '_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
