from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import vot
import sys
import time
import os
from .stark_ref_vot21lt import STARK_REF_LT


def run_vot_exp(base_tracker, base_param, ref_tracker, ref_param, use_new_box, save_dir, vis=False):
    torch.set_num_threads(1)
    dir_name = base_tracker + '_' + base_param + '--' + ref_tracker + '_' + ref_param
    save_root = os.path.join(save_dir, dir_name)
    tracker = STARK_REF_LT(base_tracker=base_tracker, base_param=base_param,
                           ref_tracker=ref_tracker, ref_param=ref_param, use_new_box=use_new_box)
    handle = vot.VOT("rectangle", "rgbd")
    selection = handle.region()
    imagefile, _ = handle.frame()
    init_box = [selection.x, selection.y, selection.width, selection.height]
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_dir = os.path.join(save_root, seq_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
    tracker.initialize(image, init_box)

    cnt = 0
    while True:
        imagefile, _ = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, conf = tracker.track(image)
        x1, y1, w, h = b1
        handle.report(vot.Rectangle(x1, y1, w, h), conf)
        if vis and (cnt % 5) == 0:
            '''Visualization'''
            # original image
            image_ori = image[:, :, ::-1].copy()  # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            # tracker box
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
        cnt += 1
