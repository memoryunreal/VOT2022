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
from lib.test.vot20.stark_vot20lt import stark_vot20_lt


def run_vot_exp(tracker_name, para_name, vis=False):
    torch.set_num_threads(1)
    save_root = os.path.join('/data/sda/v-yanbi/iccv21/LittleBoy/vot20_rgbd_debug', para_name)
    if vis and (not os.path.exists(save_root)):
        os.makedirs(save_root)
    tracker = stark_vot20_lt(tracker_name=tracker_name, para_name=para_name)
    handle = vot.VOT("rectangle", "rgbd")
    selection = handle.region()
    imagefile, _ = handle.frame()
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
        imagefile, _ = handle.frame()
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
