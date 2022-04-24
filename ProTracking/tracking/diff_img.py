import _init_paths
import cv2
import numpy as np
import os
from lib.test.evaluation import get_dataset

fps = 30
dataset = get_dataset("lasot")
# dataset = get_dataset("vot20rgbd")
save_root = "diff_exp"
for seq in dataset:
    if seq.name == "airplane-1":
    # if seq.name == "backpack_blue":
        # save_dir = os.path.join(save_root, seq.name)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        videoWriter = cv2.VideoWriter('%s.avi' % seq.name, fourcc, fps, (640, 720))
        pre_path = None
        for idx, cur_path in enumerate(seq.frames):
            if idx == 0:
                pre_path = cur_path
            if idx > 0:
                img_cur = cv2.imread(cur_path)
                img_pre = cv2.imread(pre_path)
                img_cur_f = img_cur.astype(np.float)
                img_pre_f = img_pre.astype(np.float)
                diff = np.abs(img_cur_f - img_pre_f)
                diff_img = diff.astype(np.uint8)
                pre_path = cur_path
                result = np.concatenate([img_cur, diff_img], axis=0)
                videoWriter.write(result)
        videoWriter.release()


