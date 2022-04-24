from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
filepath = os.path.abspath(__file__)
prj_dir = os.path.abspath(os.path.join(os.path.dirname(filepath), "../../../.."))
lib_dir = os.path.abspath(os.path.join(prj_dir, "lib"))
sys.path.append(prj_dir)
sys.path.append(lib_dir)

import torch
import numpy as np
import cv2
import vot

from lib.test.evaluation import Tracker


class MotionEstimator(object):
    """ Statistic of the track """
    def __init__(self):
        self.pos = None
        self.vel = None
        self.conf = 0
        self.vel_his = []  # velocity history
        self.vel_his_conf = []  # velocity history

    def initialize(self, box):
        self.pos = box[:2] + box[2:]/2
        self.wh = box[2:]
        self.vel_his = []  # velocity history
        self.vel_his_conf = []  # velocity history

    def update_vel_his(self, v, conf):
        self.vel_his.append(v)
        self.vel_his_conf.append(conf)
        if len(self.vel_his) > 20:
            self.vel_his.pop(0)
            self.vel_his_conf.pop(0)
        vel_his = np.stack(self.vel_his)
        vel_conf = np.stack(self.vel_his_conf)
        vel_conf_norm = vel_conf/vel_conf.sum()

        vel_mean = vel_conf_norm.dot(vel_his)
        vel_var = vel_his.var(0)
        return vel_mean, vel_var

    def cosine_pdf(self, x, x_, size):
        """
        Args:
            x: Tracker预测位置
            x_: 运动模型预测位置
            size: 目标尺寸
        """
        PI = 3.1415
        phase = min(np.linalg.norm(x-x_)/size, 1)*PI/2
        return np.cos(phase)

    def update(self, box, conf, th=0.3):
        pos = box[:2] + box[2:]/2
        if conf > th:
            self.wh = box[2:]

        _v = pos - self.pos  # 瞬时速度
        if self.vel is None:
            self.vel = _v  # 初始化estimator的速度
        vel_mean, vel_var = self.update_vel_his(_v, conf)  # 目标运动速度的统计量

        self.vel = vel_mean
        self.pos = pos
        self.conf = conf
        return box, self.conf

    def rescore(self, box, conf):
        pos = box[:2] + box[2:]/2

        _v = pos - self.pos  # 瞬时速度
        if self.vel is None:
            self.vel = _v  # 初始化estimator的速度

        motion_conf = self.cosine_pdf(pos, self.pos + self.vel, size=np.sqrt(self.wh.prod()))  # 当前目标运动速度的可信度
        print(motion_conf)
        conf = conf * (conf ** 2 + (1-conf**2) * motion_conf)
        return conf


class STARK_REF_LT(object):
    """STARK base tracker + STARK refinement"""
    def __init__(self, base_tracker='stark_st', base_param='baseline',
                 ref_tracker='stark_ref', ref_param='baseline', use_new_box=True):
        """use_new_box: whether to use the refined box as the new state"""
        self.use_new_box = use_new_box
        # create base tracker
        tracker_info = Tracker(base_tracker, base_param, "lasot", None)
        base_params = tracker_info.get_parameters()
        base_params.cfg.TEST.UPDATE_INTERVALS['LASOT'] = [100000]
        base_params.visualization, base_params.debug = False, False
        self.tracker = tracker_info.create_tracker(base_params)
        # create refinement module
        ref_info = Tracker(ref_tracker, ref_param, "lasot", None)
        ref_params = ref_info.get_parameters()
        ref_params.visualization, ref_params.debug = False, False
        self.ref = ref_info.create_tracker(ref_params)
        self.me = MotionEstimator()
        self.cfg = {}
        self.aux_conf_max = 0.8
        self.dimp_th = 0.2  # threshold for switch to dimp
        self.aux_conf_min = -0.0001
        self.cfg['aux_conf_max'] = self.aux_conf_max
        self.cfg['aux_conf_min'] = self.aux_conf_min

    def initialize(self, img_rgb, box, aux_tracker):
        """box: list"""
        # init on the 1st frame
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': box}
        # init base tracker
        _ = self.tracker.initialize(img_rgb, init_info)
        # init refinement module
        self.ref.initialize(img_rgb, init_info)
        self.aux_tracker = aux_tracker
        self.me.initialize(np.array(box))

    def track(self, img_rgb):
        # track with the base tracker
        outputs, update_flag = self.tracker.track(img_rgb, return_update_flag=True)
        pred_bbox = outputs['target_bbox']
        conf_score = outputs["conf_score"]
        _conf_score = self.me.rescore(np.array(pred_bbox), conf_score)  # depress teleport  # TODO: check the rescore preocess

        aux_outputs = self.aux_tracker.track(img_rgb)
        aux_pred_bbox = aux_outputs['target_bbox']
        if _conf_score < self.dimp_th:  # 如果Stark长期失效，那么super_dimp也不那么可信
            pred_bbox = aux_pred_bbox
            self.aux_conf_max = max(self.aux_conf_max * 0.99, self.aux_conf_min)
            conf_score = min(aux_outputs['conf'], self.aux_conf_max)
        else:
            self.aux_conf_max = self.cfg['aux_conf_max']

        # refine with the refinement module
        outputs_ref = self.ref.track(img_rgb, pred_bbox, update_flag)
        pred_bbox_ref = outputs_ref["target_bbox"]
        pred_bbox_ref, conf = self.me.update(np.array(pred_bbox_ref), conf_score)

        if self.use_new_box:
            self.tracker.state = pred_bbox_ref
        self.aux_tracker.update_state(pred_bbox_ref)

        return pred_bbox_ref, conf_score, _conf_score


class DiMP():
    def __init__(self, img, init_box):
        # create tracker
        from pytracking.evaluation import Tracker

        tracker_info = Tracker('dimp', 'super_dimp', None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
        self.tracker = tracker_info.tracker_class(params)

        self.im_H, self.im_W, _ = img.shape
        cx, cy, w, h = self.get_axis_aligned_bbox(np.array(init_box))
        gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        '''Initialize'''
        gt_bbox_np = np.array(gt_bbox_)
        gt_bbox_torch = torch.from_numpy(gt_bbox_np.astype(np.float32))
        init_info = {}
        init_info['init_bbox'] = gt_bbox_torch
        self.tracker.initialize(img, init_info)

    def track(self, img):
        return self.tracker.track(img)

    @staticmethod
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
            cx = x + w / 2
            cy = y + h / 2
        return cx, cy, w, h

    @staticmethod
    def bbox_clip(x1, y1, x2, y2, boundary, min_sz=10):
        x1_new = max(0, min(x1, boundary[1] - min_sz))
        y1_new = max(0, min(y1, boundary[0] - min_sz))
        x2_new = max(min_sz, min(x2, boundary[1]))
        y2_new = max(min_sz, min(y2, boundary[0]))
        return x1_new, y1_new, x2_new, y2_new

    def update_state(self, pred_bbox):

        x1, y1, w, h = pred_bbox.tolist()
        '''add boundary and min size limit'''
        x1, y1, x2, y2 = self.bbox_clip(x1, y1, x1 + w, y1 + h, (self.im_H, self.im_W))
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / self.tracker.base_target_sz.prod())
        ##### update
        self.tracker.pos = new_pos.clone()
        self.tracker.target_sz = new_target_sz
        self.tracker.target_scale = new_scale


def run_vot_exp(base_tracker, base_param, ref_tracker, ref_param, use_new_box, save_dir, vis=False):
    torch.set_num_threads(1)
    dir_name = 'longterm' + base_tracker + '_' + base_param + '--' + ref_tracker + '_' + ref_param + 'rescore3'
    save_root = os.path.join(save_dir, dir_name)
    tracker = STARK_REF_LT(base_tracker=base_tracker, base_param=base_param,
                           ref_tracker=ref_tracker, ref_param=ref_param, use_new_box=use_new_box)
    handle = vot.VOT("rectangle", "rgbd")
    selection = handle.region()
    imagefile = handle.frame()
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
    dimp = DiMP(image, init_box)
    tracker.initialize(image, init_box, dimp)

    cnt = 0
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right
        b1, conf, _conf = tracker.track(image)
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
            cv2.putText(image_b, "{:.4f}".format(conf), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            cv2.putText(image_b, "{:.4f}".format(_conf), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            image_b_name = image_name
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)
        cnt += 1


if __name__ == '__main__':
    save_dir = os.path.abspath(os.path.join(prj_dir, '../analysis/vis'))
    run_vot_exp(base_tracker='stark_st', base_param='baseline_deit', ref_tracker="stark_ref", ref_param="baseline",
                use_new_box=False, save_dir=save_dir, vis=False)
