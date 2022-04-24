from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
from copy import deepcopy
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.stark import build_starkst
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box
"""2021.4.19 STARK-ST with the meta-updater"""
import tensorflow as tf
from external.LTMU.ATOM_MU.meta_updater.tcNet import tclstm
from external.LTMU.ATOM_MU.meta_updater.tcopt import tcopts
from external.LTMU.utils.metric_net.metric_model import ft_net
from torch.autograd import Variable
from external.LTMU.utils.metric_net.me_sample_generator import *
from lib.test.evaluation.environment import env_settings
from external.LTMU.ATOM_MU.tracking_utils import process_regions


class STARK_ST_MU(BaseTracker):
    def __init__(self, params, dataset_name):
        super(STARK_ST_MU, self).__init__(params)
        network = build_starkst(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # template update
        self.z_dict1 = {}
        self.z_dict_list = []
        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        print("Update interval is: ", self.update_intervals)
        self.num_extra_template = len(self.update_intervals)

        """Meta-Updater related"""
        self.prj_dir = env_settings().prj_dir
        # meta-updater
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.Session(config=tfconfig)
        self.tc_init("stark_lstm")
        # metricnet
        self.build_metricnet()
        # record results for running the meta-updater
        self.dis_record = None
        self.state_record = None

    def initialize(self, image, info: dict):
        # initialize z_dict_list
        self.z_dict_list = []
        # get the 1st template
        z_patch_arr1, _, z_amask_arr1 = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        template1 = self.preprocessor.process(z_patch_arr1, z_amask_arr1)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1)
        # get the complete z_dict_list
        self.z_dict_list.append(self.z_dict1)
        for i in range(self.num_extra_template):
            self.z_dict_list.append(deepcopy(self.z_dict1))

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0

        """Meta-Updater related"""
        # record results for running the meta-updater
        self.dis_record = []
        self.state_record = []
        # initialize the metricnet
        self.metric_init(image, np.array(info['init_bbox']))

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # get the t-th search region
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = self.network.forward_backbone(search)
            # merge the template and the search
            feat_dict_list = self.z_dict_list + [x_dict]
            seq_dict = merge_template_search(feat_dict_list)
            # run the transformer
            out_dict, _, _ = self.network.forward_transformer(seq_dict=seq_dict, run_box_head=True, run_cls_head=True)
        # get the final result
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # get confidence score (whether the search region is reliable)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
        """traditional update strategy"""
        if self.frame_id < tcopts['time_steps']:
            for idx, update_i in enumerate(self.update_intervals):
                if self.frame_id % update_i == 0 and conf_score > 0.5:
                    z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)
                    template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                    with torch.no_grad():
                        z_dict_t = self.network.forward_backbone(template_t)
                    self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame
        """Meta-Updater related"""
        self.record_state(image, np.array(self.state))
        if len(self.state_record) >= tcopts['time_steps']:
            dis = np.array(self.dis_record[-tcopts["time_steps"]:]).reshape((tcopts["time_steps"], 1))
            state_tc = np.array(self.state_record[-tcopts["time_steps"]:])
            X_input = np.concatenate((state_tc, dis), axis=1)
            logits = self.sess.run(self.logits, feed_dict={self.X_input: np.expand_dims(X_input, axis=0)})
            update = logits[0][0] < logits[0][1]
            if update:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                template_t = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    z_dict_t = self.network.forward_backbone(template_t)
                for idx, update_i in enumerate(self.update_intervals):
                    self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            # save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            # cv2.imwrite(save_path, image_BGR)
            cv2.namedWindow("frames", 0)
            cv2.imshow("frames", image_BGR)
            cv2.waitKey(1)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "conf_score": conf_score}
        else:
            return {"target_bbox": self.state,
                    "conf_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def tc_init(self, model_dir):
        self.tc_model = tclstm()
        self.X_input = tf.placeholder("float", [None, tcopts['time_steps'], tcopts['lstm_num_input']])
        self.Inputs = self.X_input
        self.logits, _ = self.tc_model.net(self.Inputs)

        variables_to_restore = [var for var in tf.global_variables() if
                                (var.name.startswith('tclstm') or var.name.startswith('mapnet'))]
        saver = tf.train.Saver(var_list=variables_to_restore)

        checkpoint = tf.train.latest_checkpoint(
                os.path.join(self.prj_dir, 'external/LTMU/ATOM_MU/meta_updater', model_dir))
        print("Loading Meta-Updater from %s" % checkpoint)
        saver.restore(self.sess, checkpoint)

    def build_metricnet(self):
        self.metric_model = ft_net(class_num=1120)
        path = os.path.join(self.prj_dir, 'external/LTMU/utils/metric_net/metric_model/metric_model.pt')
        self.metric_model.eval()
        self.metric_model = self.metric_model.cuda()
        self.metric_model.load_state_dict(torch.load(path, map_location='cpu'))

    def metric_init(self, im, init_box):
        tmp = np.random.rand(1, 3, 107, 107)
        tmp = (Variable(torch.Tensor(tmp))).type(torch.FloatTensor).cuda()
        # get target feature
        self.metric_model(tmp)
        init_box = init_box.reshape((1, 4))
        anchor_region = me_extract_regions(im, init_box)
        anchor_region = process_regions(anchor_region)
        anchor_region = torch.Tensor(anchor_region)
        anchor_region = (Variable(anchor_region)).type(torch.FloatTensor).cuda()
        self.anchor_feature, _ = self.metric_model(anchor_region)

    def metric_eval(self, im, boxes, anchor_feature):
        box_regions = me_extract_regions(np.array(im), boxes)
        box_regions = process_regions(box_regions)
        box_regions = torch.Tensor(box_regions)
        box_regions = (Variable(box_regions)).type(torch.FloatTensor).cuda()
        box_features, class_result = self.metric_model(box_regions)
        ap_dist = torch.norm(anchor_feature - box_features, 2, dim=1).view(-1)
        return ap_dist

    def record_state(self, image, state):
        """record the normalized coordinates and the metric score in each frame"""
        local_state = np.array(state).reshape((1, 4))
        ap_dis = self.metric_eval(image, local_state, self.anchor_feature)
        self.dis_record.append(ap_dis.data.cpu().numpy()[0])
        h = image.shape[0]
        w = image.shape[1]
        self.state_record.append([local_state[0][0] / w, local_state[0][1] / h,
                                  (local_state[0][0] + local_state[0][2]) / w,
                                  (local_state[0][1] + local_state[0][3]) / h])


def get_tracker_class():
    return STARK_ST_MU
