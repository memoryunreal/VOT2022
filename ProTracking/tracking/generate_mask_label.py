import os
import sys

prj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ar_vot21_dir = os.path.join(prj_dir, "external", "AR_VOT21")
sys.path.append(ar_vot21_dir)
from external.AR_VOT21.pytracking.ARcm_seg import ARcm_seg
import cv2
import numpy as np
from lib.test.evaluation import get_dataset
import multiprocessing
import argparse
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def predict_mask_per_seq(dataset_name, seq, num_gpus):
    """predict masks for the whole sequences"""
    worker_name = multiprocessing.current_process().name
    worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
    gpu_id = worker_id % num_gpus
    torch.cuda.set_device(gpu_id)
    save_root_dir = "/data/sdc/tracking_data_mask/%s" % dataset_name
    if "lasot" in dataset_name:
        class_name = seq.name.split('-')[0]
        save_dir = os.path.join(save_root_dir, class_name, seq.name)
    elif "got10k" in dataset_name:
        save_dir = os.path.join(save_root_dir, seq.name)
    else:
        raise ValueError("Unsupported dataset_name")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # build Alpha-Refine model
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    refine_root = os.path.join(project_path, 'external/AR_VOT21/checkpoints/ltr/ARcm_seg/')
    refine_model_name = "baseline"
    refine_path = os.path.join(refine_root, refine_model_name)
    alpha = ARcm_seg(refine_path, input_sz=384)
    # predict masks
    gts = seq.ground_truth_rect
    valid_list = []
    for idx, frame_path in enumerate(seq.frames):
        save_path = os.path.join(save_dir, "%08d.jpg" % (idx + 1))
        if not os.path.exists(save_path):
            try:
                gt = gts[idx]
                image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
                mask = alpha.predict_ego_mask(image, gt)
                final_mask = ((mask > 0.5) * 255).astype(np.uint8)
                cv2.imwrite(save_path, final_mask)
                valid_list.append(1)
            except:
                valid_list.append(0)
        else:
            valid_list.append(1)
    txt_path = os.path.join(save_dir, "valid.txt")
    np.savetxt(txt_path, np.array(valid_list).astype(np.int), fmt="%d")
    print("%s is done." % seq.name)


if __name__ == "__main__":
    args = parse_args()
    dataset_names = ["got10k_train", "lasot_train"]  # lasot_train, got10k_train
    for n in dataset_names:
        dataset = get_dataset(n)
        param_list = [(n, seq, args.num_gpus) for seq in dataset]
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(predict_mask_per_seq, param_list)
