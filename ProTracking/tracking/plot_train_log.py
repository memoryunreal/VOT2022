import os
import numpy as np
import matplotlib.pyplot as plt
import math


def parse_log(log_path):
    # keys = ["Loss/total", "Loss/giou", "Loss/l1", "cls_loss", "IoU"]
    keys = ["cls_loss"]
    # parse log data
    with open(log_path, "r") as f:
        log_list = f.readlines()
    log_dict = {}
    for mode in ["train", "val"]:
        log_dict[mode] = {}
        for k in keys:
            log_dict[mode][k] = []
    for l_idx, line in enumerate(log_list):
        line_lst = line[:-1].split('  ,  ')[1:]
        for ele in line_lst:
            k, v = ele.split(': ')
            if 'train' in line:
                log_dict['train'][k].append(float(v))
            elif 'val' in line:
                log_dict['val'][k].append(float(v))
            else:
                print("illegal line:", line)
    return log_dict


def draw_plot(log_dict, key, settings, mode, label: str):
    # parse settings
    n_gpus = settings["n_gpus"]
    sample_per_epoch = settings["sample_per_epoch"][mode]
    batch_per_gpu = settings["batch_per_gpu"]
    print_interval = settings["print_interval"]
    val_interval = settings["val_interval"]
    # draw plots
    num_iter = len(log_dict[mode][key]) // n_gpus
    data_arr = np.array(log_dict[mode][key])
    log_times_per_ep = math.ceil((sample_per_epoch // (batch_per_gpu * n_gpus)) / print_interval)
    if mode == "train":
        t = np.arange(num_iter) / log_times_per_ep
    elif mode == "val":
        t = np.arange(num_iter) / log_times_per_ep * val_interval
    else:
        raise ValueError
    data_arr_new = np.zeros(num_iter)
    for i in range(num_iter):
        data_arr_new[i] = np.mean(data_arr[i * n_gpus: (i + 1) * n_gpus])
    plt.plot(t, data_arr_new, label=label)


if __name__ == "__main__":
    log_dir = '/data/sda/v-yanbi/iccv21/STARK/logs/'
    # stark-st2 R50
    log_path1 = os.path.join(log_dir, "stark_st2-baseline.log")
    settings1 = {"n_gpus": 8, "batch_per_gpu": 16, "print_interval": 50, "val_interval": 10,
                 "sample_per_epoch": {"train": 6e4, "val": 1e4}}
    log_dict1 = parse_log(log_path1)
    draw_plot(log_dict1, "cls_loss", settings1, "train", label="R50")
    # stark-st2 R101
    log_path2 = os.path.join(log_dir, "stark_st2-baseline_R101.log")
    settings2 = {"n_gpus": 8, "batch_per_gpu": 16, "print_interval": 50, "val_interval": 10,
                 "sample_per_epoch": {"train": 6e4, "val": 1e4}}
    log_dict2 = parse_log(log_path2)
    draw_plot(log_dict2, "cls_loss", settings2, "train", label='R101')
    # stark-st2 baseline_plus
    log_path2 = os.path.join(log_dir, "stark_st2-baseline_plus.log")
    settings2 = {"n_gpus": 8, "batch_per_gpu": 16, "print_interval": 50, "val_interval": 10,
                 "sample_per_epoch": {"train": 12e4, "val": 1e4}}
    log_dict2 = parse_log(log_path2)
    draw_plot(log_dict2, "cls_loss", settings2, "train", label="R50_Plus")
    plt.legend()
    # plt.show()
    plt.savefig("cls_loss_plt.png")
