import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.stark import *
# forward propagation related
from lib.train.actors import STARKSActor, STARKSTActor, STARKSPLUSActor
from lib.train.actors import STARKSLITEActor, STARKSTLITEActor
# for import modules
import importlib


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    # for loading checkpoint pth
    if "swin" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = os.path.join(settings.save_dir, "swin")

    # Create network
    if settings.script_name == "stark_s":
        net = build_starks(cfg)
    elif settings.script_name == "stark_st1" or settings.script_name == "stark_st2" \
            or settings.script_name == "stark_ref":
        net = build_starkst(cfg)
    elif settings.script_name == "stark_s_lite":
        net = build_starks_lite(cfg)
    elif "stark_st_lite" in settings.script_name:
        net = build_starkst_lite(cfg)
    elif settings.script_name == "stark_s_plus":
        net = build_starksplus(cfg)
    elif "stark_st_bn" in settings.script_name:
        net = build_starkst_bn(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")

    # Loss functions and Actors
    if settings.script_name == "stark_s" or settings.script_name == "stark_st1" \
            or settings.script_name == "stark_st_bn1" or settings.script_name == "stark_ref":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "stark_st2" or settings.script_name == "stark_st_bn2":
        objective = {'cls': BCEWithLogitsLoss()}
        loss_weight = {'cls': 1.0}
        actor = STARKSTActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name in ["stark_s_lite", "stark_st_lite1"]:
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSLITEActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name in ["stark_s_lite2"]:
        objective = {'cls': BCEWithLogitsLoss()}
        loss_weight = {'cls': 1.0}
        actor = STARKSTLITEActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == "stark_s_plus":
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = STARKSPLUSActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    else:
        raise ValueError("illegal script name")

    if cfg.TRAIN.DEEP_SUPERVISION:
        raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler)

    # train process
    if settings.script_name in ["stark_st2", "stark_st_bn2", "stark_st_lite2"]:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True, load_previous_ckpt=True)
    else:
        trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
