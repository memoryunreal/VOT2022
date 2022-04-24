from lib.test.vot20.stark_vot20rgbd import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
run_vot_exp('stark_st', 'baseline_deit', vis=False)
