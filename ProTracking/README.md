# STARK-RGBD
The STARK tracker for the VOT2021-RGBD challenge

## Install the environment
Environment Requirements:
- Operating System: Ubuntu 18.04\16.04
- CUDA: 10.2, 10.1 and 10.0 are tested
- GCC: 6.5.0 and 5.4.0 are tested
- Dependencies: All dependencies are specified in `install.sh`
- Device: *Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz* and *GeForce RTX 2080 Ti*

**Environment Setup**: Use the Anaconda
```
# create environment
conda create -n stark python=3.6
conda activate stark

# install pytorch with conda (we find cudatoolkit=10.1 is compatible with /usr/local/cuda-10.2)
conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.1 -c pytorch

# install dependancies and download model files
bash install.sh

# set up paths for the project
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

## Test and evaluate STARK on benchmarks

**VOT2021-RGBD**
- run the folloing script.
```
cd VOT21/stark_deit_ref_motion_dimp
bash exp.sh
```
Note: The <PROJECT_DIR> in [trackers.ini](VOT21/stark_deit_ref_motion_dimp/trackers.ini) requires absolute path 
  of the STARK project on your local machine. Our [exp.sh](VOT21/stark_deit_ref_motion_dimp/exp.sh) can automatically locate the
  <PROJECT_DIR>. However, you can also manually specify <PROJECT_DIR> in [trackers.ini](VOT21/stark_deit_ref_motion_dimp/trackers.ini)



## Runtime Tips (Important)
When the PrRoIPooling module *is compiled for the first time* or *the environment is changed*, the following exception may be raised.

    `STARK_RGBD/ltr/external/PreciseRoIPooling/pytorch/prroi_pool/src/prroi_pooling_gpu.c: In function`

The solution is **opening a new terminal** or **deactive then re-active the conda environment**.

# edit file /pytracking/evaluation/local.py

# edit file /lib/models/stark/deit line 9 && download resnet model && pytracking/networks/*.pth
/home/lz/.cache/torch/hub/checkpoints/deit_base_distilled_patch16_384-d0272ac0.pth

https://download.pytorch.org/models/resnet50-19c8e357.pth


