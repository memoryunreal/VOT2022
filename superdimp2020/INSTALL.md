# Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking. The instrustions have been tested on an Ubuntu 18.04 system. We recommend using the [install script](install.sh) if you have not already tried that.  

### Requirements  
* Conda installation with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.

## Step-by-step instructions  
#### Create and activate a conda environment
```bash
conda create --name pytracking python=3.7
conda activate pytracking
```

#### Install PyTorch  
Install PyTorch with cuda10.  
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

**Note:**  
- It is possible to use any PyTorch supported version of CUDA (not necessarily v10).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install matplotlib, pandas, opencv,  visdom, gdown  
```bash
conda install matplotlib pandas
pip install opencv-python visdom gdown
```


#### Install ninja-build for Precise ROI pooling  
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  


#### Setup the environment  
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

```

#### Download the pre-trained networks  
You can download the pre-trained networks from the [google drive folder](https://drive.google.com/open?id=1FzGPxJy2k1rrY8bYGR6z20TySJVCACuI). 
The networks shoud be saved in the directory pytracking/networks.
You can also download the networks using the [gdown](https://github.com/wkentaro/gdown) python package.

```bash
# Download the default network for DiMP-50
gdown https://drive.google.com/uc\?id\=1cUE-NfHXvAI_dBBVlBj__CQaKQFvIKCU -O pytracking/networks/dimp50.pth.tar

# Download the network for SuperDiMP
gdown https://drive.google.com/uc\?id\=1QniYCYszrP0XVM6zKKNRueCOmDoKCNct -O pytracking/networks/super_dimp.pth.tar

# Download the network for LWTL
gdown https://drive.google.com/uc\?id\=1aXWbKnwehXs2JOCY68fQdzSImfTcATRB -O pytracking/networks/lwtl.pth.tar
```
