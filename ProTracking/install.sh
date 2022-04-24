conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.1 -c pytorch
echo "****************** Installing yaml ******************"
pip install PyYAML
pip install yacs

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
pip install tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
# sudo apt-get install libturbojpeg
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing thop tool for FLOPs and Params computing ******************"
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing vot-toolkit python ******************"
#pip install git+https://github.com/votchallenge/vot-toolkit-python

# echo ""
# echo ""
# echo "****************** Download model files ******************"
# pip install gdown
# mkdir -p checkpoints/train/stark_st2/baseline_deit
# mkdir -p checkpoints/train/stark_ref/baseline
# gdown https://drive.google.com/u/0/uc?id=1VGQCaWnjh5sILKmH_H7Lyr4I-sel5lFx -O checkpoints/train/stark_st2/baseline_deit/  # STARKST_ep0050.pth.tar
# gdown https://drive.google.com/u/0/uc?id=1cfSAQYJMi7gn_ZCQZMhVp3pfqsSVTBEK -O checkpoints/train/stark_ref/baseline/  # STARKST_ep0500.pth.tar
# gdown https://drive.google.com/u/0/uc?id=1NuiLooc_eWpee1FdBYKMAmTvFbAnD_KU -O checkpoints/ARcm_r34/  # SEcmnet_ep0040-a.pth.tar

# mkdir pytracking/networks
# gdown https://drive.google.com/u/0/uc?id=1qDptswis2FxihLRYLVRGDvx6aUoAVVLv -O pytracking/networks/  # super_dimp.pth

# wget https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth -P $HOME/.cache/torch/hub/checkpoints/
# wget https://download.pytorch.org/models/resnet50-19c8e357.pth $HOME/.cache/torch/hub/checkpoints/

# echo ""
# echo ""
# echo "****************** Set up project path ******************"
# python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

# After running this command, you can also modify paths by editing these two files:
#    - lib/train/admin/local.py  # paths about training
#    - lib/test/evaluation/local.py  # paths about testing
pip install attributee==0.1.3
echo ""
echo ""
echo "****************** Installation complete! ******************"
