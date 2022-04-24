#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y --name $conda_env_name

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown

echo ""
echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"
echo "************************* Need sudo privilege ******************"
sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** Downloading networks ******************"
mkdir pytracking/networks

echo ""
echo ""
echo "****************** DiMP50 Network ******************"
gdown https://drive.google.com/uc\?id\=1cUE-NfHXvAI_dBBVlBj__CQaKQFvIKCU -O pytracking/networks/dimp50.pth.tar

echo ""
echo ""
echo "****************** SuperDiMP Network ******************"
gdown https://drive.google.com/uc\?id\=1QniYCYszrP0XVM6zKKNRueCOmDoKCNct -O pytracking/networks/super_dimp.pth.tar

echo ""
echo ""
echo "****************** LWTL Network ******************"
gdown https://drive.google.com/uc\?id\=1aXWbKnwehXs2JOCY68fQdzSImfTcATRB -O pytracking/networks/lwtl.pth.tar

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"


echo ""
echo ""
echo "****************** Installation complete! ******************"
