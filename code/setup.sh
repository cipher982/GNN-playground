#!/bin/bash

# install cuda 11
wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda_11.4.2_470.57.02_linux.run
sudo sh cuda_11.4.2_470.57.02_linux.run

# Conda installs
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
bash ~/anaconda.sh -b -p $HOME/anaconda
echo -e '\nexport PATH=$HOME/anaconda/bin:$PATH' >> $HOME/.bashrc && source $HOME/.bashrc;
conda install -y pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install -y pytorch-geometric -c rusty1s -c conda-forge
conda install -y ipykernel pyarrow s3fs matplotlib
pip install pytorch_lightning

# install dgmc
git clone https://github.com/rusty1s/deep-graph-matching-consensus.git
cd deep-graph-matching-consensus && python setup.py install

python -m ipykernel install --user --name=torch

exec jupyter lab --no-browser --port=8889 --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''

# install torch
#pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

