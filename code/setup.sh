# install cuda 11
wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda_11.4.2_470.57.02_linux.run
sudo sh cuda_11.4.2_470.57.02_linux.run

# install torch
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# install geometric
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+${CUDA}.html
pip3 install torch-geometric

# install dgmc
git clone https://github.com/rusty1s/deep-graph-matching-consensus.git
cd deep-graph-matching-consensus && python3 setup.py install
