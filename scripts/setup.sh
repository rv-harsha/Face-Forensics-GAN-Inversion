#!/bin/bash

conda install -c conda-forge gdown opencv
gdown https://drive.google.com/uc?id=1pfn_IPUogQz9y0zjraNBf698rW8GVl3N
mkdir outputs tmp

# conda install -c conda-forge/label/gcc7 gcc_linux-64

## py37 env
rm -rf ~/.cache/torch_extensions/

# conda install -c 3dhubs gcc-5  # This does not work anymore
# conda install -c psi4 gcc-5
conda install -c remenska libgcc-5
conda install -c msarahan libgcc

# conda install libgcc=5.2.0 -c conda-forge

# Donot use this
conda install -c conda-forge/label/gcc7 gcc_linux-64
conda install libgcc

$$$$$ conda install -c anaconda gcc_linux-64=5.4.0 $$$$
$$$$$ conda install -c anaconda gxx_linux-64=5.4.0 $$$$

$$$$$ conda install -c anaconda gcc_linux-64=7.3.0 $$$$
$$$$$ conda install -c anaconda gxx_linux-64=7.3.0 $$$$

conda create --name py37 anaconda python=3.7.2 

# Create symbolic links in the environment
cd /nas/home/raidurga/anaconda3/envs/<py38>/bin
ln -s x86_64-conda_cos6-linux-gnu-gcc gcc
ln -s x86_64-conda_cos6-linux-gnu-cpp cpp
rm symlink1 symlink2

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install imageio-ffmpeg -c conda-forge --no-deps
pip install pyspng

# Add env variable
conda env config vars set LD_LIBRARY_PATH=~/anaconda3/lib:$LD_LIBRARY_PATH
conda env config vars list
# To unset the environment variable, run 
conda env config vars unset my_var -n test-env
------------------


# TO Avoid this error
cd ~/anaconda3/lib
mv -vf libstdc++.so.6 libstdc++.so.6.old
ln -s ~/anaconda3/envs/py38/lib/libstdc++.so.6 ./libstdc++.so.6
-----------------

# For deepface env
pip install deepface
conda install -c conda-forge yaml
-----------


srun --ntasks=1 --partition ALL --account other --qos limited --cpus-per-task=4 --mem=16G --gres=gpu:1 --time=10:30:00 --pty bash

conda remove --name myenv --all
conda info --envs


================================
# Create env
conda create --name sg2 anaconda python=3.7 gcc_linux-64=5.4.0 gxx_linux-64=5.4.0 -c anaconda pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch imageio-ffmpeg -c conda-forge 

pip install pyspng
conda install libgcc=5.2.0 -c conda-forge

# Adds the channel "new_channel" to the top of the channel list, making it the highest priority:
conda config --add channels anaconda
conda config --add channels conda-forge
conda config --add channels pytorch

# Adds the new channel to the bottom of the channel list, making it the lowest priority:
conda config --append channels new_channel

# Set channel priority
conda config --set channel_priority false
rm -rf ~/.condarc ~/.conda ~/.continuum ~/miniconda

# To pin conda packages 
conda list "^(tensorflow|keras)$" | tail -n+4 | awk '{ print $1 " ==" $2 }' > $CONDA_PREFIX/conda-meta/pinned

==============


sbatch --account other --qos limited --partition ALL --requeue --mem=16G --time=1-23:59:59 --gres=gpu:1 --job-name=ai2ai_rnw_run_001!rnw_img_0010_1e1 --output=/nas/vista-hdd01/users/raidurga/outputs/ai2ai_rnw_run_001/logs/rnw_img_0010_1e1.log --cpus-per-task=4 --export=jid=rnw_img_0010_1e1,job_path=/nas/vista-hdd01/users/raidurga/outputs/ai2ai_rnw_run_001/expts/img_0010/1e1/,target=/nas/vista-hdd01/users/raidurga/workspace/styleGAN2/dataset/Temp/img_0010.png,rnw=1e1 run_projector.sh 2>&1 | tee -a /nas/vista-hdd01/users/raidurga/outputs/ai2ai_rnw_run_001/sbatch/img_0010.sbatch
Submitted batch job 52260


~/anaconda3/envs/sg2/bin/python projector.py --target /nas/vista-hdd01/users/raidurga/workspace/styleGAN2/dataset/Temp/img_0010.png --regularize_noise_weight 1e1 --outdir /nas/vista-hdd01/users/raidurga/outputs/ai2ai_rnw_run_001/



########################
conda create --name sg2 anaconda python=3.7 -c conda-forge
conda activate sg2
conda deactivate
conda update conda
conda activate sg2
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 imageio-ffmpeg -c pytorch -c conda-forge
pip install pyspng
conda update --all

ln -s x86_64-conda_cos6-linux-gnu-cpp cpp
ln -s x86_64-conda_cos6-linux-gnu-gcc gcc
ln -s x86_64-conda_cos6-linux-gnu-g++ g++
ln -s x86_64-conda_cos6-linux-gnu-c++ c++

x86_64-conda_cos6-linux-gnu-nm
x86_64-conda_cos6-linux-gnu-ld.gold
x86_64-conda_cos6-linux-gnu-ld.bfd
x86_64-conda_cos6-linux-gnu-elfedit
x86_64-conda_cos6-linux-gnu-dwp
x86_64-conda_cos6-linux-gnu-c++filt
x86_64-conda_cos6-linux-gnu-as
x86_64-conda_cos6-linux-gnu-ar
x86_64-conda_cos6-linux-gnu-strip
x86_64-conda_cos6-linux-gnu-strings
x86_64-conda_cos6-linux-gnu-size
x86_64-conda_cos6-linux-gnu-readelf
x86_64-conda_cos6-linux-gnu-ranlib
x86_64-conda_cos6-linux-gnu-objdump
x86_64-conda_cos6-linux-gnu-objcopy
x86_64-conda_cos6-linux-gnu-gprof
x86_64-conda_cos6-linux-gnu-addr2line
c99
c89
x86_64-conda_cos6-linux-gnu-ct-ng.config
x86_64-conda_cos6-linux-gnu-cpp
x86_64-conda_cos6-linux-gnu-gcc-nm
x86_64-conda_cos6-linux-gnu-gcc-ar
x86_64-conda_cos6-linux-gnu-gcc
x86_64-conda_cos6-linux-gnu-gcc-ranlib
x86_64-conda_cos6-linux-gnu-g++
x86_64-conda_cos6-linux-gnu-c++


nvidia-smi --query-gpu=gpu_name,gpu_bus_id,vbios_version --format=csv


nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total --format=csv

#==================
# ENVIRONMENT SETUP
#==================

conda create --name <env_name> anaconda python=3.7 gcc_linux-64=5.4.0 gxx_linux-64=5.4.0 -c anaconda -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate <env_name>
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 imageio-ffmpeg -c pytorch -y
pip install pyspng
cd $CONDA_PREFIX/bin
ln -s x86_64-conda_cos6-linux-gnu-cpp cpp
ln -s x86_64-conda_cos6-linux-gnu-gcc gcc
ln -s x86_64-conda_cos6-linux-gnu-g++ g++
ln -s x86_64-conda_cos6-linux-gnu-c++ c++
cd
conda deactivate <env_name>

#===============================
# GIT SETUP FOR BASE ENV
#===============================
conda activate base
conda install git -c conda-forge -y
git config --global http.sslVerify false
git clone https://gitlab.vista.isi.edu/raidurga/aifi.git

#===============================
# ENVIRONMENT SETUP FOR DEEPFACE
#===============================
conda create --name deepface anaconda python=3.8.5 gcc_linux-64=9.3.0 gxx_linux-64=9.3.0 -c anaconda

conda install --file DeepFace/requirements.txt -c anaconda --yes

Provided you have the CMake, Boost, Boost.Python, and X11/XQuartz installed on your system


conda create --name cmake python=3.8.5 gcc_linux-64 gxx_linux-64 cmake boost boost-cpp xorg-libx11 numpy pandas gdown tqdm Pillow tensorflow keras Flask mtcnn dlib lightgbm -c conda-forge

#===============================

#===============================

conda create --name df_test python=3.7.10 openpyxl numpy pandas gdown tqdm Pillow Flask mtcnn dlib lightgbm tensorflow=2.2.0 tensorflow-estimator=2.2 keras=2.3.1 scipy=1.6.3 tensorboard=2.2.2 -c conda-forge
conda activate deepface;
pip install opencv-contrib-python retina-face deepface


numpy>=1.14.0
pandas>=0.23.4
gdown>=3.10.1
tqdm>=4.30.0
Pillow>=5.2.0
opencv-python>=4.2.0.34
opencv-contrib-python>=4.3.0.36
tensorflow>=1.9.0
keras>=2.2.0
Flask>=1.1.2
mtcnn>=0.1.0
lightgbm>=2.3.1
retina-face>=0.0.1
dlib>=19.20.0

Installing collected packages: tensorflow-estimator, tensorboard, scipy, opencv-python, retina-face, deepface

conda create deepface python=3.7.10 dlib
pip install deepface

srun --ntasks=1 --partition ALL --account other --qos limited --cpus-per-task=2 --mem-per-cpu=4GB --mem-per-gpu=8G --gres=gpu:1 --gpus=1 --spread-job  --gpus-per-node=1 --time=00:30:00 --spread-job --pty bash

# echo ''
# echo Allocated GPU Stats
# echo ''
# echo "$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,memory.total --format=csv)"

(deepface) [raidurga@vista14 workspace]$ sinfo -o "%20N  %10c  %10m  %25f  %10G "
NODELIST              CPUS        MEMORY      AVAIL_FEATURES             GRES       
vista[15-16,21-22]    32+         236000+     (null)                     gpu:rtx800 
vista[10,13]          32+         370000+     (null)                     (null)     
vista[18-19]          24+         90000+      (null)                     gpu:titanx 
vista[08,14]          12          240000+     (null)                     gpu:rtx208 
vista[01-05,09,12]    40          106000+     (null)                     gpu:1080:8 
vista[06-07,11]       32+         106000+     (null)                     gpu:titanx 
vista17               32          110000      (null)                     gpu:titanx 
vista20               32          120000      (null)                     gpu:titanx 

squeue -u raidurga | grep PD | awk '{print $1}' | wc -l
queue -u raidurga | grep PD | awk '{print $1}' | xargs -n 1 scancel


# To run for reprocessing
nohup submit_batch_projector.sh /nas/vista-hdd01/users/raidurga/outputs/ai2ai_run_003/expts/reprocess.csv &