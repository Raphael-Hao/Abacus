#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /setup.sh
# \brief: setup the environment
# Author: raphael hao

mkdir -p ~/.local
chmod +x Anaconda3.sh
bash ./Anaconda3.sh -p "$HOME"/.local/anaconda3
pip3 install torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl torchvision-0.8.2+cu110-cp38-cp38-linux_x86_64.whl
pip3 install -r requirements.txt
GPU_CNT=$(nvidia-smi -L | wc -l)
./nvidia_utils.sh --mps 1 --ngpu "$GPU_CNT"