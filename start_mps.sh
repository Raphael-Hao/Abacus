#!/usr/bin/env bash
# Author: raphael hao

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d