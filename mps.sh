#!/usr/bin/env bash
# Author: raphael hao
if [ "$1" = "start" ]; then
  echo "Starting the mps server"
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
  export CUDA_VISIBLE_DEVICES=GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22
  nvidia-cuda-mps-control -d
elif [ "$1" = "stop" ]; then
  echo "Stopping the mps server"
  echo quit | nvidia-cuda-mps-control
else
  echo "Invalid args, supported args: start or stop"
fi
