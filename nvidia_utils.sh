#!/usr/bin/env bash
# Author: raphael hao

MiG_ENABLED=false
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  # MIG related task
  --mig)
    TASK="MIG"
    JOB="switch"
    ENABLE_MIG="$2"
    shift
    shift
    ;;
  -c | --cgi)
    TASK="MIG"
    JOB="create"
    MIG_CNT="$2"
    shift
    shift
    ;;
  -d | --dgi)
    TASK="MIG"
    JOB="destroy"
    shift
    ;;
  # MPS related task
  -m | --mps)
    TASK="MPS"
    JOB="$2"
    shift
    shift
    ;;
  -n | --ngpu)
    MULTI_GPU=true
    GPU_CNT="$2"
    shift
    shift
    ;;
  -g | --gi)
    MIG_ENABLED=true
    MIG_CNT="$2"
    shift
    shift
    ;;
  *)                   # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift
    ;;
  esac
done

MIG_2=("MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/3/0"
  "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/5/0")
MIG_4=("MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/7/0"
  "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/8/0"
  "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/9/0"
  "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/11/0")

if [ "$TASK" = "MPS" ]; then
  echo "MPS management!!!"
  if [ "$JOB" = 1 ]; then
    echo "Starting the mps server"
    if [ "$MIG_ENABLED" = true ]; then
      if [ "$MIG_CNT" = 1 ]; then
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-0
        export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-0
        export CUDA_VISIBLE_DEVICES=MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/2/0
        nvidia-cuda-mps-control -d
        echo "MPS server at INSTANCE ${CUDA_VISIBLE_DEVICES} started"
      elif [ "$MIG_CNT" = 2 ]; then
        for ((i = 0; i < "$MIG_CNT"; i++)); do
          echo "${MIG_2[${i}]}"
          export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-"${i}"
          export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-"${i}"
          export CUDA_VISIBLE_DEVICES="${MIG_2[${i}]}"
          nvidia-cuda-mps-control -d
          echo "MPS server at INSTANCE ${CUDA_VISIBLE_DEVICES} started"
        done
      elif [ "$MIG_CNT" = 4 ]; then
        for ((i = 0; i < "$MIG_CNT"; i++)); do
          echo "${MIG_4[${i}]}"
          export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-"${i}"
          export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-"${i}"
          export CUDA_VISIBLE_DEVICES="${MIG_4[${i}]}"
          nvidia-cuda-mps-control -d
          echo "MPS server at INSTANCE ${CUDA_VISIBLE_DEVICES} started"
        done
      fi
    elif [ "$MULTI_GPU" = true ]; then
      for ((i = 0; i < "$GPU_CNT"; i++)); do
        # echo "${MIG_2[${i}]}"
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-"${i}"
        export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-"${i}"
        export CUDA_VISIBLE_DEVICES="${i}"
        nvidia-cuda-mps-control -d
        echo "MPS server at GPU ${CUDA_VISIBLE_DEVICES} started"
      done
    else
      export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
      export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
      export CUDA_VISIBLE_DEVICES=GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22
      nvidia-cuda-mps-control -d
      echo "MPS server at GPU ${CUDA_VISIBLE_DEVICES} started"
    fi
  elif [ "$JOB" = 0 ]; then
    echo "Stopping the mps server"
    if [ "$MIG_ENABLED" = true ]; then
      if [ "$MIG_CNT" = 1 ]; then
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-0
        export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-0
        export CUDA_VISIBLE_DEVICES=MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/2/0
        echo quit | nvidia-cuda-mps-control
        echo "MPS server at INSTANCE ${CUDA_VISIBLE_DEVICES} stopped"
      elif [ "$MIG_CNT" = 2 ]; then
        for ((i = 0; i < "$MIG_CNT"; i++)); do
          echo "${MIG_2[${i}]}"
          export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-"${i}"
          export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-"${i}"
          export CUDA_VISIBLE_DEVICES="${MIG_2[${i}]}"
          echo quit | nvidia-cuda-mps-control
          echo "MPS server at INSTANCE ${CUDA_VISIBLE_DEVICES} stopped"
        done
      elif [ "$MIG_CNT" = 4 ]; then
        for ((i = 0; i < "$MIG_CNT"; i++)); do
          echo "${MIG_4[${i}]}"
          export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-"${i}"
          export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-"${i}"
          export CUDA_VISIBLE_DEVICES="${MIG_4[${i}]}"
          echo quit | nvidia-cuda-mps-control
          echo "MPS server at INSTANCE ${CUDA_VISIBLE_DEVICES} stopped"
        done
      fi
    elif [ "$MULTI_GPU" = true ]; then
      for ((i = 0; i < "$GPU_CNT"; i++)); do
        # echo "${MIG_2[${i}]}"
        export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-"${i}"
        export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-"${i}"
        export CUDA_VISIBLE_DEVICES="${i}"
        echo quit | nvidia-cuda-mps-control
        echo "MPS server at GPU ${CUDA_VISIBLE_DEVICES} started"
      done
    else
      export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
      export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
      export CUDA_VISIBLE_DEVICES=GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22
      echo quit | nvidia-cuda-mps-control
      echo "MPS server at INSTANCE ${CUDA_VISIBLE_DEVICES} stopped"
    fi
  fi
elif [ "$TASK" = "MIG" ]; then
  echo "MIG mannagement!!!"
  if [ "$JOB" = "create" ]; then
    echo "Creating the GPU instances"
    if [ "$MIG_CNT" = 1 ]; then
      sudo nvidia-smi mig -cgi 5 -C
    elif [ "$MIG_CNT" = 2 ]; then
      sudo nvidia-smi mig -cgi 14,14 -C
    elif [ "$MIG_CNT" = 4 ]; then
      sudo nvidia-smi mig -cgi 19,19,19,19 -C
    fi
  elif [ "$JOB" = "destroy" ]; then
    echo "Destroying the GPU instances"
    sudo nvidia-smi mig -dci -i 0
    sudo nvidia-smi mig -dgi -i 0
  elif [ "$JOB" = "switch" ]; then
    sudo nvidia-smi -i 0 -mig "$ENABLE_MIG"
  fi
else
  echo "Invalid args, supported args: start or stop"
fi
