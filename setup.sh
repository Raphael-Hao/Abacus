#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /setup.sh
# \brief: setup the environment
# Author: raphael hao

GPU_CNT=$(nvidia-smi -L | wc -l)
./nvidia_utils.sh --mps 1 --ngpu "$GPU_CNT"