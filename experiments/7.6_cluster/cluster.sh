#!/usr/bin/env bash
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /cluster.sh
# \brief:
# Author: raphael hao

combination='1 2 5 6'

qos_target='100'

# qos
echo "working dir $(pwd)"

python main.py --task scheduler --model_num 4 --comb "$combination" --policy Clock --load 50 --qos $qos_target --queries 1000 --thld 5 --ways 2 --abandon
