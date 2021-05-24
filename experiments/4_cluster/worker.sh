#!/usr/bin/env bash
# Author: raphael hao

combination=(
  '1 2 5 6'
)

qos_target=(
  '100'
)

# qos
echo "working dir $(pwd)"

python main.py --task server --model_num 4 --comb 1 2 5 6 --policy SJF --load 50 --qos ${qos_target} --queries 1000 --thld 5 --ways 2 --abandon

python main.py --task server --model_num 4 --comb 1 2 5 6 --policy FCFS --load 50 --qos ${qos_target} --queries 1000 --thld 5 --ways 2 --abandon

python main.py --task server --model_num 4 --comb 1 2 5 6 --policy EDF --load 50 --qos ${qos_target} --queries 1000 --thld 5 --ways 2 --abandon

python main.py --task server --model_num 4 --comb 1 2 5 6 --policy Abacus --load 50 --qos ${qos_target} --queries 1000 --thld 5 --ways 2 --abandon

# throughput
# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy SJF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy EDF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done
