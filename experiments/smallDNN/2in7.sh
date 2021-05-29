#!/usr/bin/env bash
# Author: raphael hao

combination=(
  # '0 1'
  # '0 2'
  # '0 3'
  # '0 4'
  # '0 5'
  # '0 6'
  # '1 2'
  # '1 3'
  # '1 4'
  # '1 5'
  # '1 6'
  # '2 3'
  # '2 4'
  # '2 5'
  # '2 6'
  # '3 4'
  # '3 5'
  # '3 6'
  '4 5'
  # '4 6'
  # '5 6'
)

qos_target=(
  # '50'
  # '75'
  # '50'
  # '30'
  # '30'
  # '40'
  # '80'
  # '80'
  # '50'
  # '50'
  # '80'
  # '80'
  # '80'
  # '80'
  # '80'
  # '40'
  # '40'
  # '40'
  '20'
  # '40'
  # '40'
)
# qos
echo "working dir $(pwd)"
comb_len=${#combination[@]}
tested_comb=0
testing=$((comb_len - tested_comb))
echo "$testing combination are tested"

# for ((i = tested_comb; i < comb_len; i++)); do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy SJF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for ((i = tested_comb; i < comb_len; i++)); do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for ((i = tested_comb; i < comb_len; i++)); do
#   python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy EDF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --platform single --gpu A100 --device 0 --node 0
done

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
