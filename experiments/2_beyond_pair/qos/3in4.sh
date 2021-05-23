#!/usr/bin/env bash
# Author: raphael hao

combination=(
  '1 2 5'
  '1 2 6'
  '1 5 6'
  '2 5 6'
)

qos_target=(
  '100'
  '150'
  '80'
  '80'
)

# qos
echo "working dir $(pwd)"
comb_len=${#combination[@]}
tested_comb=3
testing=$((comb_len - tested_comb))
echo "$testing combination are tested"

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --model_num 3 --comb ${combination["$i"]} --policy SJF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --model_num 3 --comb ${combination["$i"]} --policy FCFS --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --model_num 3 --comb ${combination["$i"]} --policy EDF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --model_num 3 --comb ${combination["$i"]} --policy Abacus --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
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
