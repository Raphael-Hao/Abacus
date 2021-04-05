#!/usr/bin/env bash
# Author: raphael hao

combination=(
  '0 1'
  '0 2'
  '0 3'
  '0 4'
  '0 5'
  '0 6'
  '1 2'
  '1 3'
  '1 4'
  '1 5'
  '1 6'
  '2 3'
  '2 4'
  '2 5'
  '2 6'
  '3 4'
  '3 5'
  '3 6'
  '4 5'
  '4 6'
  '5 6'
)

qos_target=(
  '100'
  '150'
  '100'
  '50'
  '50'
  '75'
  '160'
  '150'
  '90'
  '80'
  '130'
  '150'
  '150'
  '150'
  '150'
  '80'
  '80'
  '80'
  '30' # problems
  '60'
  '60'
)
# qos
echo "working dir $(pwd)"
comb_len=${#combination[@]}
tested_comb=20
testing=$((comb_len - tested_comb))
echo "$testing combination are tested"

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy SJF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy EDF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
done

# throughput
# for i in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy SJF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for i in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$i"]} --policy EDF --load 50 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon
# done
