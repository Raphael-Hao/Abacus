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
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
  '100'
)
# qos
echo "working dir $(pwd)"

# for comb_id in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy Abacus --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2 --abandon
# done

# for comb_id in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy SJF --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2
# done

# for comb_id in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy FCFS --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2
# done

# for comb_id in {0..20}; do
#   python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy EDF --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2
# done

# throughput
for comb_id in {0..20}; do
  python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy Abacus --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2 --abandon
done

for comb_id in {0..20}; do
  python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy SJF --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2 --abandon
done

for comb_id in {0..20}; do
  python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy FCFS --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2 --abandon
done

for comb_id in {0..20}; do
  python main.py --task serve --model_num 2 --comb ${combination["$comb_id"]} --policy EDF --load 50 --qos ${qos_target["$comb_id"]} --queries 1000 --thld 5 --ways 2 --abandon
done