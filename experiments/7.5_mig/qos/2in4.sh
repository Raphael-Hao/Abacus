#!/usr/bin/env bash
# Author: raphael hao

combination=(
  '1 2'
  '1 5'
  '1 6'
  '2 5'
  '2 6'
  '5 6'
)

qos_target=(
  '130'
  '130'
  '130'
  '130'
  '130'
  '130'
)

# qos
echo "working dir $(pwd)"
comb_len=${#combination[@]}
tested_comb=0
testing=$((comb_len - tested_comb))
echo "$testing combination are tested"

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy SJF --load 25 --qos ${qos_target["$i"]} --queries 500 --thld 5 --ways 2 --abandon --mig 2 --gpu A100 --device 0 --node 0
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 25 --qos ${qos_target["$i"]} --queries 500 --thld 5 --ways 2 --abandon --mig 2 --gpu A100 --device 0 --node 0
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy EDF --load 25 --qos ${qos_target["$i"]} --queries 500 --thld 5 --ways 2 --abandon --mig 2 --gpu A100 --device 0 --node 0
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 25 --qos ${qos_target["$i"]} --queries 500 --thld 5 --ways 2 --abandon --mig 2 --gpu A100 --device 0 --node 0
done

cp -r results/mig/2in4 data/server/7.5_mig/qos/

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
