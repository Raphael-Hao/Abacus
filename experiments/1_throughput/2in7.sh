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
  '100' # Res50+Res101
  '150' # Res50+Res152
  '100' # Res50+IncepV3
  '50' # Res50+VGG16
  '50' # Res50+VGG19
  '75' # Res50+Bert
  '160' # Res101+Res152
  '150' # Res101+IncepV3
  '90' # Res101+VGG16
  '80' # Res101+VGG19
  '130' # Res101+Bert
  '150' # Res152+IncepV3
  '150' # Res152+VGG16
  '150' # Res152+VGG19
  '150' # Res152+Bert
  '80' # IncepV3+VGG16
  '80' # IncepV3+VGG19
  '80' # IncepV3+Bert
  '40' # VGG16+VGG19
  '60' # VGG16+Bert
  '60' # VGG19+Bert
)
# qos
echo "working dir $(pwd)"
comb_len=${#combination[@]}
tested_comb=0
testing=$((comb_len - tested_comb))
echo "$testing combination are tested"

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy SJF --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy FCFS --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy EDF --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
done

for ((i = tested_comb; i < comb_len; i++)); do
  python main.py --task server --platform single --model_num 2 --comb ${combination["$i"]} --policy Abacus --load 100 --qos ${qos_target["$i"]} --queries 1000 --thld 5 --ways 2 --abandon --gpu A100 --device 0 --node 0
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
