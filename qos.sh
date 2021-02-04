#!/usr/bin/env bash
# Author: raphael hao

myArray=(
  "1 3"
  "0 2"
  "2 5"
  "0 3"
  "1 2"
  "1 5"
  "0 4"
  "4 5"
  "1 4"
  "0 5"
  "2 3"
  "3 5"
  "0 1"
  "3 4"
  "2 4"
  "5 6"
  "4 6"
  "3 6"
  "2 6"
  "1 6"
  "0 6"
)

for item in "${myArray[@]}"; do
  python main.py --task serve --combination $item #--policy Abacus
done
