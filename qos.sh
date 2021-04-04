#!/usr/bin/env bash
# Author: raphael hao

myArray=(
  "0 1"
  "0 2"
  "0 3"
  "0 4"
  "0 5"
  "0 6"
  "1 2"
  "1 3"
  "1 4"
  "1 5"
  "1 6"
  "2 3"
  "2 4"
  "2 5"
  "2 6"
  "3 4"
  "3 5"
  "3 6"
  "4 5"
  "4 6"
  "5 6"
)

for item in "${myArray[@]}"; do
  python main.py --task serve --combination "$item" #--policy Abacus
done
