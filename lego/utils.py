#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import sys
import time
import itertools

def timestamp(name, stage):
    print("TIMESTAMP, %s, %s, %f" % (name, stage, time.time()), file=sys.stderr)

def gen_model_combinations(models, combination_len, done_combinations=None):
    id_combinations = [i for i in range(len(models)) for j in range(combination_len)]
    id_combinations = set(itertools.combinations(id_combinations, combination_len))
    for profiled in done_combinations:
        id_combinations.remove(profiled)
    model_combinations = []
    for id_comb in id_combinations:
        model_comb = []
        for id in id_comb:
            model_comb.append(models[id])
        model_combinations.append(model_comb)
    return model_combinations