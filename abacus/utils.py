#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import sys
import time
import random
import numpy as np
import numpy.ma as ma

def gen_background_combinations(models, background_combinations):
    model_combinations = []
    for pair in background_combinations:
        model_combinations.append((models[pair[0]], models[pair[1]]))
    return model_combinations

def gen_model_combinations(models, profiling_combinations=None):
    # id_combinations = [i for i in range(len(models)) for j in range(combination_len)]
    # id_combinations = set(itertools.combinations(id_combinations, combination_len))
    # print(id_combinations)
    # for profiled in done_combinations:
    #     id_combinations.remove(profiled)
    # id_combinations = list(id_combinations)
    # id_combinations = sorted(id_combinations, key=lambda x: (x[0], x[1]))
    model_combinations = []
    for id_comb in profiling_combinations:
        model_comb = []
        for id in id_comb:
            model_comb.append(models[id])
        model_combinations.append(model_comb)
    print(model_combinations)
    return model_combinations

def gen_partition(model_len, if_qos=False, if_new=False):
    start = 0
    end = model_len
    if if_new == False:
        start = random.randrange(0, model_len)
    if if_qos == False:
        # start = random.randrange(0, model_len - 3)
        # end = start + 4 + random.randrange(0, model_len - start - 3)
        end = random.randrange(start, model_len + 1)
    if start <= end:
        return start, end
    else:
        raise ValueError

def make_record(model_config, raw_record):
    record_max = np.max(raw_record)
    record_min = np.min(raw_record)
    formated_record = np.delete(
        raw_record, np.where((raw_record == record_max) | (raw_record == record_min))
    )
    median = np.median(formated_record)
    mean = np.mean(formated_record)
    var = np.std(formated_record)
    record = [j for sub in model_config for j in sub]
    record.append(median)
    record.append(mean)
    record.append(var)
    return record
