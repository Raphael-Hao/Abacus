#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import sys
import time
import itertools
import random
import numpy as np
import numpy.ma as ma


def timestamp(name, stage):
    print("TIMESTAMP, %s, %s, %f" % (name, stage, time.time()), file=sys.stderr)


def gen_background_combinations(models, background_combinations):
    model_combinations = []
    for pair in background_combinations:
        model_combinations.append((models[pair[0]], models[pair[1]]))
    return model_combinations


def gen_model_combinations(models, combination_len, done_combinations=None):
    id_combinations = [i for i in range(len(models)) for j in range(combination_len)]
    id_combinations = set(itertools.combinations(id_combinations, combination_len))
    print(id_combinations)
    for profiled in done_combinations:
        id_combinations.remove(profiled)
    model_combinations = []
    for id_comb in id_combinations:
        model_comb = []
        for id in id_comb:
            model_comb.append(models[id])
        model_combinations.append(model_comb)
    print(model_combinations)
    return model_combinations


def gen_partition(model_len):
    start = random.randrange(0, model_len - 3)
    end = start + 4 + random.randrange(0, model_len - start - 3)
    if start < end:
        return start, end
    else:
        return end, start


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
