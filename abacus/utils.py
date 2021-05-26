#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import sys
import time
import random
import numpy as np
import numpy.ma as ma
import logging


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


class Query:
    MODELS_LEN = {
        0: 18,
        1: 35,
        2: 52,
        3: 14,
        4: 19,
        5: 22,
        6: 12,
    }

    def __init__(
        self,
        id,
        model_id,
        batch_size,
        seq_len,
        start_stamp=None,
        qos_target=60,
        load_id=0,
    ) -> None:
        self.id = id
        self.model_id = model_id
        self.load_id = load_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.start_pos = 0
        self.end_pos = 0
        self.op_len = self.MODELS_LEN[model_id]
        if start_stamp == None:
            self.start_stamp = time.time()
        else:
            self.start_stamp = start_stamp
        logging.debug("start_stamp: {:.2f}".format(self.start_stamp))
        self.qos_targt = qos_target
        self.state = "new"

    def remain_ops(self):
        return self.op_len - self.end_pos

    def set_op_pos(self, new_pos):
        if self.end_pos != 0:
            self.state = "inter"
        self.start_pos = self.end_pos
        if new_pos == -1:
            self.end_pos = self.op_len
        else:
            self.end_pos = new_pos
        return self.start_pos, self.end_pos

    def get_op_pos(self):
        return self.start_pos, self.end_pos

    def get_headromm(self):
        head_room = self.qos_targt - (time.time() - self.start_stamp) * 1000
        logging.debug("headroom: {:.2f}".format(head_room))
        return head_room

    def if_processed(self):
        return self.end_pos == self.op_len

    def latency_ms(self):
        return (time.time() - self.start_stamp) * 1000

    def __str__(self) -> str:
        return "model_id: {}, bs: {}, seq_len: {}, start: {}, end: {}, op_len: {}, qos_target: {}, state: {}".format(
            self.model_id,
            self.batch_size,
            self.seq_len,
            self.start_pos,
            self.end_pos,
            self.op_len,
            self.qos_targt,
            self.state,
        )
