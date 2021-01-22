#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from lego.worker import ModelProc, model_list, model_len
from lego.utils import timestamp

import torch.multiprocessing as mp

import torch
import datetime
import numpy as np
import itertools
import random
from tqdm import tqdm
import csv


def gen_partition(model_len):
    start = random.randrange(0, model_len - 4)
    end = start + 4 + random.randrange(0, model_len - start - 4)
    if start < end:
        return start, end
    else:
        return end, start


def make_record(model_config, median, mean, var):
    record = [j for sub in model_config for j in sub]
    record.append(median)
    record.append(mean)
    record.append(var)
    return record


def gen_model_combinations(models, combination_len):
    id_combinations = [i for i in range(len(models)) for j in range(combination_len)]
    id_combinations = set(itertools.combinations(id_combinations, combination_len))
    model_combinations = []
    for id_comb in id_combinations:
        model_comb = []
        for id in id_comb:
            model_comb.append(models[id])
        model_combinations.append(model_comb)
    return model_combinations


if __name__ == "__main__":
    mp.set_start_method("spawn")
    total_models = 2
    supported_batchsize = [1, 2, 4, 8, 16]
    total_test = 200
    test_loop = 100

    barrier = mp.Barrier(total_models + 1)

    all_profiled_models = [
        "resnet50",
        "resnet101",
        "resnet152",
        "inception_v3",
        "vgg16",
        "vgg19",
        "bert"
    ]

    for model_combination in gen_model_combinations(all_profiled_models, total_models):
        profile_filename = model_combination[0]
        for model_name in model_combination[1:]:
            profile_filename = profile_filename + "_" + model_name
        profile_filename += ".csv"
        profile_file = open(profile_filename, "w")
        wr = csv.writer(profile_file, dialect="excel")
        profile_head = ["model", "start", "end", "bs"] * total_models + [
            "median",
            "mean",
            "var",
        ]
        wr.writerow(profile_head)

        worker_list = []
        for model_name in model_combination:
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = ModelProc(
                model_name, supported_batchsize, pipe_child, barrier
            )
            model_worker.start()
            worker_list.append((model_worker, pipe_parent))
        barrier.wait()

        for bs_it in itertools.product(supported_batchsize,repeat=2):
            for test_i in range(total_test):
                model_config = []
                for i in range(total_models):
                    start, end = gen_partition(
                        model_len[model_combination[i]]
                    )
                    model_config.append([model_combination[i], start, end, bs_it[i]])
                    model_worker, model_pipe = worker_list[i]
                    model_pipe.send(
                        (
                            model_config[i][0],
                            "prepare",
                            model_config[i][1],
                            model_config[i][2],
                            model_config[i][3],
                        )
                    )

                barrier.wait()
                record = []
                with tqdm(range(test_loop)) as t:
                    for loop_i in t:
                        start_time = datetime.datetime.now()
                        for i in range(total_models):
                            _, model_pipe = worker_list[i]
                            model_pipe.send(
                                (
                                    model_config[i][0],
                                    "forward",
                                    model_config[i][1],
                                    model_config[i][2],
                                    model_config[i][3],
                                )
                            )
                        barrier.wait()
                        elapsed_time_us = (
                            datetime.datetime.now() - start_time
                        ).microseconds
                        t.set_postfix(elapsed=elapsed_time_us)
                        t.update()
                        record.append(elapsed_time_us)

                profile_record = make_record(
                    model_config, np.median(record), np.average(record), np.std(record)
                )
                wr.writerow(profile_record)
                profile_file.flush()
        for i in range(total_models):
            _, model_pipe = worker_list[i]
            model_pipe.send(("none", "terminate", -1, -1, -1))

        for worker, _ in worker_list:
            worker.join()