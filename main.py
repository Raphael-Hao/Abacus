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


if __name__ == "__main__":
    mp.set_start_method("spawn")
    total_models = 2
    supported_batchsize = [1, 2, 4, 8, 16]
    total_test = 5000
    test_loop = 100

    barrier = mp.Barrier(total_models + 1)

    profiled_model_list = [
        "resnet50",
        "resnet101",
        "resnet152",
        "inception_v3",
        "vgg16",
        "vgg19",
    ]

    profiled_model_lists = [profiled_model_list for _ in range(total_models)]

    for model_permutation in itertools.product(*profiled_model_lists):
        profile_filename = model_permutation[0]
        for model_name in model_permutation[1:]:
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
        for model_name in model_permutation:
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = ModelProc(
                model_name, supported_batchsize, pipe_child, barrier
            )
            model_worker.start()
            worker_list.append((model_worker, pipe_parent))
        barrier.wait()

        supported_batchsizes = [supported_batchsize for _ in range(total_models)]
        for bs_it in itertools.product(*supported_batchsizes):
            for test_i in range(total_test):
                model_config = []

                for i in range(total_models):
                    start, end = gen_partition(model_len[model_permutation[i]])
                    model_config.append([model_permutation[i], start, end, bs_it[i]])
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