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


def gen_partition(model_len):
    start = random.randrange(0, model_len - 4)
    end = start + 4 + random.randrange(0, model_len - start - 4)
    if start < end:
        return start, end
    else:
        return end, start


if __name__ == "__main__":
    mp.set_start_method("spawn")
    total_models = 2
    worker_model_list = ["resnet101", "resnet101"]
    supported_batchsize = [1, 2, 4, 8, 16]
    total_test = 10000
    test_loop = 1000

    barrier = mp.Barrier(total_models + 1)

    worker_list = []
    for model_name in worker_model_list:
        pipe_parent, pipe_child = mp.Pipe()
        model_worker = ModelProc(model_name, supported_batchsize, pipe_child, barrier)
        model_worker.start()
        worker_list.append((model_worker, pipe_parent))
    barrier.wait()

    supported_batchsizes = [supported_batchsize for _ in range(total_models)]
    for bs_it in itertools.product(*supported_batchsizes):
        model_config = []
        for i in range(total_models):
            start, end = gen_partition(model_len[worker_model_list[i]])
            model_worker, model_pipe = worker_list[i]
            model_pipe.send((worker_model_list[i], "prepare", start, end, bs_it[i]))
            model_config.append([start, end])
        barrier.wait()
        record = []
        for test_iter in range(10):
            start_time = datetime.datetime.now()
            for i in range(total_models):
                _, model_pipe = worker_list[i]
                model_pipe.send(
                    (
                        worker_model_list[i],
                        "forward",
                        model_config[i][0],
                        model_config[i][1],
                        bs_it[i],
                    )
                )
            barrier.wait()
            elapsed_time = datetime.datetime.now() - start_time
            print(elapsed_time.microseconds)
            record.append(elapsed_time.microseconds)
        avg = np.average(record)
        median = np.median(record)
        diff = np.std(record)
        print(avg)
        print(median)
        print(diff)

    for i in range(total_models):
        _, model_pipe = worker_list[i]
        model_pipe.send(("none", "terminate", -1, -1, -1))

    for worker, _ in worker_list:
        worker.join()