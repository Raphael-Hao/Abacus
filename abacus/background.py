#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import torch.multiprocessing as mp
import datetime
import numpy as np
import itertools
import random
from tqdm import tqdm
import csv

from abacus.worker import BackgroundWorker
from abacus.utils import gen_background_combinations, gen_partition, make_record


def background(args):
    mp.set_start_method("spawn")
    barrier = mp.Barrier(args.total_models + 1)
    barrier2 = mp.Barrier(2)

    for model_combination in gen_background_combinations(
        args.models_name, args.background_combinations
    ):
        print(model_combination)
        profile_filename = "background_" + model_combination[0]
        for model_name in model_combination[1:]:
            profile_filename = profile_filename + "_" + model_name
        profile_filename += ".csv"
        profile_file = open(profile_filename, "a+")
        wr = csv.writer(profile_file, dialect="excel")
        profile_head = [
            "model",
            "bs",
            "seq_len",
            "latency",
        ]
        wr.writerow(profile_head)

        id = 0
        worker_list = []
        shared_flag = mp.RawValue('i', 1)
        for model_name in model_combination:
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = BackgroundWorker(
                args,
                model_name,
                args.supported_batchsize,
                args.supported_seqlen,
                pipe_child,
                barrier,
                barrier2,
                isBackground = False if id == 0 else True,
                shared_flag = None if id == 0 else shared_flag
            )
            model_worker.start()
            worker_list.append((model_worker, pipe_parent))
            id = id + 1
        barrier.wait()

        model_name = model_combination[0]
        bs = args.supported_batchsize[4]
        print("batch size : {} sent to server".format(bs))
        seq_len = 0
        _, model_pipe = worker_list[0]
        with tqdm(range(args.total_test)) as t:
            for loop_i in t:
                start_time = datetime.datetime.now()
                model_pipe.send(
                    (
                        model_name,
                        "forward",
                        bs,
                        seq_len,
                    )
                )
                barrier2.wait()
                elapsed_time_us = (
                    datetime.datetime.now() - start_time
                ).microseconds
                t.set_postfix(elapsed=elapsed_time_us)
                t.update(1)
                wr.writerow((model_name, bs, seq_len, elapsed_time_us))
                profile_file.flush()

        model_pipe.send((None, "terminate", -1, -1))

        shared_flag.value = 0
        for worker, _ in worker_list:
            worker.join()