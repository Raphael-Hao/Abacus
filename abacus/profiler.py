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

from abacus.worker import ProfilerWorker
from abacus.utils import gen_model_combinations, gen_partition, make_record


def profile(args):
    mp.set_start_method("spawn")
    barrier = mp.Barrier(args.total_models + 1)

    for model_combination in gen_model_combinations(
        args.models_name, args.total_models, args.profiled_combinations
    ):
        print(model_combination)
        profile_filename = model_combination[0]
        for model_name in model_combination[1:]:
            profile_filename = profile_filename + "_" + model_name
        profile_filename += ".csv"
        profile_file = open(profile_filename, "a+")
        wr = csv.writer(profile_file, dialect="excel")
        profile_head = [
            "model",
            "start",
            "end",
            "bs",
            "seq_len",
        ] * args.total_models + [
            "median",
            "mean",
            "var",
        ]
        wr.writerow(profile_head)

        worker_list = []
        for model_name in model_combination:
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = ProfilerWorker(
                args,
                model_name,
                args.supported_batchsize,
                args.supported_seqlen,
                pipe_child,
                barrier,
            )
            model_worker.start()
            worker_list.append((model_worker, pipe_parent))
        barrier.wait()

        for bs_it in itertools.product(args.supported_batchsize, repeat=2):
            for test_i in range(args.total_test):
                model_config = []
                for i in range(args.total_models):
                    start, end = gen_partition(args.models_len[model_combination[i]])
                    seq_len = (
                        random.choice(args.supported_seqlen)
                        if model_combination[i] == "bert"
                        else 0
                    )
                    model_config.append(
                        [model_combination[i], start, end, bs_it[i], seq_len]
                    )
                    model_worker, model_pipe = worker_list[i]
                    model_pipe.send(
                        (
                            model_config[i][0],
                            "prepare",
                            model_config[i][1],
                            model_config[i][2],
                            model_config[i][3],
                            model_config[i][4],
                        )
                    )
                barrier.wait()
                record = []
                with tqdm(range(args.test_loop)) as t:
                    for loop_i in t:
                        start_time = datetime.datetime.now()
                        for i in range(args.total_models):
                            _, model_pipe = worker_list[i]
                            model_pipe.send(
                                (
                                    model_config[i][0],
                                    "forward",
                                    model_config[i][1],
                                    model_config[i][2],
                                    model_config[i][3],
                                    model_config[i][4],
                                )
                            )
                        barrier.wait()
                        elapsed_time_us = (
                            datetime.datetime.now() - start_time
                        ).microseconds
                        t.set_postfix(elapsed=elapsed_time_us)
                        t.update(1)
                        record.append(elapsed_time_us)

                profile_record = make_record(
                    model_config, np.median(record), np.average(record), np.std(record)
                )
                wr.writerow(profile_record)
                profile_file.flush()
        for i in range(args.total_models):
            _, model_pipe = worker_list[i]
            model_pipe.send(("none", "terminate", -1, -1, -1, -1))

        for worker, _ in worker_list:
            worker.join()
