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
import os

from abacus.option import RunConfig
from abacus.worker import ProfilerWorker
from abacus.utils import gen_model_combinations, gen_partition, make_record


def profile(run_config: RunConfig):
    barrier = mp.Barrier(run_config.total_models + 1)
    profile_data_path = os.path.join(run_config.data_path, "profile")
    os.makedirs(profile_data_path, exist_ok=True)
    for model_combination in gen_model_combinations(
        run_config.models_name,
        run_config.profiling_combinations,
    ):
        print(model_combination)
        profile_filename = model_combination[0]
        for model_name in model_combination[1:]:
            profile_filename = profile_filename + "_" + model_name
        profile_filename += ".csv"
        profile_file = open(os.path.join(profile_data_path, profile_filename), "w+")
        wr = csv.writer(profile_file, dialect="excel")
        profile_head = [
            "model",
            "start",
            "end",
            "bs",
            "seq_len",
        ] * run_config.total_models + [
            "median",
            "mean",
            "var",
        ]
        wr.writerow(profile_head)

        worker_list = []
        for worker_id, model_name in enumerate(model_combination):
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = ProfilerWorker(
                run_config,
                model_name,
                run_config.supported_batchsize,
                run_config.supported_seqlen,
                pipe_child,
                barrier,
                worker_id,
            )
            model_worker.start()
            worker_list.append((model_worker, pipe_parent))
        barrier.wait()

        for bs_it in itertools.product(
            run_config.supported_batchsize, repeat=run_config.total_models
        ):
            model_ids = [i for i in range(run_config.total_models)]
            profiled_config = set()
            for test_i in range(run_config.total_test):
                model_config = []
                qos_query_cnt = random.randrange(1, run_config.total_models + 1)
                new_query_cnt = random.randrange(1, run_config.total_models + 1)
                qos_ids = random.sample(model_ids, qos_query_cnt)
                new_ids = random.sample(model_ids, new_query_cnt)
                for i in range(run_config.total_models):
                    start, end = gen_partition(
                        run_config.models_len[model_combination[i]],
                        True if i in qos_ids else False,
                        True if i in new_ids else False,
                    )
                    seq_len = (
                        random.choice(run_config.supported_seqlen)
                        if model_combination[i] == "bert"
                        else 0
                    )
                    model_config.append(
                        [model_combination[i], start, end, bs_it[i], seq_len]
                    )
                pendding_profile_config = tuple(tuple(i) for i in model_config)
                if pendding_profile_config in profiled_config:
                    print(
                        "Profiled model config: {}, {}, {}, {}, {},{}, {}, {}, {}, {}".format(
                            model_config[0][0],
                            model_config[0][1],
                            model_config[0][2],
                            model_config[0][3],
                            model_config[0][4],
                            model_config[1][0],
                            model_config[1][1],
                            model_config[1][2],
                            model_config[1][3],
                            model_config[1][4],
                        )
                    )
                else:
                    profiled_config.add(pendding_profile_config)
                    for i in range(run_config.total_models):
                        _, model_pipe = worker_list[i]
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
                    with tqdm(range(run_config.test_loop)) as t:
                        for loop_i in t:
                            start_time = datetime.datetime.now()
                            for i in range(run_config.total_models):
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
                            # barrier.wait()
                            # start_time = datetime.datetime.now()
                            barrier.wait()
                            elapsed_time_us = (
                                datetime.datetime.now() - start_time
                            ).microseconds
                            t.set_postfix(elapsed=elapsed_time_us)
                            t.update(1)
                            record.append(elapsed_time_us)

                    profile_record = make_record(model_config, record)
                    wr.writerow(profile_record)
                    profile_file.flush()
        for i in range(run_config.total_models):
            _, model_pipe = worker_list[i]
            model_pipe.send(("none", "terminate", -1, -1, -1, -1))

        for worker, _ in worker_list:
            worker.join()
