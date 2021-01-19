#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from lego.worker import ModelProc
from lego.utils import timestamp

import torch.multiprocessing as mp
import torch
import datetime
from time import sleep

if __name__ == "__main__":
    mp.set_start_method("spawn")

    num_models = 3
    barrier = mp.Barrier(num_models + 1)
    worker_model_list = ["resnet50", "resnet101", "resnet152"]
    worker_list = []
    for model_name in worker_model_list:
        pipe_parent, pipe_child = mp.Pipe()
        model_worker = ModelProc(model_name, pipe_child, barrier)
        model_worker.start()
        worker_list.append((model_worker, pipe_parent))
    barrier.wait()

    for _i in range(100):
        start_time = datetime.datetime.now()
        for i in range(num_models):
            _, model_pipe = worker_list[i]
            model_pipe.send((worker_model_list[i],1, 2))
        barrier.wait()
        elapsed_time = datetime.datetime.now() - start_time
        print(elapsed_time)

    for i in range(num_models):
        _, model_pipe = worker_list[i]
        model_pipe.send(("terminate", 1, 2))

    for worker, _ in worker_list:
        worker.join()