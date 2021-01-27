#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import os
import datetime
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from abacus.option import RunConfig
from abacus.utils import timestamp
from abacus.worker import ServerWorker
from abacus.modeling.models import MLPregression


class Scheduler(Process):
    def __init__(
        self, policy, serve_combinations, predictor_path, barrier, queues, pipes
    ) -> None:
        self._policy = policy
        self._predictor = MLPregression()
        self._predictor.load_state_dict(torch.load(predictor_path, map_location="cpu"))
        self._barrier = barrier
        self._queues = queues
        self._pipes = pipes

    def run(self) -> None:
        timestamp("Scheduler", "Ready")
        self._scheduling_queries = {}
        while True:
            pass

    def FCFS_schedule(self):
        pass

    def Abacus_schedule(self):
        pass

    def SJF_schedule(self):
        pass


class AbacusServer:
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._barrier = mp.Barrier(run_config.total_models + 1)
        self._workers = {}
        self._queues = {}
        self._pipes = {}

    def send_query(self, model_name, batch_size, seq_length):
        query_start = datetime.datetime.now()
        self._queues[model_name].put((model_name, batch_size, seq_length, query_start))

    def start_up(self):
        timestamp("All Server Workers", "Initializing")
        for model_id in self._run_config.serve_combination:
            model_name = self._run_config.models_name[model_id]
            pipe_parent, pipe_child = mp.Pipe()
            model_worker = ServerWorker(
                self._run_config,
                model_name,
                self._run_config.supported_batchsize,
                self._run_config.supported_seqlen,
                pipe_child,
                self._barrier,
            )
            model_worker.start()
            self._workers[model_name] = model_worker
            self._queues[model_name] = mp.Queue()
            self._pipes[model_name] = pipe_parent
        self._barrier.wait()
        timestamp("All Server Workers", "Initialized")
        timestamp("Scheduler", "Initializing")
        predictor_path = os.path.join(self._run_config.path, "model/predictor.ckpt")
        self._scheduler = Scheduler(
            policy=self._run_config.policy,
            serve_combinations=self._run_config.serve_combination,
            predictor_path=predictor_path,
            barrier=self._barrier,
            queues=self._queues,
            pipes=self._pipes,
        )
        self._scheduler.start()
        timestamp("Scheduler", "Initialized")
