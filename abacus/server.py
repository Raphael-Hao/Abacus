#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import os
import datetime
import torch
import csv
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from abacus.option import RunConfig
from abacus.utils import timestamp
from abacus.worker import ServerWorker
from abacus.modeling.models import MLPregression


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

    def __init__(self, model_id, batch_size, seq_len, qos_target=50) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.start_pos = 0
        self.end_pos = 0
        self.op_len = self.MODELS_LEN[model_id]
        self.start_stamp = datetime.datetime.now()
        self.qos_targt = qos_target

    def remain_ops(self):
        return self.op_len - self.end_pos

    def set_op_pos(self, new_pos):
        self.start_pos = self.end_pos
        if new_pos == -1:
            self.end_pos = self.op_len
        else:
            self.end_pos = new_pos
        return self.start_pos, self.end_pos

    def get_op_pos(self):
        return self.start_pos, self.end_pos

    def get_headromm(self):
        return (
            self.qos_targt
            - (datetime.datetime.now() - self.start_stamp).microseconds / 1000
        )

    @property
    def latency_ms(self):
        return (datetime.datetime.now() - self.start_stamp).microseconds / 1000


class Scheduler(Process):
    def __init__(
        self,
        run_config: RunConfig,
        barrier,
        queues,
        pipes,
    ) -> None:
        self._policy = run_config.policy
        self._serve_combination = run_config.serve_combination
        result_dir = os.path.join(run_config.path, "data/server/" + self._policy)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        result_fname = ""
        for model_id in self._serve_combination:
            result_fname += run_config.models_name[model_id]
        result_fname += ".csv"
        self._result_path = os.path.join(result_dir, result_fname)

        self._barrier = barrier
        self._queues = queues
        self._pipes = pipes
        self._models_feature = len(Query.MODELS_LEN)
        self._input_features = 4

    def run(self) -> None:
        self._result_file = open(self._result_path, "w+")
        self._wr = csv.writer(self._result_path, dialect="excel")
        result_header = ["model_id", "bs", "seq_len", "latency"]
        self._wr.writerow(result_header)
        self._result_file.flush()
        self._predictor = MLPregression()
        predictor_ckpt_path = os.path.join(
            self._run_config.path, "model/predictor.ckpt"
        )
        self._predictor.load_state_dict(
            torch.load(predictor_ckpt_path, map_location="cpu")
        )
        timestamp("Scheduler", "Ready")
        self._scheduling_queries = {key: None for key in self._serve_combination}
        while True:
            self.prepare_queries()
            self.SJF_schedule()

    def prepare_queries(self):
        for model_id in self._serve_combination:
            if self._scheduling_queries[model_id] is None:
                self._scheduling_queries[model_id] = self._queues[model_id].get()

    def Abacus_schedule(self):
        raise NotImplementedError

    def FCFS_schedule(self):
        waiting_scheduling = {}
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            query.set_op_pos(-1)
            arrival_stamp = query.start_stamp
            waiting_scheduling[model_id] = arrival_stamp
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                "new", query.start_pos, query.end_pos, query.batch_size, query.seq_len
            )
            self._barrier.wait()
            self.clean_scheduled_queries(model_id=model_id)
            self._wr.writerow(
                query.model_id, query.batch_size, query.seq_len, query.latency_ms
            )
        self._result_file.flush()

    def SJF_schedule(self):
        waiting_scheduling = {}
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            query.set_op_pos(-1)
            latency = self._predictor(self.get_feature(query))
            waiting_scheduling[model_id] = latency
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                "new", query.start_pos, query.end_pos, query.batch_size, query.seq_len
            )
            self._barrier.wait()
            self.clean_scheduled_queries(model_id=model_id)
            self._wr.writerow(
                query.model_id, query.batch_size, query.seq_len, query.latency_ms
            )
        self._result_file.flush()

    def clean_scheduled_queries(self, model_id):
        self._scheduling_queries[model_id] = None

    def get_feature(self, query_l: Query, query_r: Query = None):
        input_feature = torch.zeros(15)
        input_feature[query_l.model_id] += 1
        input_feature[7] = query_l.start_pos
        input_feature[8] = query_l.end_pos
        input_feature[9] = query_l.batch_size
        input_feature[10] = query_l.seq_len
        if query_r is not None:
            input_feature[query_r.model_id] += 1
            input_feature[11] = query_r.start_pos
            input_feature[12] = query_r.end_pos
            input_feature[13] = query_r.batch_size
            input_feature[14] = query_r.seq_len
        return input_feature

    def find_urgent(self):
        pass


class AbacusServer:
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        if self._run_config.policy == "Abacus":
            self._barrier = mp.Barrier(run_config.total_models + 1)
        else:
            self._barrier = mp.Barrier(2)
        self._workers = {}
        self._queues = {}
        self._pipes = {}

    def send_query(self, model_id, batch_size, seq_len):
        self._queues[model_id].put(
            Query(model_id=model_id, batch_size=batch_size, seq_len=seq_len)
        )

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
            self._workers[model_id] = model_worker
            self._queues[model_id] = mp.Queue()
            self._pipes[model_id] = pipe_parent
        self._barrier.wait()
        timestamp("All Server Workers", "Initialized")
        timestamp("Scheduler", "Initializing")
        self._scheduler = Scheduler(
            run_config=self._run_config,
            barrier=self._barrier,
            queues=self._queues,
            pipes=self._pipes,
        )
        self._scheduler.start()
        timestamp("Scheduler", "Initialized")
