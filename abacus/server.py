#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import os
import time
import numpy as np
import random
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
        self.start_stamp = time.time()
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
        return self.qos_targt - (time.time() - self.start_stamp) * 1000

    @property
    def latency_ms(self):
        return (time.time() - self.start_stamp).microseconds * 1000


class AbacusServer:
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._warmup_barrier = mp.Barrier(run_config.total_models + 1)
        if self._run_config.policy == "Abacus":
            self._barrier = mp.Barrier(run_config.total_models + 1)
        else:
            self._barrier = mp.Barrier(2)
        self._stop_flag = mp.RawValue("i", 1)
        self._workers = {}
        self._queues = {}
        self._pipes = {}
        random.seed(0)

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
                self._warmup_barrier,
            )
            model_worker.start()
            self._workers[model_id] = model_worker
            self._queues[model_id] = mp.Queue()
            self._pipes[model_id] = pipe_parent
        self._warmup_barrier.wait()
        timestamp("All Server Workers", "Initialized")
        timestamp("Scheduler", "Initializing")
        self._scheduler = Scheduler(
            run_config=self._run_config,
            barrier=self._barrier,
            queues=self._queues,
            pipes=self._pipes,
            stop_flag=self._stop_flag,
        )
        self._scheduler.start()
        timestamp("Scheduler", "Initialized")

    def prepare_test_queries(self, total_queries=1000, average_duration=20):
        self._test_queries = []
        for i in range(total_queries):
            model_id = random.choice(self._run_config.serve_combination)
            sleep_duration = random.expovariate(average_duration)
            bs = random.choice(self._run_config.supported_batchsize)
            seq_len = (
                random.choice(self._run_config.supported_seqlen) if model_id == 6 else 0
            )
            self._test_queries.append((model_id, sleep_duration, bs, seq_len))

    def start_test(self):
        self.prepare_test_queries(
            self._run_config.total_queries, self._run_config.average_duration
        )
        i = 0
        for model_id, sleep_duration, bs, seq_len in self._test_queries:
            i += 1
            # timestamp("abacus", "send {}-th query, sleeping {:.5f}".format(i, sleep_duration))
            self.send_query(model_id=model_id, batch_size=bs, seq_len=seq_len)
            time.sleep(sleep_duration)

    def stop_test(self):
        while not self.if_all_processed():
            print("Queries remain to be processed")
            time.sleep(0.05)
        self._stop_flag.value = 0
        for pipe in self._pipes.values():
            pipe.send(("terminate", -1, -1, -1, -1))
        self._scheduler.join()
        for worker in self._workers.values():
            worker.join()

    def if_all_processed(self):
        for queue in self._queues.values():
            if not queue.empty():
                return False
        return True


class Scheduler(Process):
    def __init__(
        self,
        run_config: RunConfig,
        barrier,
        queues,
        pipes,
        stop_flag,
    ) -> None:
        super().__init__()
        self._policy = run_config.policy
        if self._policy == "Abacus":
            self._search_ways = run_config.search_ways
            self._threashold = run_config.threshold
        self._serve_combination = run_config.serve_combination
        result_dir = os.path.join(run_config.path, "data/server/" + self._policy)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_fname = ""
        for model_id in self._serve_combination:
            result_fname += run_config.models_name[model_id]
        result_fname += ".csv"
        self._result_path = os.path.join(result_dir, result_fname)
        self._predictor_ckpt_path = os.path.join(
            run_config.path, "model/predictor.ckpt"
        )

        self._barrier = barrier
        self._queues = queues
        self._pipes = pipes
        self._stop_flag = stop_flag
        self._models_feature = len(Query.MODELS_LEN) + 4 * run_config.total_models

    def run(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self._result_file = open(self._result_path, "w+")
        self._wr = csv.writer(self._result_file, dialect="excel")
        result_header = ["model_id", "bs", "seq_len", "latency"]
        self._wr.writerow(result_header)
        self._result_file.flush()
        self._predictor = MLPregression()

        timestamp("Scheduler", "loading predictor")
        self._predictor.load_state_dict(
            torch.load(self._predictor_ckpt_path, map_location="cpu")
        )
        self._predictor.eval()
        timestamp("Scheduler", "Ready")
        self._scheduling_queries = {key: None for key in self._serve_combination}
        if self._policy == "Abacus":
            self._schedule_func = self.Abacus_schedule
            self._schedule_flag = False
        elif self._policy == "SJF":
            self._schedule_func = self.SJF_schedule
        elif self._policy == "FCFS":
            self._schedule_func = self.FCFS_schedule
        else:
            raise NotImplementedError
        if self._policy == "FCFS" or self._policy == "SJF":
            while self._stop_flag.value == 1:
                self.pop_queries()
                self._schedule_func()
        elif self._policy == "Abacus":
            while self._stop_flag.value == 1:
                self.pop_queries()
                self.Abacus_schedule()

    def pop_queries(self):
        for model_id in self._serve_combination:
            if (self._scheduling_queries[model_id] is None) and (
                not self._queues[model_id].empty()
            ):
                self._scheduling_queries[model_id] = self._queues[model_id].get()

    def Abacus_schedule(self):
        if self._schedule_flag:
            self._barrier.wait()
        waiting_scheduling, abandoned_scheduling = self.urgent_sort()
        qos_id = waiting_scheduling[0][0]
        qos = waiting_scheduling[0][1]
        qos_query: Query = self._scheduling_queries[qos_id]
        qos_query.set_op_pos(-1)
        total_colocated = len(waiting_scheduling)
        if total_colocated == 1:
            with torch.no_grad():
                latency = self._predictor(
                    self.get_layer_feature(0, 0, 1, query_l=qos_query)[1]
                )[0]
            if qos < latency:
                abandoned_scheduling.append(qos_id)
        elif total_colocated == 2:
            bg_id = waiting_scheduling[1][0]
            bg_query: Query = self._scheduling_queries[bg_id]
            start_pos = bg_query.end_pos
            end_pos = bg_query.op_len
            search_ways = (
                self._search_ways
                if (bg_query.end_pos - bg_query.start_pos + 1) > self._search_ways
                else (bg_query.end_pos - bg_query.start_pos + 1)
            )
            while start_pos < end_pos:
                op_poses, features = self.get_layer_feature(
                    start_pos=bg_query.start_pos,
                    end_pos=bg_query.end_pos,
                    search_ways=search_ways,
                    query_l=qos_query,
                    query_m=bg_query,
                )
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                for i in range(search_ways):
                    headroom = qos - latencies[i]
                    if headroom < 0:
                        end_pos = op_poses[i]
                    else:
                        if headroom < self._threashold:
                            bg_query.set_op_pos(op_poses[i])
                            break
                        else:
                            start_pos = op_poses[i]
            ##FIXME check if append the qos query to abandoned one
        elif total_colocated == 3:
            pass

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array([query.model_id, query.batch_size, query.seq_len, -1])
            )
            self.clean_scheduled_queries(model_id=model_id)
        self._result_file.flush()

    def FCFS_schedule(self):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if headroom > 0:
                query.set_op_pos(-1)
                arrival_stamp = query.start_stamp
                waiting_scheduling[model_id] = arrival_stamp
            else:
                abandoned_scheduling.append(model_id)
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                ("new", query.start_pos, query.end_pos, query.batch_size, query.seq_len)
            )
            self._barrier.wait()
            self.clean_scheduled_queries(model_id=model_id)
            self._wr.writerow(
                np.array(
                    [query.model_id, query.batch_size, query.seq_len, query.latency_ms]
                )
            )
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array([query.model_id, query.batch_size, query.seq_len, -1])
            )
            self.clean_scheduled_queries(model_id=model_id)
        self._result_file.flush()

    def SJF_schedule(self):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if headroom > 0:
                query.set_op_pos(-1)
                with torch.no_grad():
                    latency = self._predictor(
                        self.get_layer_feature(
                            start_pos=0, end_pos=0, search_ways=1, query_l=query
                        )
                    )[0]
                waiting_scheduling[model_id] = latency
            else:
                abandoned_scheduling.append(model_id)
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        print(waiting_scheduling)
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                ("new", query.start_pos, query.end_pos, query.batch_size, query.seq_len)
            )
            self._barrier.wait()
            self._wr.writerow(
                np.array(
                    [query.model_id, query.batch_size, query.seq_len, query.latency_ms]
                )
            )
            self.clean_scheduled_queries(model_id=model_id)
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array([query.model_id, query.batch_size, query.seq_len, -1])
            )
            self.clean_scheduled_queries(model_id=model_id)
        self._result_file.flush()

    def clean_scheduled_queries(self, model_id):
        self._scheduling_queries[model_id] = None

    def urgent_sort(self):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if headroom > 0:
                waiting_scheduling[model_id] = headroom
            else:
                abandoned_scheduling.append(model_id)
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        return waiting_scheduling, abandoned_scheduling

    def get_model_feature(
        self,
        search_ways,
        query_l: Query,
        query_m: Query = None,
        query_r: Query = None,
    ):
        input_feature = torch.zeros(search_ways, self._models_feature)
        input_feature[0:search_ways, query_l.model_id] += 1
        input_feature[0:search_ways, 7] = query_l.start_pos
        input_feature[0:search_ways, 8] = query_l.end_pos
        input_feature[0:search_ways, 9] = query_l.batch_size
        input_feature[0:search_ways, 10] = query_l.seq_len
        if query_m is not None:
            input_feature[1:search_ways, query_r.model_id] += 1
            input_feature[1:search_ways, 11] = query_r.end_pos
            input_feature[1:search_ways, 12] = query_r.op_len
            input_feature[1:search_ways, 13] = query_r.batch_size
            input_feature[1:search_ways, 14] = query_r.seq_len
        if query_r is not None:
            input_feature[2, query_r.model_id] += 1
            input_feature[2, 15] = query_r.end_pos
            input_feature[2, 16] = query_r.op_len
            input_feature[2, 17] = query_r.batch_size
            input_feature[2, 18] = query_r.seq_len
        return input_feature

    def get_layer_feature(
        self,
        start_pos,
        end_pos,
        search_ways,
        query_l: Query,
        query_m: Query = None,
        query_r: Query = None,
    ):
        step = (end_pos - start_pos + 1) // search_ways
        poses = []
        input_feature = torch.zeros(search_ways, self._models_feature)
        input_feature[0:search_ways, query_l.model_id] += 1
        input_feature[0:search_ways, 7] = query_l.start_pos
        input_feature[0:search_ways, 8] = query_l.end_pos
        input_feature[0:search_ways, 9] = query_l.batch_size
        input_feature[0:search_ways, 10] = query_l.seq_len
        if query_r is not None:
            input_feature[0:search_ways, query_r.model_id] += 1
            input_feature[0:search_ways, 11] = query_r.start_pos
            input_feature[0:search_ways, 12] = query_r.end_pos
            input_feature[0:search_ways, 13] = query_r.batch_size
            input_feature[0:search_ways, 14] = query_r.seq_len
            input_feature[0:search_ways, query_r.model_id] += 1
            input_feature[0:search_ways, 15] = query_r.end_pos
            input_feature[0:search_ways, 17] = query_r.batch_size
            input_feature[0:search_ways, 18] = query_r.seq_len
            for i in range(search_ways):
                start_pos += step
                if i == search_ways - 1:
                    start_pos = end_pos
                input_feature[i, 16] = start_pos
                poses.append(start_pos)
        if query_m is not None:
            input_feature[0:search_ways, query_r.model_id] += 1
            input_feature[0:search_ways, 11] = query_r.end_pos
            input_feature[0:search_ways, 13] = query_r.batch_size
            input_feature[0:search_ways, 14] = query_r.seq_len
            for i in range(search_ways):
                start_pos += step
                if i == search_ways - 1:
                    start_pos = end_pos
                input_feature[i, 12] = start_pos
                poses.append(start_pos)
        return poses, input_feature
