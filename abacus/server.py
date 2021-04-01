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
from abacus.modeling.predictor.mlp import MLPregression


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

    def __init__(self, model_id, batch_size, seq_len, qos_target=60) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.start_pos = 0
        self.end_pos = 0
        self.op_len = self.MODELS_LEN[model_id]
        self.start_stamp = time.time()
        self.qos_targt = qos_target
        self.state = "new"

    def remain_ops(self):
        return self.op_len - self.end_pos

    def set_op_pos(self, new_pos):
        if self.end_pos != 0:
            self.state = "inter"
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

    def if_processed(self):
        return self.end_pos == self.op_len

    @property
    def latency_ms(self):
        return (time.time() - self.start_stamp) * 1000


class AbacusServer:
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._warmup_barrier = mp.Barrier(run_config.total_models + 1)
        if self._run_config.policy == "Abacus":
            self._barrier = []
            for i in range(run_config.total_models):
                self._barrier.append(mp.Barrier(i + 2))
        else:
            self._barrier = [mp.Barrier(2)]
        self._stop_flag = mp.RawValue("i", 1)
        self._workers = {}
        self._queues = {}
        self._pipes = {}
        self._qos_target = run_config.qos_target
        random.seed(0)

    def send_query(self, model_id, batch_size, seq_len):
        self._queues[model_id].put(
            Query(
                model_id=model_id,
                batch_size=batch_size,
                seq_len=seq_len,
                qos_target=self._qos_target,
            )
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
            pipe.send(("terminate", -1, -1, -1, -1, -1))
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
            run_config.path,
            "model/2in7/predictor.ckpt"
            # run_config.path, "model/3in4/predictor.ckpt"
        )
        self._abandon = run_config.abandon
        self._barrier = barrier
        self._queues = queues
        self._pipes = pipes
        self._stop_flag = stop_flag
        self._models_feature = len(Query.MODELS_LEN) + 4 * run_config.total_models

    def run(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self._result_file = open(self._result_path, "w+")
        self._wr = csv.writer(self._result_file, dialect="excel")

        # print(torch.get_num_threads())
        # torch.set_num_threads(56)

        self._predictor = MLPregression(self._models_feature)

        timestamp("Scheduler", "loading predictor")
        self._predictor.load_state_dict(
            torch.load(self._predictor_ckpt_path, map_location="cpu")
        )
        self._predictor.eval()
        timestamp("Scheduler", "Ready")
        self._scheduling_queries = {key: None for key in self._serve_combination}
        self._scheduled_queries = {key: None for key in self._serve_combination}

        result_header = ["model_id", "bs", "seq_len", "latency"]
        if self._policy == "Abacus":
            self._schedule_func = self.Abacus_schedule
            # the number of scheduled queries in last round of scheduling, equal with schedule_flag + 1
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            result_header = ["model_id", "bs", "seq_len", "search_times", "latency"]
        elif self._policy == "SJF":
            self._schedule_func = self.SJF_schedule
        elif self._policy == "FCFS":
            self._schedule_func = self.FCFS_schedule
        elif self._policy == "EDF":
            self._schedule_func = self.EDF_schedule
        else:
            raise NotImplementedError
        self._wr.writerow(result_header)
        self._result_file.flush()

        if self._policy == "FCFS" or self._policy == "SJF" or self._policy == "EDF":
            while self._stop_flag.value == 1:
                self.pop_queries()
                self._schedule_func(self._abandon)
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
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        if len(waiting_scheduling) > 0:
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            # print(qos)
            # qos = waiting_scheduling[0][1]
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)
            total_colocated = len(waiting_scheduling)
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        self._predicted_latency = self._predictor(
                            self.get_layer_feature(0, 0, 1, l_query=qos_query)[1]
                        )[0]
                    self.Abacus_reset()
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduled_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    self.Abacus_reset()
                    self._pipes[qos_id].send(
                        (
                            qos_query.state,
                            0,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self.clean_scheduled_queries(model_ids=qos_id)
                    self._schedule_flag = 0
            elif total_colocated == 2:
                bg_id = waiting_scheduling[1][0]
                bg_query: Query = self._scheduling_queries[bg_id]
                start_pos = bg_query.end_pos
                end_pos = bg_query.op_len
                searched_pos = None
                latencies = None
                search_times = 0
                while start_pos < end_pos:
                    search_times += 1
                    search_ways = (
                        self._search_ways
                        if (end_pos - start_pos) > self._search_ways
                        else (end_pos - start_pos)
                    )
                    op_poses, features = self.get_layer_feature(
                        start_pos=start_pos,
                        end_pos=end_pos,
                        search_ways=search_ways,
                        l_query=qos_query,
                        m_query=bg_query,
                    )
                    with torch.no_grad():
                        latencies = self._predictor(features).numpy()
                    # print(
                    #     "searching for qos target: {} form start pos: {}, end pos: {}, search po: {}, predicted: {}".format(
                    #         qos, start_pos, end_pos, op_poses[0], latencies
                    #     )
                    # )
                    for i in range(search_ways):
                        headroom = qos - latencies[i]
                        if headroom < 0:
                            end_pos = op_poses[i]
                        else:
                            if headroom < self._threashold:
                                searched_pos = op_poses[i]
                                self._predicted_latency = latencies[i]
                                break
                            else:
                                start_pos = op_poses[i] + 1
                    if searched_pos is not None:
                        break
                self.Abacus_reset(search_times=search_times)
                if searched_pos is None:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduled_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                else:
                    bg_query.set_op_pos(searched_pos)
                    if searched_pos > 0:
                        self._pipes[qos_id].send(
                            (
                                qos_query.state,
                                1,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[bg_id].send(
                            (
                                bg_query.state,
                                1,
                                bg_query.start_pos,
                                bg_query.end_pos,
                                bg_query.batch_size,
                                bg_query.seq_len,
                            )
                        )
                        if bg_query.if_processed():
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[bg_id] = bg_query
                            self.clean_scheduled_queries(model_ids=[bg_id, qos_id])
                        else:
                            self._scheduled_queries[qos_id] = qos_query
                            self.clean_scheduled_queries(model_ids=qos_id)
                        self._schedule_flag = 1
                    elif searched_pos == 0:
                        self._pipes[qos_id].send(
                            (
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduled_queries(model_ids=qos_id)
                        self._schedule_flag = 0
            elif total_colocated >= 3:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                m_id = waiting_scheduling[1][0]
                m_query: Query = self._scheduling_queries[m_id]
                r_id = waiting_scheduling[2][0]
                r_query: Query = self._scheduling_queries[r_id]
                searched_query = None
                searched_pos = None
                features = self.get_query_feature(
                    total_colocated, qos_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if qos >= latencies[2]:
                    searched_query = 3
                elif qos >= latencies[1]:
                    searched_query = 2
                elif qos >= latencies[0]:
                    searched_query = 1
                else:
                    searched_query = 0

                if searched_query == 3:
                    m_query.set_op_pos(-1)
                    r_query.set_op_pos(-1)
                elif searched_query == 2:
                    m_query.set_op_pos(-1)
                    start_pos = r_query.end_pos
                    end_pos = r_query.op_len
                    searched_pos = None
                    while start_pos < end_pos:
                        search_times += 1
                        search_ways = (
                            self._search_ways
                            if (end_pos - start_pos) > self._search_ways
                            else (end_pos - start_pos)
                        )
                        op_poses, features = self.get_layer_feature(
                            start_pos=start_pos,
                            end_pos=end_pos,
                            search_ways=search_ways,
                            l_query=qos_query,
                            m_query=m_query,
                            r_query=r_query,
                        )
                        with torch.no_grad():
                            latencies = self._predictor(features).numpy()
                        for i in range(search_ways):
                            headroom = qos - latencies[i]
                            if headroom < 0:
                                end_pos = op_poses[i]
                            else:
                                if headroom < self._threashold:
                                    searched_pos = op_poses[i]
                                    self._predicted_latency = latencies[i]
                                    break
                                else:
                                    start_pos = op_poses + 1
                        if searched_pos is not None:
                            break
                elif searched_query == 1:
                    start_pos = m_query.end_pos
                    end_pos = m_query.op_len
                    searched_pos = None
                    while start_pos < end_pos:
                        search_times += 1
                        search_ways = (
                            self._search_ways
                            if (end_pos - start_pos) > self._search_ways
                            else (end_pos - start_pos)
                        )
                        op_poses, features = self.get_layer_feature(
                            start_pos=start_pos,
                            end_pos=end_pos,
                            search_ways=search_ways,
                            l_query=qos_query,
                            m_query=m_query,
                        )
                        with torch.no_grad():
                            latencies = self._predictor(features).numpy()
                        for i in range(search_ways):
                            headroom = qos - latencies[i]
                            if headroom < 0:
                                end_pos = op_poses[i]
                            else:
                                if headroom < self._threashold:
                                    searched_pos = op_poses[i]
                                    self._predicted_latency = latencies[i]
                                    break
                                else:
                                    start_pos = op_poses + 1
                        if searched_pos is not None:
                            break
                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.state,
                                0,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduled_queries(model_ids=qos_id)
                        self._schedule_flag = 0
                elif searched_query == 1:
                    self._pipes[qos_id].send(
                        (
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[m_id].send(
                        (
                            m_query.state,
                            1,
                            m_query.start_pos,
                            m_query.end_pos,
                            m_query.batch_size,
                            m_query.seq_len,
                        )
                    )
                    if m_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[m_id] = m_query
                        self.clean_scheduled_queries(model_ids=[qos_id, m_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduled_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query >= 2:
                    self._pipes[qos_id].send(
                        (
                            qos_query.state,
                            2,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[m_id].send(
                        (
                            m_query.state,
                            2,
                            m_query.start_pos,
                            m_query.end_pos,
                            m_query.batch_size,
                            m_query.seq_len,
                        )
                    )
                    self._pipes[r_id].send(
                        (
                            r_query.state,
                            2,
                            r_query.start_pos,
                            r_query.end_pos,
                            r_query.batch_size,
                            r_query.seq_len,
                        )
                    )
                    if r_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[m_id] = m_query
                        self._scheduled_queries[r_id] = r_query
                        self.clean_scheduled_queries(model_ids=[qos_id, m_id, r_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[m_id] = m_query
                        self.clean_scheduled_queries(model_ids=[qos_id, m_id])
                    self._schedule_flag = 2
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            # print(model_id)
            self._wr.writerow(
                np.array([query.model_id, query.batch_size, query.seq_len, 0, -1])
            )
            self.clean_scheduled_queries(model_ids=model_id)
        self._result_file.flush()

    def Abacus_reset(self, search_times=0):
        if self._schedule_flag >= 0:
            self._barrier[self._schedule_flag].wait()
            self._schedule_flag = -1
            for model_id in self._serve_combination:
                query = self._scheduled_queries[model_id]
                if query is not None:
                    self._wr.writerow(
                        [
                            query.model_id,
                            query.batch_size,
                            query.seq_len,
                            self._search_times,
                            query.latency_ms,
                        ]
                    )
                    self._scheduled_queries[model_id] = None
            self._search_times = search_times

    def FCFS_schedule(self, abandon=False):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if abandon == True:
                if headroom > 0:
                    query.set_op_pos(-1)
                    arrival_stamp = query.start_stamp
                    waiting_scheduling[model_id] = arrival_stamp
                else:
                    abandoned_scheduling.append(model_id)
            else:
                query.set_op_pos(-1)
                arrival_stamp = query.start_stamp
                waiting_scheduling[model_id] = arrival_stamp
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                (
                    "new",
                    0,
                    query.start_pos,
                    query.end_pos,
                    query.batch_size,
                    query.seq_len,
                )
            )
            self._barrier[0].wait()
            self.clean_scheduled_queries(model_ids=model_id)
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
            self.clean_scheduled_queries(model_ids=model_id)
        self._result_file.flush()

    def EDF_schedule(self, abandon=False):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if abandon == True:
                if headroom > 0:
                    query.set_op_pos(-1)

                    waiting_scheduling[model_id] = headroom
                else:
                    abandoned_scheduling.append(model_id)
            else:
                query.set_op_pos(-1)

                waiting_scheduling[model_id] = headroom
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        # print(waiting_scheduling)
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                (
                    "new",
                    0,
                    query.start_pos,
                    query.end_pos,
                    query.batch_size,
                    query.seq_len,
                )
            )
            self._barrier[0].wait()
            self._wr.writerow(
                np.array(
                    [query.model_id, query.batch_size, query.seq_len, query.latency_ms]
                )
            )
            self.clean_scheduled_queries(model_ids=model_id)
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array([query.model_id, query.batch_size, query.seq_len, -1])
            )
            self.clean_scheduled_queries(model_ids=model_id)
        self._result_file.flush()

    def clean_scheduled_queries(self, model_ids):
        if isinstance(model_ids, list):
            for i in model_ids:
                self._scheduling_queries[i] = None
        else:
            self._scheduling_queries[model_ids] = None

    def urgent_sort(self, abandon):
        waiting_scheduling = {}
        abandoned_scheduling = []
        for model_id in self._serve_combination:
            query: Query = self._scheduling_queries[model_id]
            if query is None:
                continue
            headroom = query.get_headromm()
            if abandon == True:
                if headroom > 0:
                    waiting_scheduling[model_id] = headroom
                else:
                    abandoned_scheduling.append(model_id)
            else:
                waiting_scheduling[model_id] = headroom
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        return waiting_scheduling, abandoned_scheduling

    def get_query_feature(
        self,
        search_ways,
        qos_query: Query,
        l_query: Query = None,
        m_query: Query = None,
        r_query: Query = None,
    ):
        input_feature = torch.zeros(search_ways, self._models_feature)
        input_feature[0:search_ways, qos_query.model_id] += 1
        input_feature[0:search_ways, 7] = qos_query.start_pos
        input_feature[0:search_ways, 8] = qos_query.end_pos
        input_feature[0:search_ways, 9] = qos_query.batch_size
        input_feature[0:search_ways, 10] = qos_query.seq_len
        if l_query is not None:
            input_feature[1:search_ways, l_query.model_id] += 1
            input_feature[1:search_ways, 11] = l_query.end_pos
            input_feature[1:search_ways, 12] = l_query.op_len
            input_feature[1:search_ways, 13] = l_query.batch_size
            input_feature[1:search_ways, 14] = l_query.seq_len
        if m_query is not None:
            input_feature[2:search_ways, m_query.model_id] += 1
            input_feature[2:search_ways, 15] = m_query.end_pos
            input_feature[2:search_ways, 16] = m_query.op_len
            input_feature[2:search_ways, 17] = m_query.batch_size
            input_feature[2:search_ways, 18] = m_query.seq_len
        if r_query is not None:
            input_feature[3, r_query.model_id] += 1
            input_feature[3, 19] = r_query.end_pos
            input_feature[3, 20] = r_query.op_len
            input_feature[3, 21] = r_query.batch_size
            input_feature[3, 22] = r_query.seq_len
        return input_feature

    def get_layer_feature(
        self,
        start_pos,
        end_pos,
        search_ways,
        qos_query: Query,
        l_query: Query = None,
        m_query: Query = None,
        r_query: Query = None,
    ):
        step = (end_pos - start_pos) // (search_ways + 1)
        poses = []
        input_feature = torch.zeros(search_ways, self._models_feature)
        input_feature[0:search_ways, qos_query.model_id] += 1
        input_feature[0:search_ways, 7] = qos_query.start_pos
        input_feature[0:search_ways, 8] = qos_query.end_pos
        input_feature[0:search_ways, 9] = qos_query.batch_size
        input_feature[0:search_ways, 10] = qos_query.seq_len
        if l_query is not None:
            raise NotImplementedError
        if r_query is not None:
            input_feature[0:search_ways, r_query.model_id] += 1
            input_feature[0:search_ways, 11] = r_query.start_pos
            input_feature[0:search_ways, 12] = r_query.end_pos
            input_feature[0:search_ways, 13] = r_query.batch_size
            input_feature[0:search_ways, 14] = r_query.seq_len
            input_feature[0:search_ways, r_query.model_id] += 1
            input_feature[0:search_ways, 15] = r_query.end_pos
            input_feature[0:search_ways, 17] = r_query.batch_size
            input_feature[0:search_ways, 18] = r_query.seq_len
            for i in range(search_ways):
                start_pos += step
                input_feature[i, 16] = start_pos
                poses.append(start_pos)
        if m_query is not None:
            input_feature[0:search_ways, m_query.model_id] += 1
            input_feature[0:search_ways, 11] = m_query.end_pos
            input_feature[0:search_ways, 13] = m_query.batch_size
            input_feature[0:search_ways, 14] = m_query.seq_len
            for i in range(search_ways):
                start_pos += step
                input_feature[i, 12] = start_pos
                poses.append(start_pos)
        return poses, input_feature