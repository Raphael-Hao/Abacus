#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import os
import logging
import time
import numpy as np
import random
import torch
import csv
import torch.multiprocessing as mp
from torch.multiprocessing import Process

from abacus.option import RunConfig
from abacus.worker import ServerWorker
from abacus.modeling.predictor.mlp import MLPregression

# import abacus.abacus_pb2 as abacus_pb2
# import abacus.abacus_pb2_grpc as abacus_pb2_grpc

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

    def __init__(self, id, model_id, batch_size, seq_len, qos_target=60) -> None:
        self.id = id
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

    def latency_ms(self):
        return (time.time() - self.start_stamp) * 1000

    def __str__(self) -> str:
        return "model_id: {}, bs: {}, seq_len: {}, start: {}, end: {}, op_len: {}, qos_target: {}, state: {}".format(
            self.model_id,
            self.batch_size,
            self.seq_len,
            self.start_pos,
            self.end_pos,
            self.op_len,
            self.qos_targt,
            self.state,
        )


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
        self._stop_event = mp.Event()
        self._workers = {}
        self._queues = {}
        self._pipes = {}
        self._qos_target = run_config.qos_target
        random.seed(0)

    def send_query(self, id, model_id, batch_size, seq_len):
        self._queues[model_id].put(
            Query(
                id=id,
                model_id=model_id,
                batch_size=batch_size,
                seq_len=seq_len,
                qos_target=self._qos_target,
            )
        )

    def start_up(self):
        logging.info("All Server Workers Initializing")
        for worker_id, model_id in enumerate(self._run_config.serve_combination):
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
                worker_id,
            )
            model_worker.start()
            self._workers[model_id] = model_worker
            self._queues[model_id] = mp.Queue()
            self._pipes[model_id] = pipe_parent
        self._warmup_barrier.wait()
        logging.info("All Server Workers Initialized")
        logging.info("Scheduler Initializing")
        self._scheduler = Scheduler(
            run_config=self._run_config,
            barrier=self._barrier,
            queues=self._queues,
            pipes=self._pipes,
            stop_event=self._stop_event,
        )
        self._scheduler.start()
        logging.info("Scheduler Initialized")

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
            self.send_query(id=i, model_id=model_id, batch_size=bs, seq_len=seq_len)
            time.sleep(sleep_duration)

    def stop_test(self):
        while not self.if_all_processed():
            logging.info("Queries remain to be processed")
            time.sleep(0.01)
        self._stop_event.set()
        self._scheduler.join()
        for pipe in self._pipes.values():
            pipe.send((-1, "terminate", -1, -1, -1, -1, -1))
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
        stop_event,
    ) -> None:
        super().__init__()
        self._policy = run_config.policy
        if self._policy == "Abacus":
            self._search_ways = run_config.search_ways
            self._threashold = run_config.threshold

        if run_config.total_models == 2:
            if run_config.mig == 0:
                predictor_path = "model/A100/2in7/all.ckpt"
                log_path = "results/A100/2in7/" + self._policy
            elif run_config.mig == 2:
                predictor_path = "model/mig/2in4/all.ckpt"
                log_path = "results/mig/2in4/" + self._policy
            else:
                raise NotImplementedError
        elif run_config.total_models == 3:
            if run_config.mig == 0:
                predictor_path = "model/A100/3in4/all.ckpt"
                log_path = "results/A100/3in4/" + self._policy
            else:
                raise NotImplementedError
        elif run_config.total_models == 4:
            if run_config.mig == 0:
                predictor_path = "model/A100/4in4/all.ckpt"
                log_path = "results/A100/4in4/" + self._policy
            elif run_config.mig == 1:
                predictor_path = "model/mig/4in4/all.ckpt"
                log_path = "results/mig/4in4/" + self._policy
            else:
                raise NotImplementedError
        elif run_config.total_models == 1:
            if run_config.mig == 0:
                predictor_path = None
                log_path = "results/A100/1in4/" + self._policy
            elif run_config.mig == 4:
                predictor_path = None
                log_path = "results/mig/1in4/" + self._policy
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        log_dir = os.path.join(run_config.path, log_path)
        os.makedirs(log_dir, exist_ok=True)

        self._serve_combination = run_config.serve_combination
        result_fname = ""
        for model_id in self._serve_combination:
            result_fname += run_config.models_name[model_id]
        result_fname += ".csv"
        self._result_path = os.path.join(log_dir, result_fname)
        if predictor_path is not None:
            self._predictor_ckpt_path = os.path.join(run_config.path, predictor_path)
        else:
            self._predictor_ckpt_path = None
        self._abandon = run_config.abandon
        self._barrier = barrier
        self._queues = queues
        self._pipes = pipes
        self._stop_event = stop_event
        self._models_feature = len(Query.MODELS_LEN) + 4 * run_config.total_models

    def run(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self._result_file = open(self._result_path, "w+")
        self._wr = csv.writer(self._result_file, dialect="excel")

        if self._predictor_ckpt_path is not None:
            logging.info("Scheduler loading predictor")
            self._predictor = MLPregression(self._models_feature)
            self._predictor.load_state_dict(
                torch.load(self._predictor_ckpt_path, map_location="cpu")
            )
            self._predictor.eval()
            logging.info("Scheduler warmpup predictor")
            warmp_up_input = torch.zeros(16, self._models_feature)
            with torch.no_grad():
                for i in range(200):
                    warmp_up_output = self._predictor(warmp_up_input).numpy()
            logging.info("Scheduler end up warm up")
        logging.info("Scheduler Ready")
        self._scheduling_queries = {key: None for key in self._serve_combination}
        self._scheduled_queries = {key: None for key in self._serve_combination}

        result_header = ["query_id", "model_id", "bs", "seq_len", "latency"]
        if self._policy == "Abacus":
            self._schedule_func = self.Abacus_schedule
            # the number of scheduled queries in last round of scheduling, equal with schedule_flag + 1
            self._schedule_flag = -1
            self._predicted_latency = 0
            # search times for the optimal operator group
            self._search_times = 0
            result_header = [
                "query_id",
                "model_id",
                "bs",
                "seq_len",
                "search_times",
                "latency",
            ]
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
        logging.info("Performing scheudling with Policy: {}".format(self._policy))
        if self._policy == "FCFS" or self._policy == "SJF" or self._policy == "EDF":
            while not self._stop_event.is_set() or self.remain_schedule():
                self.pop_queries()
                self._schedule_func(self._abandon)
        elif self._policy == "Abacus":
            while not self._stop_event.is_set():
                self.pop_queries()
                self.Abacus_schedule()
            self.Abacus_reset()

    def Abacus_schedule(self):
        waiting_scheduling, abandoned_scheduling = self.urgent_sort(True)
        total_colocated = len(waiting_scheduling)
        if total_colocated > 0:
            qos_id = waiting_scheduling[0][0]
            qos = waiting_scheduling[0][1] - self._predicted_latency
            qos_query: Query = self._scheduling_queries[qos_id]
            qos_query.set_op_pos(-1)
            if total_colocated == 1:
                if self._abandon:
                    with torch.no_grad():
                        self._predicted_latency = self._predictor(
                            self.get_layer_feature(0, 0, 1, qos_query=qos_query)[1]
                        )[0]
                    self.Abacus_reset()
                    if qos < self._predicted_latency:
                        abandoned_scheduling.append(qos_id)
                        self._predicted_latency = 0
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
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
                            qos_query.id,
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
            elif total_colocated >= 2:
                l_id = waiting_scheduling[1][0]
                l_query: Query = self._scheduling_queries[l_id]
                if total_colocated >= 3:
                    m_id = waiting_scheduling[2][0]
                    m_query: Query = self._scheduling_queries[m_id]
                else:
                    m_id = None
                    m_query = None
                if total_colocated >= 4:
                    r_id = waiting_scheduling[3][0]
                    r_query: Query = self._scheduling_queries[r_id]
                else:
                    r_id = None
                    r_query = None
                searched_query = None
                searched_pos = None
                features = self.get_query_feature(
                    total_colocated, qos_query, l_query, m_query, r_query
                )
                search_times = 1
                with torch.no_grad():
                    latencies = self._predictor(features).numpy()
                if total_colocated == 4:
                    if qos >= latencies[3]:
                        searched_query = 4
                    elif qos >= latencies[2]:
                        searched_query = 3
                    elif qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0
                elif total_colocated == 3:
                    if qos >= latencies[2]:
                        searched_query = 3
                    elif qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0
                elif total_colocated == 2:
                    if qos >= latencies[1]:
                        searched_query = 2
                    elif qos >= latencies[0]:
                        searched_query = 1
                    else:
                        searched_query = 0

                if searched_query == 4:
                    l_query.set_op_pos(-1)
                    m_query.set_op_pos(-1)
                    r_query.set_op_pos(-1)
                elif searched_query == 3:
                    if total_colocated >= 4:
                        # search in r query
                        l_query.set_op_pos(-1)
                        m_query.set_op_pos(-1)
                        start_pos = r_query.end_pos
                        end_pos = r_query.op_len
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
                                qos_query=qos_query,
                                l_query=l_query,
                                m_query=m_query,
                                r_query=r_query,
                            )
                            with torch.no_grad():
                                latencies = self._predictor(features).numpy()
                            for i in range(search_ways):
                                headroom = qos - latencies[i]
                                if headroom < 0:
                                    end_pos = op_poses[i]
                                    if i == 0:
                                        searched_pos = op_poses[i]
                                        break
                                else:
                                    if headroom < self._threashold:
                                        searched_pos = op_poses[i]
                                        self._predicted_latency = latencies[i]
                                        break
                                    else:
                                        start_pos = op_poses[i] + 1
                                if i == search_ways - 1:
                                    searched_pos = op_poses[i]
                            if searched_pos is not None:
                                break
                    else:
                        # only l and m query
                        l_query.set_op_pos(-1)
                        m_query.set_op_pos(-1)
                elif searched_query == 2:
                    l_query.set_op_pos(-1)
                    if total_colocated >= 3:
                        # search in m query
                        start_pos = m_query.end_pos
                        end_pos = m_query.op_len
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
                                qos_query=qos_query,
                                l_query=l_query,
                                m_query=m_query,
                            )
                            with torch.no_grad():
                                latencies = self._predictor(features).numpy()
                            for i in range(search_ways):
                                headroom = qos - latencies[i]
                                if headroom < 0:
                                    end_pos = op_poses[i]
                                    if i == 0:
                                        searched_pos = op_poses[i]
                                        break
                                else:
                                    if headroom < self._threashold:
                                        searched_pos = op_poses[i]
                                        self._predicted_latency = latencies[i]
                                        break
                                    else:
                                        start_pos = op_poses[i] + 1
                                if i == search_ways - 1:
                                    searched_pos = op_poses[i]
                            if searched_pos is not None:
                                break
                elif searched_query == 1:
                    # search in l query
                    start_pos = l_query.end_pos
                    end_pos = l_query.op_len

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
                            qos_query=qos_query,
                            l_query=l_query,
                        )
                        with torch.no_grad():
                            latencies = self._predictor(features).numpy()
                        for i in range(search_ways):
                            headroom = qos - latencies[i]
                            if headroom < 0:
                                end_pos = op_poses[i]
                                if i == 0:
                                    searched_pos = op_poses[i]
                            else:
                                if headroom < self._threashold:
                                    searched_pos = op_poses[i]
                                    self._predicted_latency = latencies[i]
                                    break
                                else:
                                    start_pos = op_poses[i] + 1
                            if i == search_ways - 1:
                                searched_pos = op_poses[-1]
                        if searched_pos is not None:
                            break

                self.Abacus_reset(search_times=search_times)

                if searched_query == 0:
                    if self._abandon:
                        abandoned_scheduling.append(qos_id)
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
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
                    l_query.set_op_pos(searched_pos)
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            1,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            1,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    if l_query.if_processed():
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduled_queries(model_ids=[qos_id, l_id])
                    else:
                        self._scheduled_queries[qos_id] = qos_query
                        self.clean_scheduled_queries(model_ids=qos_id)
                    self._schedule_flag = 1
                elif searched_query == 2:
                    if total_colocated >= 3:
                        m_query.set_op_pos(searched_pos)
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                2,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                2,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._pipes[m_id].send(
                            (
                                m_query.id,
                                m_query.state,
                                2,
                                m_query.start_pos,
                                m_query.end_pos,
                                m_query.batch_size,
                                m_query.seq_len,
                            )
                        )
                        if m_query.if_processed():
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self._scheduled_queries[m_id] = m_query
                            self.clean_scheduled_queries(model_ids=[qos_id, l_id, m_id])
                        else:
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self.clean_scheduled_queries(model_ids=[qos_id, l_id])
                        self._schedule_flag = 2
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                1,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                1,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self.clean_scheduled_queries(model_ids=[qos_id, l_id])
                        self._schedule_flag = 1

                elif searched_query == 3:
                    if total_colocated >= 4:
                        r_query.set_op_pos(searched_pos)
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                3,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                3,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._pipes[m_id].send(
                            (
                                m_query.id,
                                m_query.state,
                                3,
                                m_query.start_pos,
                                m_query.end_pos,
                                m_query.batch_size,
                                m_query.seq_len,
                            )
                        )
                        self._pipes[r_id].send(
                            (
                                r_query.id,
                                r_query.state,
                                3,
                                r_query.start_pos,
                                r_query.end_pos,
                                r_query.batch_size,
                                r_query.seq_len,
                            )
                        )
                        if r_query.if_processed():
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self._scheduled_queries[m_id] = m_query
                            self._scheduled_queries[r_id] = r_query
                            self.clean_scheduled_queries(
                                model_ids=[qos_id, l_id, m_id, r_id]
                            )
                        else:
                            self._scheduled_queries[qos_id] = qos_query
                            self._scheduled_queries[l_id] = l_query
                            self._scheduled_queries[m_id] = m_query
                            self.clean_scheduled_queries(model_ids=[qos_id, l_id, m_id])
                        self._schedule_flag = 3
                    else:
                        self._pipes[qos_id].send(
                            (
                                qos_query.id,
                                qos_query.state,
                                2,
                                qos_query.start_pos,
                                qos_query.end_pos,
                                qos_query.batch_size,
                                qos_query.seq_len,
                            )
                        )
                        self._pipes[l_id].send(
                            (
                                l_query.id,
                                l_query.state,
                                2,
                                l_query.start_pos,
                                l_query.end_pos,
                                l_query.batch_size,
                                l_query.seq_len,
                            )
                        )
                        self._pipes[m_id].send(
                            (
                                m_query.id,
                                m_query.state,
                                2,
                                m_query.start_pos,
                                m_query.end_pos,
                                m_query.batch_size,
                                m_query.seq_len,
                            )
                        )
                        self._scheduled_queries[qos_id] = qos_query
                        self._scheduled_queries[l_id] = l_query
                        self._scheduled_queries[m_id] = m_query
                        self.clean_scheduled_queries(model_ids=[qos_id, l_id, m_id])
                        self._schedule_flag = 2
                elif searched_query == 4:
                    self._pipes[qos_id].send(
                        (
                            qos_query.id,
                            qos_query.state,
                            3,
                            qos_query.start_pos,
                            qos_query.end_pos,
                            qos_query.batch_size,
                            qos_query.seq_len,
                        )
                    )
                    self._pipes[l_id].send(
                        (
                            l_query.id,
                            l_query.state,
                            3,
                            l_query.start_pos,
                            l_query.end_pos,
                            l_query.batch_size,
                            l_query.seq_len,
                        )
                    )
                    self._pipes[m_id].send(
                        (
                            m_query.id,
                            m_query.state,
                            3,
                            m_query.start_pos,
                            m_query.end_pos,
                            m_query.batch_size,
                            m_query.seq_len,
                        )
                    )
                    self._pipes[r_id].send(
                        (
                            r_query.id,
                            r_query.state,
                            3,
                            r_query.start_pos,
                            r_query.end_pos,
                            r_query.batch_size,
                            r_query.seq_len,
                        )
                    )
                    self._scheduled_queries[qos_id] = qos_query
                    self._scheduled_queries[l_id] = l_query
                    self._scheduled_queries[m_id] = m_query
                    self._scheduled_queries[r_id] = r_query
                    self.clean_scheduled_queries(model_ids=[qos_id, l_id, m_id, r_id])
                    self._schedule_flag = 3
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.Abacus_reset()

        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [query.id, query.model_id, query.batch_size, query.seq_len, 0, -1]
                )
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
                            query.id,
                            query.model_id,
                            query.batch_size,
                            query.seq_len,
                            self._search_times,
                            query.latency_ms(),
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
                    query.id,
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
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.latency_ms(),
                    ]
                )
            )
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [query.id, query.model_id, query.batch_size, query.seq_len, -1]
                )
            )
            self.clean_scheduled_queries(model_ids=model_id)
        self._result_file.flush()

    def SJF_schedule(self, abandon=False):
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
                    with torch.no_grad():
                        latency = self._predictor(
                            self.get_layer_feature(
                                start_pos=0, end_pos=0, search_ways=1, qos_query=query
                            )[1]
                        )[0]
                    waiting_scheduling[model_id] = latency
                else:
                    abandoned_scheduling.append(model_id)
            else:
                query.set_op_pos(-1)
                with torch.no_grad():
                    latency = self._predictor(
                        self.get_layer_feature(
                            start_pos=0, end_pos=0, search_ways=1, qos_query=query
                        )[1]
                    )[0]
                waiting_scheduling[model_id] = latency
        waiting_scheduling = sorted(waiting_scheduling.items(), key=lambda x: x[1])
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                (
                    query.id,
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
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.latency_ms(),
                    ]
                )
            )
            self.clean_scheduled_queries(model_ids=model_id)
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [query.id, query.model_id, query.batch_size, query.seq_len, -1]
                )
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
        for model_id, _ in waiting_scheduling:
            query: Query = self._scheduling_queries[model_id]
            self._pipes[model_id].send(
                (
                    query.id,
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
                    [
                        query.id,
                        query.model_id,
                        query.batch_size,
                        query.seq_len,
                        query.latency_ms(),
                    ]
                )
            )
            self.clean_scheduled_queries(model_ids=model_id)
        for model_id in abandoned_scheduling:
            query = self._scheduling_queries[model_id]
            self._wr.writerow(
                np.array(
                    [query.id, query.model_id, query.batch_size, query.seq_len, -1]
                )
            )
            self.clean_scheduled_queries(model_ids=model_id)
        self._result_file.flush()

    def pop_queries(self):
        for model_id in self._serve_combination:
            if (self._scheduling_queries[model_id] is None) and (
                not self._queues[model_id].empty()
            ):
                self._scheduling_queries[model_id] = self._queues[model_id].get()

    def remain_schedule(self):
        for key in self._scheduling_queries:
            if self._scheduling_queries[key] is not None:
                return True
        return False

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
        if r_query is not None:
            input_feature[0:search_ways, l_query.model_id] += 1
            input_feature[0:search_ways, 11] = l_query.start_pos
            input_feature[0:search_ways, 12] = l_query.end_pos
            input_feature[0:search_ways, 13] = l_query.batch_size
            input_feature[0:search_ways, 14] = l_query.seq_len
            input_feature[0:search_ways, m_query.model_id] += 1
            input_feature[0:search_ways, 15] = m_query.start_pos
            input_feature[0:search_ways, 16] = m_query.end_pos
            input_feature[0:search_ways, 17] = m_query.batch_size
            input_feature[0:search_ways, 18] = m_query.seq_len
            input_feature[0:search_ways, r_query.model_id] += 1
            input_feature[0:search_ways, 19] = r_query.start_pos
            # input_feature[0:search_ways, 20] = r_query.end_pos
            input_feature[0:search_ways, 21] = r_query.batch_size
            input_feature[0:search_ways, 22] = r_query.seq_len
            for i in range(search_ways):
                start_pos += step
                input_feature[i, 20] = start_pos
                poses.append(start_pos)
        elif m_query is not None:
            input_feature[0:search_ways, l_query.model_id] += 1
            input_feature[0:search_ways, 11] = l_query.start_pos
            input_feature[0:search_ways, 12] = l_query.end_pos
            input_feature[0:search_ways, 13] = l_query.batch_size
            input_feature[0:search_ways, 14] = l_query.seq_len
            input_feature[0:search_ways, m_query.model_id] += 1
            input_feature[0:search_ways, 15] = m_query.start_pos
            # input_feature[0:search_ways, 16] = m_query.end_pos
            input_feature[0:search_ways, 17] = m_query.batch_size
            input_feature[0:search_ways, 18] = m_query.seq_len
            for i in range(search_ways):
                start_pos += step
                input_feature[i, 16] = start_pos
                poses.append(start_pos)
        if l_query is not None:
            input_feature[0:search_ways, l_query.model_id] += 1
            input_feature[0:search_ways, 11] = l_query.start_pos
            # input_feature[0:search_ways, 12] = l_query.end_pos
            input_feature[0:search_ways, 13] = l_query.batch_size
            input_feature[0:search_ways, 14] = l_query.seq_len
            for i in range(search_ways):
                start_pos += step
                input_feature[i, 12] = start_pos
                poses.append(start_pos)
        logging.debug("input feature: {}".format(input_feature))
        return poses, input_feature