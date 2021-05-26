#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /client.py
# \brief: client for Abacus
# Author: raphael hao

import logging
import random
import time

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from abacus.utils import Query
from abacus.option import RunConfig
from abacus.loadbalancer.abacus import AbacusLoadBalancer
from abacus.loadbalancer.clock import ClockLoadBalancer


class Cluster:
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._queues = {}
        self._load_balancers = {}
        self._qos_target = run_config.qos_target
        random.seed(0)

    def start_load_balancer(self):
        logging.info("Cluster Initializing")
        logging.info("Model queues Initializing")
        for model_id in self._run_config.serve_combination:
            self._queues[model_id] = mp.Queue()
        logging.info("Loadbalancer Initializing")
        if self._run_config.policy == "Abacus":
            for model_id in self._run_config.serve_combination:
                load_balancer = AbacusLoadBalancer(
                    run_config=self._run_config,
                    model_id=model_id,
                    query_q=self._queues[model_id],
                    qos_target=self._run_config.qos_target,
                )
                load_balancer.start()
                self._load_balancers[model_id] = load_balancer
        elif self._run_config.policy == "Clock":
            self._node_q = mp.Queue()
            for node_id in range(self._run_config.node_cnt):
                self._node_q.put(node_id)
            for model_id in self._run_config.serve_combination:
                self._load_balancers[model_id] = ClockLoadBalancer(
                    run_config=self._run_config,
                    model_id=model_id,
                    query_q=self._queues[model_id],
                    node_q=self._node_q,
                    qos_target=self._run_config.qos_target,
                )
                self._load_balancers[model_id].start()
        else:
            raise NotImplementedError

    def prepare_test_queries(self, total_queries=1000, average_duration=20):
        logging.info("Preparing test queries")
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

    def start_long_term_test(self):
        id = 0
        load_id = 0
        total_loads = len(self._run_config.loads)
        average_duration = self._run_config.loads[load_id]
        start_stamp = time.time()
        while True:
            if (time.time() - start_stamp) >= self._run_config.load_change_dura:
                load_id += 1
                if load_id == total_loads:
                    break
                average_duration = self._run_config.loads[load_id]
            id += 1
            model_id = random.choice(self._run_config.serve_combination)
            sleep_duration = random.expovariate(average_duration)
            bs = random.choice(self._run_config.supported_batchsize)
            seq_len = (
                random.choice(self._run_config.supported_seqlen) if model_id == 6 else 0
            )
            self.send_query(id=id, model_id=model_id, batch_size=bs, seq_len=seq_len)
            time.sleep(sleep_duration)

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