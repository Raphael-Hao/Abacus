#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /client.py
# \brief: client for Abacus
# Author: raphael hao

import logging
import random

import torch.multiprocessing as mp
from torch.multiprocessing import Process

from abacus.utils import Query
from abacus.option import RunConfig
from abacus.loadbalancer.abacus import AbacusLoadBalancer
from abacus.loadbalancer.abacus import AbacusLoadBalancer

import abacus.service_pb2 as service_pb2
import abacus.service_pb2_grpc as service_pb2_grpc


class Cluster:
    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        self._queues = {}
        self._qos_target = run_config.qos_target
        random.seed(0)

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
        logging.info("Cluster Initializing")
        for model_id in self._run_config.serve_combination:
            self._queues[model_id] = mp.Queue()


class AbacusCluster(Cluster):
    def start():
        raise NotImplementedError


class ClockCluster:
    def start():
        raise NotImplementedError
