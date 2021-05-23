#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /abacus.py
# \brief:
# Author: raphael hao
import grpc
import random

from abacus.option import RunConfig
from abacus.utils import Query
from abacus.loadbalancer.base import LoadBalancer

import abacus.service_pb2 as service_pb2
import abacus.service_pb2_grpc as service_pb2_grpc


class AbacusLoadBalancer(LoadBalancer):
    def __init__(self, run_config: RunConfig, query_q, qos_target) -> None:
        super().__init__(run_config=run_config, query_q=query_q, qos_target=qos_target)

    def run(self) -> None:
        self._channel_dict = {}
        self._stub_dict = {}
        self._node_list = []
        for node_id in range(self._run_config.node_cnt):
            ip = self._run_config.ip_dict[node_id]
            channel = grpc.insecure_channel("{}:50051".format(ip))
            stub = service_pb2_grpc.DNNServerStub(channel=channel)
            self._channel_dict[node_id] = channel
            self._stub_dict[node_id] = stub
            self._node_list.append(node_id)
        while True:
            if not self._query_q.empty():
                query: Query = self._query_q.get()
                node_id = random.choice(self._node_list)
                result = self._stub_dict[node_id].Inference(
                    service_pb2.Query(
                        id=query.id,
                        model_id=query.model_id,
                        bs=query.batch_size,
                        seq_len=query.seq_len,
                        start_stamp=query.start_stamp,
                        qos_target=query.qos_targt,
                    )
                )
