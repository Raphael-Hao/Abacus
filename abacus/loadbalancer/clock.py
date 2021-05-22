#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /clock.py
# \brief:
# Author: raphael hao

from abacus.option import RunConfig
from abacus.loadbalancer.base import LoadBalancer

import abacus.service_pb2 as service_pb2
import abacus.service_pb2_grpc as service_pb2_grpc


class ClockLoadBalancer(LoadBalancer):
    def __init__(self, run_config: RunConfig, query_q, node_q, qos_target) -> None:
        super().__init__(run_config=run_config, query_q=query_q, qos_target=qos_target)