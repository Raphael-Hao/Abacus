#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /abacus.py
# \brief:
# Author: raphael hao

from abacus.loadbalancer.base import LoadBalancer
import abacus.service_pb2 as service_pb2
import abacus.service_pb2_grpc as service_pb2_grpc
class AbacusLoadBalancer(LoadBalancer):
    def __init__(self, queues, qos_target) -> None:
        super().__init__()