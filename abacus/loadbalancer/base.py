#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /base.py
# \brief:
# Author: raphael hao

import torch.multiprocessing as mp
from torch.multiprocessing import Process
from abacus.option import RunConfig


class LoadBalancer(Process):
    def __init__(self, run_config: RunConfig, query_q, qos_target) -> None:
        self._run_config = run_config

    def start_up():
        raise NotImplementedError