#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /base.py
# \brief:
# Author: raphael hao

import torch.multiprocessing as mp
from torch.multiprocessing import Process


class LoadBalancer(Process):
    def __init__(self, queues, qos_target) -> None:
        super().__init__()

    def start_up():
        raise NotImplementedError