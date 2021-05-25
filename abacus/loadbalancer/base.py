#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /base.py
# \brief:
# Author: raphael hao

from torch.multiprocessing import Process
from abacus.option import RunConfig


class LoadBalancer(Process):
    def __init__(self, run_config: RunConfig,model_id, query_q, qos_target) -> None:
        super().__init__()
        self._run_config = run_config
        self._model_id = model_id
        self._query_q = query_q
        self._qos_tgt = qos_target