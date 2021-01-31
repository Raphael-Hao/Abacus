#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from torch.multiprocessing import Process
from abacus.option import RunConfig
class AbacusWorker(Process):
    def __init__(
        self,
        run_config:RunConfig,
        model_name: str,
        supported_batchsize,
        supported_seqlen,
        recv_pipe,
    ):
        super().__init__()
        self._model_name = model_name
        self._model_func = run_config.models_list[model_name]
        self._pipe = recv_pipe
        self._supported_batchsize = supported_batchsize
        self._supported_seqlen = supported_seqlen
        self._device = run_config.device
