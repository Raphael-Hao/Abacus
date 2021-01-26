#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from torch.multiprocessing import Process
class AbacusWorker(Process):
    def __init__(
        self,
        args,
        model_name: str,
        supported_batchsize,
        supported_seqlen,
        recv_pipe,
    ):
        super().__init__()
        self._model_name = model_name
        self._model_func = args.models_list[model_name]
        self._pipe = recv_pipe
        self._supported_batchsize = supported_batchsize
        self._supported_seqlen = supported_seqlen
