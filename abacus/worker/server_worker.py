#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from torch.multiprocessing import Process
import torch
import torch.nn as nn
import os
import numpy as np
from torch.cuda.streams import Stream

from abacus.utils import timestamp
from abacus.worker.worker import AbacusWorker

class ServerWorker(AbacusWorker):
    def __ini__(
        self,
        args,
        model_name: str,
        supported_batchsize,
        supported_seqlen,
        recv_pipe,
        barrier,
    ):
        super().__init__(
            args, model_name, supported_batchsize, supported_seqlen, recv_pipe
        )
        self._barrier = barrier

    def run(self) -> None:
        timestamp("Server Woker for {}".format(self._model_name), "Starting")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
        torch.device("cuda")
        torch.backends.cudnn.enabled = True
        if self._model_name == "inception_v3":
            self._inputs = {
                k: torch.rand(k, 3, 299, 299).half().cuda()
                for k in self._supported_batchsize
            }
        else:
            self._inputs = {
                k: torch.rand(k, 3, 224, 224).half().cuda()
                for k in self._supported_batchsize
            }
        self._model = self._model_func().half().cuda().eval()