#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import os
import logging
import numpy as np
import random

from torch.multiprocessing import Process
import torch
import torch.nn as nn
from torch.cuda.streams import Stream

from abacus.worker.worker import AbacusWorker


class BackgroundWorker(AbacusWorker):
    def __init__(
        self,
        args,
        model_name: str,
        supported_batchsize,
        supported_seqlen,
        recv_pipe,
        barrier,
        barrier2,
        isBackground,
        shared_flag,
    ):
        super().__init__(
            args, model_name, supported_batchsize, supported_seqlen, recv_pipe
        )
        self._barrier = barrier
        self._barrier2 = barrier2
        self.isBackground = isBackground
        self.shared_flag = shared_flag

    def run(self) -> None:
        logging.info("worker", "starting")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
        torch.device("cuda")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if self._model_name == "inception_v3":
            self._inputs = {
                k: torch.rand(k, 3, 299, 299).half().cuda()
                for k in self._supported_batchsize
            }
        elif self._model_name == "bert":
            self._inputs = {
                k: {
                    seqlen: torch.LongTensor(np.random.rand(k, seqlen, 768))
                    .half()
                    .cuda()
                    for seqlen in self._supported_seqlen
                }
                for k in self._supported_batchsize
            }
            self._masks = {
                k: {
                    seqlen: torch.LongTensor(np.zeros((k, 1, 1, seqlen))).half().cuda()
                    for seqlen in self._supported_seqlen
                }
                for k in self._supported_batchsize
            }
        else:
            self._inputs = {
                k: torch.rand(k, 3, 224, 224).half().cuda()
                for k in self._supported_batchsize
            }
        self._model = self._model_func().half().cuda().eval()
        if self._model_name != "bert":
            self._submodules = self._model.get_submodules()
            self._total_module = len(self._submodules)
        self._stream = Stream(0)
        self._event = torch.cuda.Event(enable_timing=False)
        logging.info("worker", "warming")
        with torch.cuda.stream(self._stream):
            for k in self._supported_batchsize:
                if self._model_name == "bert":
                    for seqlen in self._supported_seqlen:
                        self._model.run(
                            self._inputs[k][seqlen], self._masks[k][seqlen], 0, 12
                        )
                else:
                    self._model(self._inputs[k])

        torch.cuda.synchronize()
        self._barrier.wait()
        with torch.no_grad():
            if self.isBackground:
                if self._model_name == "bert":
                    while self.shared_flag.value == 1:
                        bs = random.choice([1, 16])
                        seq_len = random.choice([8, 64])
                        self._model.run(
                            self._inputs[bs][seq_len], self._masks[bs][seq_len], 0, 12
                        )
                    print("background terminated!")
                    return
                else:
                    while self.shared_flag.value == 1:
                        bs = random.choice([1, 16])
                        self._model(self._inputs[bs])
                    print("background terminated!")
                    return

            while True:
                model_name, action, bs, seq_len = self._pipe.recv()

                if action == "forward":
                    if self._model_name == "bert":
                        self._model.run(
                            self._inputs[bs][seq_len], self._masks[bs][seq_len], 0, 12
                        )
                    else:
                        self._model(self._inputs[bs])
                    torch.cuda.synchronize()
                    self._barrier2.wait()
                elif action == "terminate":
                    return
                else:
                    raise NotImplementedError
