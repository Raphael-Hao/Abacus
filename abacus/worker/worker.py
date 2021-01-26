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


class ProfilerWorker(AbacusWorker):
    def __init__(
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
        timestamp("worker", "starting")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
        torch.device("cuda")
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        if self._model_name == "inception_v3":
            self._inputs = {
                k: torch.rand(k, 3, 299, 299).half().cuda()
                for k in self._supported_batchsize
            }
        elif self._model_name == "bert":
            self._inputs = {
                k: {
                    seqlen: torch.LongTensor(np.random.rand(k, seqlen, 768)).half().cuda()
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
        timestamp("worker", "warming")
        with torch.cuda.stream(self._stream):
            for k in self._supported_batchsize:
                if self._model_name == "bert":
                    for seqlen in self._supported_seqlen:
                        self._model.run(self._inputs[k][seqlen], self._masks[k][seqlen], 0, 12)
                else:
                    self._model(self._inputs[k])

        torch.cuda.synchronize()
        self._barrier.wait()

        while True:
            model_name, action, start, end, bs, seq_len = self._pipe.recv()

            if action == "prepare":
                if self._model_name == "bert":
                    self._inter_input = self._model.run(
                        self._inputs[bs][seq_len], self._masks[bs][seq_len], 0, start
                    )
                else:
                    submodel = nn.Sequential(*self._submodules[:start])
                    self._inter_input = submodel(self._inputs[bs])
                    self._submodel = nn.Sequential(*self._submodules[start:end])
                torch.cuda.synchronize()
                self._barrier.wait()
            elif action == "forward":
                if self._model_name == "bert":
                    self._model.run(self._inter_input, self._masks[bs][seq_len], start, end)
                else:
                    self._submodel(self._inter_input)
                torch.cuda.synchronize()
                self._barrier.wait()
            elif action == "terminate":
                break
            else:
                raise NotImplementedError


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
        return super().run()