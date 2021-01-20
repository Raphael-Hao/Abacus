#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from torch.multiprocessing import Process
import torch
import torch.nn as nn
import os
from torch.cuda.streams import Stream

from lego.network.resnet_splited import resnet50, resnet101, resnet152

# from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lego.utils import timestamp

model_list = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

model_len = {
    "resnet50": 18,
    "resnet101": 35,
    "resnet152": 52,
}


def scheduling():
    raise NotImplementedError


class ModelProc(Process):
    def __init__(self, model_name: str, supported_batchsize, recv_pipe, barrier):
        super().__init__()
        self._model_name = model_name
        self._model_func = model_list[model_name]
        self._pipe = recv_pipe
        self._barrier = barrier
        self._supported_batchsize = supported_batchsize

    def run(self) -> None:
        timestamp("worker", "starting")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "100"
        torch.device("cuda")
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        self._inputs = {
            k: torch.rand(k, 3, 224, 224).half().cuda()
            for k in self._supported_batchsize
        }
        self._model = self._model_func().half().cuda()
        self._submodules = self._model.get_submodules()
        self._total_module = len(self._submodules)
        self._stream = Stream(0)
        self._event = torch.cuda.Event(enable_timing=False)
        timestamp("worker", "warming")
        with torch.cuda.stream(self._stream):
            for k in self._supported_batchsize:
                self._model(self._inputs[k])
        torch.cuda.synchronize()
        self._barrier.wait()
        i = 0

        while True:
            model_name, action, start, end, bs = self._pipe.recv()

            if action == "prepare":
                submodel = nn.Sequential(*self._submodules[:start])
                self._inter_input = submodel(self._inputs[bs])
                self._submodel = nn.Sequential(*self._submodules[start:end])
                self._barrier.wait()
            elif action == "forward":
                self._submodel(self._inter_input)
                torch.cuda.synchronize()
                # self.event_.record(self.stream_)
                # self.event_.synchronize()
                self._barrier.wait()
            elif action == "terminate":
                break
            else:
                raise NotImplementedError
