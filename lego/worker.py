#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from torch.multiprocessing import Process
import torch
from torch.cuda.streams import Stream
from lego.network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lego.utils import timestamp

model_list = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}


def scheduling():
    raise NotImplementedError


class ModelProc(Process):
    def __init__(self, model_name: str, recv_pipe, barrier):
        super().__init__()
        self._model_name = model_name
        self._model_func = model_list[model_name]
        self._pipe = recv_pipe
        self._barrier = barrier

    def run(self) -> None:
        timestamp("worker", "starting")
        cuda = torch.device("cuda")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self._inputs = [torch.rand(2, 3, 224, 224).half().cuda() for _ in range(100)]
        self._model = self._model_func().half().cuda().eval()
        self._stream = Stream(0)
        self._event = torch.cuda.Event(enable_timing=False)
        timestamp("worker", "warming")
        with torch.cuda.stream(self._stream):
            for i in range(10):
                self._model(self._inputs[i])
        torch.cuda.synchronize()
        self._barrier.wait()
        i = 0
        while True:
            model_name, start, end = self._pipe.recv()
            # with torch.cuda.stream(self.stream_):

            if self._model_name == model_name:
                self._model(self._inputs[i])
                torch.cuda.synchronize()
                # self.event_.record(self.stream_)
                # self.event_.synchronize()
                self._barrier.wait()
                i += 1
            elif model_name == "terminate":
                break
            else:
                break
