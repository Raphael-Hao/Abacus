#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from torch.multiprocessing import Process
import torch
import torch.nn as nn
import os
import numpy as np
from torch.cuda.streams import Stream

from lego.network.resnet_splited import resnet50, resnet101, resnet152
from lego.network.inception_splited import inception_v3
from lego.network.vgg_splited import vgg16, vgg19
from lego.network.bert import BertModel

# from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lego.utils import timestamp

model_list = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "inception_v3": inception_v3,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "bert": BertModel,
}

model_len = {
    "resnet50": 18,
    "resnet101": 35,
    "resnet152": 52,
    "inception_v3": 14,
    "vgg16": 19,
    "vgg19": 22,
    "bert": 12,
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
        if self._model_name == "inception_v3":
            self._inputs = {
                k: torch.rand(k, 3, 299, 299).half().cuda()
                for k in self._supported_batchsize
            }
        elif self._model_name == "bert":
            self._inputs = {
                k: torch.LongTensor(np.zeros((k, 10))).cuda()
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
                self._model(self._inputs[k])
        torch.cuda.synchronize()
        self._barrier.wait()

        while True:
            model_name, action, start, end, bs = self._pipe.recv()

            if action == "prepare":
                if self._model_name == "bert":
                    self._inter_input = self._model.prepare(self._inputs[bs], start)
                else:
                    submodel = nn.Sequential(*self._submodules[:start])
                    self._inter_input = submodel(self._inputs[bs])
                    self._submodel = nn.Sequential(*self._submodules[start:end])
                torch.cuda.synchronize()
                self._barrier.wait()
            elif action == "forward":
                if self._model_name == "bert":
                    self._model.run(self._inter_input, start, end)
                else:
                    self._submodel(self._inter_input)
                torch.cuda.synchronize()
                self._barrier.wait()
            elif action == "terminate":
                break
            else:
                raise NotImplementedError
