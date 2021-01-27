#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import torch
import numpy as np
from abacus.option import parse_options,RunConfig

import datetime

class TestWorker:
    def __init__(
        self,
        args:RunConfig,
        model_name: str,
    ):
        self._model_name = model_name
        self._model_func = args.models_list[model_name]
        self._supported_batchsize = args.supported_batchsize
        self._supported_seqlen = args.supported_seqlen

    def run(self) -> None:

        torch.device("cuda")
        torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
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
        self._event = torch.cuda.Event(enable_timing=False)
        for k in self._supported_batchsize:
            self._model(self._inputs[k])

        torch.cuda.synchronize()

        while True:
            bs = 1
            seq_len = 16
            cnt = 100
            while cnt!=0:
                cnt -=1
                start = datetime.datetime.now()
                self._model(self._inputs[bs])
                torch.cuda.synchronize()
                end = datetime.datetime.now()
                print((end-start).microseconds/1000)

if __name__ == "__main__":
    run_config = parse_options()
    test_worker = TestWorker(run_config, "resnet152")
    test_worker.run()