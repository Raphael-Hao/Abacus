#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import logging, sys

import torch

from abacus.profiler import profile
from abacus.server import AbacusServer
from abacus.trainer import train_predictor
from abacus.option import parse_options
from abacus.background import background


if __name__ == "__main__":
    run_config = parse_options()
    torch.set_printoptions(linewidth=200)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if run_config.debug is True else logging.INFO,
    )
    print(run_config)
    if run_config.task == "profile":
        profile(run_config=run_config)
    elif run_config.task == "serve":
        abacus_server = AbacusServer(run_config=run_config)
        abacus_server.start_up()
        abacus_server.start_test()
        abacus_server.stop_test()
    elif run_config.task == "train":
        train_predictor(run_config)
    elif run_config.task == "background":
        background(args=run_config)
