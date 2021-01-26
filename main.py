#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import torch.multiprocessing as mp
from abacus.profiler import profile
from abacus.server import AbacusServer
from abacus.trainer import train_predictor
from abacus.option import parse_options

if __name__ == "__main__":
    args = parse_options()
    print(args)

    if args.task == "profile":
        profile(args=args)
    elif args.task == "serve":
        abacus_server = AbacusServer()
        abacus_server.send_query()
    elif args.task == "train":
        train_predictor(args)