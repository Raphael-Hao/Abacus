#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import logging, sys
import grpc
from concurrent import futures
import torch
import torch.multiprocessing as mp

from abacus.profiler import profile
from abacus.server import AbacusServer, ClockServer
from abacus.cluster import Cluster
from abacus.trainer import train_predictor
from abacus.option import parse_options
from abacus.background import background
import abacus.service_pb2 as service_pb2
import abacus.service_pb2_grpc as service_pb2_grpc

if __name__ == "__main__":
    run_config = parse_options()
    # mp.set_start_method("spawn")
    torch.set_printoptions(linewidth=200)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if run_config.debug is True else logging.INFO,
    )
    print(run_config)
    if run_config.task == "profile":
        profile(run_config=run_config)
    elif run_config.task == "server":
        if run_config.platform == "single":
            server = AbacusServer(run_config=run_config)
            server.start_test()
            server.stop_test()
        elif run_config.platform == "cluster":
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
            if run_config.policy == "Abacus":
                ServerClass = AbacusServer
            elif run_config.policy == "Clock":
                ServerClass = ClockServer
            else:
                raise NotImplementedError
            service_pb2_grpc.add_DNNServerServicer_to_server(
                ServerClass(run_config=run_config), server
            )
            server.add_insecure_port("[::]:50051")
            server.start()
            server.wait_for_termination()
        else:
            raise NotImplementedError
    elif run_config.task == "scheduler":
        abacus_client = Cluster(run_config=run_config)
        abacus_client.start_load_balancer()
        abacus_client.start_long_term_test()
    elif run_config.task == "train":
        train_predictor(run_config)
    elif run_config.task == "background":
        background(args=run_config)
