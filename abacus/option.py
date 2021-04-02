#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import argparse
import sys
import os
from abacus.network.resnet_splited import resnet50, resnet101, resnet152
from abacus.network.inception_splited import inception_v3
from abacus.network.vgg_splited import vgg16, vgg19
from abacus.network.bert import BertModel


class RunConfig:
    def __init__(self, args) -> None:
        self.task = args.task
        # general configurations
        self.total_models = args.model_num
        self.device = 0
        self.mps_devices = {
            1: [
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/2/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/2/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/2/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/2/0",
            ],
            2: [
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/3/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/3/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/5/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/5/0",
            ],
            4: [
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/7/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/8/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/9/0",
                "MIG-GPU-95be3bb0-41c3-8f7b-47af-20c3799bcf22/10/0",
            ],
        }
        self.mps_pipe_dirs = {
            1: [
                "/tmp/nvidia-mps-0",
                "/tmp/nvidia-mps-0",
                "/tmp/nvidia-mps-0",
                "/tmp/nvidia-mps-0",
            ],
            2: [
                "/tmp/nvidia-mps-0",
                "/tmp/nvidia-mps-0",
                "/tmp/nvidia-mps-1",
                "/tmp/nvidia-mps-1",
            ],
            4: [
                "/tmp/nvidia-mps-0",
                "/tmp/nvidia-mps-1",
                "/tmp/nvidia-mps-2",
                "/tmp/nvidia-mps-3",
            ],
        }
        self.mps_log_dirs = {
            1: [
                "/tmp/nvidia-log-0",
                "/tmp/nvidia-log-0",
                "/tmp/nvidia-log-0",
                "/tmp/nvidia-log-0",
            ],
            2: [
                "/tmp/nvidia-log-0",
                "/tmp/nvidia-log-0",
                "/tmp/nvidia-log-1",
                "/tmp/nvidia-log-1",
            ],
            4: [
                "/tmp/nvidia-log-0",
                "/tmp/nvidia-log-1",
                "/tmp/nvidia-log-2",
                "/tmp/nvidia-log-3",
            ],
        }
        # self.path = "/state/partition/whcui/repository/project/Abacus"
        self.path = "/home/whcui/project/Abacus"
        self.data_path = os.path.join(self.path, "data")

        self.mig = args.mig
        os.makedirs(self.data_path, exist_ok=True)

        self.models_name = [
            "resnet50",  # 0
            "resnet101",  # 1
            "resnet152",  # 2
            "inception_v3",  # 3
            "vgg16",  # 4
            "vgg19",  # 5
            "bert",  # 6
        ]

        self.models_id = {
            "resnet50": 0,
            "resnet101": 1,
            "resnet152": 2,
            "inception_v3": 3,
            "vgg16": 4,
            "vgg19": 5,
            "bert": 6,
        }

        self.models_len = {
            "resnet50": 18,
            "resnet101": 35,
            "resnet152": 52,
            "inception_v3": 14,
            "vgg16": 19,
            "vgg19": 22,
            "bert": 12,
        }

        # self.supported_batchsize = [
        #     1,
        #     2,
        #     4,
        #     8,
        #     16,
        #     32,
        # ]
        # mig batch size
        self.supported_batchsize = [
            # 1,
            # 2,
            4,
            8,
            16,
            32,
        ]
        self.supported_seqlen = [8, 16, 32, 64]

        if self.task != "train":
            self.models_list = {
                "resnet50": resnet50,
                "resnet101": resnet101,
                "resnet152": resnet152,
                "inception_v3": inception_v3,
                "vgg16": vgg16,
                "vgg19": vgg19,
                "bert": BertModel,
            }

        if self.task == "serve":
            """
            [server configuration]
            """
            self.serve_combination = tuple(args.combination)
            self.policy = args.policy
            # self.policy = "SJF"
            # self.policy = "FCFS"
            # self.policy = "EDF"
            self.threshold = args.threshold
            self.qos_target = args.qos
            self.search_ways = args.ways
            self.total_queries = 1000
            self.average_duration = args.load
            self.abandon = True

        elif self.task == "profile":
            """
            [profiled configurations]
            """
            self.total_test = args.test  # 2in7
            # self.total_test = 50  # 3in4
            # self.total_test = 1
            self.test_loop = 100
            if self.total_models == 4:
                self.profiling_combinations = [(1, 2, 5, 6)]
            elif self.total_models == 3:
                self.profiling_combinations = [
                    # 1, 2, 5, 6
                    (1, 2, 5),
                    (1, 2, 6),
                    (1, 5, 6),
                    (2, 5, 6),
                ]
            elif self.total_models == 2:
                if self.mig == 0:
                    self.profiling_combinations = [
                        (0, 1),
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (0, 6),
                        (1, 2),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (1, 6),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                        (2, 6),
                        (3, 4),
                        (3, 5),
                        (3, 6),
                        (4, 5),
                        (4, 6),
                        (5, 6),
                    ]
                elif self.mig == 2:
                    self.profiling_combinations = [
                        (1, 2),
                        (1, 5),
                        (1, 6),
                        (2, 5),
                        (2, 6),
                        (5, 6),
                    ]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        elif args.task == "train":
            """
            [prediction model configurations]
            """
            if self.total_models == 4:
                self.training_combinations = [1, 2, 5, 6]
            elif self.total_models == 3:
                self.training_combinations = [
                    #   1, 2, 5, 6
                    (1, 2, 5),
                    (1, 2, 6),
                    (1, 5, 6),
                    (2, 5, 6),
                ]
            elif self.total_models == 2:
                if self.mig == 0:
                    self.training_combinations = [
                        (0, 1),
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (0, 6),
                        (1, 2),
                        (1, 3),
                        (1, 4),
                        (1, 5),
                        (1, 6),
                        (2, 3),
                        (2, 4),
                        (2, 5),
                        (2, 6),
                        (3, 4),
                        (3, 5),
                        (3, 6),
                        (4, 5),
                        (4, 6),
                        (5, 6),
                    ]
                elif self.mig == 2:
                    self.training_combinations = [
                        (1, 2),
                        (1, 5),
                        (1, 6),
                        (2, 5),
                        (2, 6),
                        (5, 6),
                    ]
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            self.mode = args.mode
            if self.mode == "single":
                self.model_combination = args.model_comb
            self.modeling = args.modeling
            if self.mode == "all" and self.modeling == "mlp":
                self.profile_predictor = True
            else:
                self.profile_predictor = False

            self.hyper_params = {
                "all": [0.001, 250],
                # "all": [0.002, 280],
                "resnet101_inception_v3": [0.001, 100],
                "bert_bert": [0.001, 100],
                "vgg19_bert": [0.001, 100],
                "resnet50_resnet152": [0.001, 100],
                "resnet101_bert": [0.001, 100],
                "resnet152_vgg19": [0.001, 100],
                "resnet50_inception_v3": [0.001, 100],
                "resnet101_resnet152": [0.001, 100],
                "inception_v3_inception_v3": [0.001, 100],
                "vgg19_vgg19": [0.001, 150],
                "vgg16_vgg16": [0.001, 150],
                "resnet101_vgg19": [0.001, 100],
                "inception_v3_bert": [0.001, 100],
                "resnet152_resnet152": [0.0001, 200],
                "resnet50_vgg16": [0.001, 100],
                "resnet101_resnet101": [0.001, 100],
                "resnet50_resnet50": [0.001, 100],
                "resnet152_bert": [0.001, 100],
                "vgg16_vgg19": [0.001, 200],
                "resnet101_vgg16": [0.001, 100],
                "resnet50_vgg19": [0.001, 100],
                "resnet152_inception_v3": [0.001, 100],
                "inception_v3_vgg19": [0.001, 100],
                "resnet50_resnet101": [0.001, 100],
                "vgg16_bert": [0.001, 100],
                "resnet50_bert": [0.001, 100],
                "inception_v3_vgg16": [0.001, 100],
                "resnet152_vgg16": [0.001, 100],
            }
        elif args.task == "background":
            self.background_combinations = [
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
            ]
            self.total_test = 1000
        else:
            print("Not supported task, supported: server, profile, train")
            raise NotImplementedError


def parse_options():
    parser = argparse.ArgumentParser(description="Abacus")

    parser.add_argument(
        "--task",
        type=str,
        default="profile",
        required=True,
        choices=["profile", "train", "serve", "background"],
    )

    parser.add_argument(
        "--model_num",
        type=int,
        default="2",
        required=True,
        choices=[2, 3, 4],
    )

    parser.add_argument("--mig", type=int, default=0, choices=[0, 1, 2, 4])

    """[summary]
    online serve
    """
    parser.add_argument("--comb", type=int, required="serve" in sys.argv, nargs="+")
    parser.add_argument(
        "--policy",
        type=str,
        default="Abacus",
        required="serve" in sys.argv,
        choices=["Abacus", "SJF", "FCFS"],
    )
    parser.add_argument("--load", type=int, required="serve" in sys.argv, default=50)
    parser.add_argument("--qos", type=int, required="serve" in sys.argv, default=60)
    parser.add_argument("--thld", type=int, required="serve" in sys.argv, default=60)
    parser.add_argument("--ways", type=int, required="serve" in sys.argv, default=60)

    ## profiling
    parser.add_argument("--test", type=int, required="profile" in sys.argv, default=200)

    ## training
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        required="train" in sys.argv,
        choices=["all", "onebyone", "single"],
    )
    parser.add_argument(
        "--modeling",
        type=str,
        default="mlp",
        required="train" in sys.argv,
        choices=["mlp", "svm", "lr"],
    )
    parser.add_argument(
        "--model_comb",
        type=str,
        default="resnet152_resnet152",
        required="single" in sys.argv,
    )
    args = parser.parse_args()

    run_config = RunConfig(args=args)
    return run_config
