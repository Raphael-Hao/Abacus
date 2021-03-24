#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import argparse
import os
from abacus.network.resnet_splited import resnet50, resnet101, resnet152
from abacus.network.inception_splited import inception_v3
from abacus.network.vgg_splited import vgg16, vgg19
from abacus.network.bert import BertModel


class RunConfig:
    def __init__(self, args) -> None:
        self.task = args.task
        # general configurations
        self.total_models = 3
        self.device = 0
        self.path = "/state/partition/whcui/repository/project/Abacus"
        # self.path = "/home/cwh/Lego"
        self.data_path = os.path.join(self.path, "data")
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

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

        self.supported_batchsize = [
            1,
            2,
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
            self.policy = "EDF"
            self.threshold = 2
            self.qos_target = 60
            self.search_ways = 2
            self.total_queries = 1000
            self.average_duration = args.load
            self.abandon = True

        elif self.task == "profile":
            """
            [profiled configurations]
            """
            self.total_test = 200
            # self.total_test = 1
            self.test_loop = 100
            self.profiled_combinations = [
                # 1, 2, 5, 6
                # (1, 2, 5),
                # (1, 2, 6),
                # (1, 5, 6),
                # (2, 5, 6),
                (3, 4, 6),
                (0, 1, 1),
                (0, 5, 5),
                (2, 3, 5),
                (0, 4, 5),
                (2, 2, 5),
                (4, 6, 6),
                (0, 1, 2),
                (0, 5, 6),
                (1, 4, 4),
                (0, 4, 4),
                (1, 1, 6),
                (2, 2, 4),
                (2, 3, 6),
                (0, 1, 3),
                (1, 4, 5),
                (0, 1, 4),
                (1, 1, 4),
                (5, 6, 6),
                (0, 1, 5),
                (3, 3, 3),
                (0, 1, 6),
                (0, 2, 3),
                (0, 0, 6),
                (2, 6, 6),
                (2, 3, 3),
                (3, 3, 5),
                (0, 2, 2),
                (0, 0, 5),
                (0, 6, 6),
                (3, 6, 6),
                (5, 5, 5),
                (3, 3, 4),
                (0, 2, 5),
                (0, 0, 4),
                (2, 4, 5),
                (0, 2, 4),
                (0, 0, 3),
                (2, 4, 4),
                (0, 3, 3),
                (3, 3, 6),
                (5, 5, 6),
                (0, 0, 2),
                (1, 2, 2),
                (1, 6, 6),
                (0, 2, 6),
                (0, 0, 1),
                (2, 4, 6),
                (1, 2, 3),
                (2, 2, 6),
                (0, 0, 0),
                (4, 5, 5),
                (0, 3, 6),
                (1, 2, 4),
                (2, 3, 4),
                (1, 1, 1),
                (1, 5, 5),
                (2, 2, 3),
                (4, 5, 6),
                (2, 2, 2),
                (4, 4, 6),
                (0, 3, 4),
                (1, 1, 3),
                (1, 3, 6),
                (4, 4, 5),
                (0, 3, 5),
                (2, 5, 5),
                (6, 6, 6),
                (1, 1, 2),
                (1, 3, 5),
                (3, 4, 4),
                (4, 4, 4),
                # (3, 5, 6), ###
                (1, 1, 5),
                (1, 3, 4),
                (3, 4, 5),
                (1, 4, 6),
                (3, 5, 5),
                (0, 4, 6),
                (1, 3, 3),
            ]
            # self.profiled_combinations = [
            #     (0, 0),
            #     (1, 1),
            #     (2, 2),
            #     (3, 3),
            #     (4, 4),
            #     (5, 5),
            #     (6, 6),
            #     # (0, 1),
            #     # (0, 2),
            #     # (0, 3),
            #     # (0, 4),
            #     # (0, 5),
            #     # (0, 6),
            #     # (1, 2),
            #     # (1, 3),
            #     # (1, 4),
            #     # (1, 5),
            #     # (1, 6),
            #     # (2, 3),
            #     # (2, 4),
            #     # (2, 5),
            #     # (2, 6),
            #     # (3, 4),
            #     # (3, 5),
            #     # (3, 6),
            #     # (4, 5),
            #     # (4, 6),
            #     # (5, 6),
            # ]

        elif args.task == "train":
            """
            [prediction model configurations]
            """
            self.trained_combinations = [
                # 2, 3, 5, 6
                (3, 4, 6),
                (0, 1, 1),
                (0, 5, 5),
                # (2, 3, 5),
                (0, 4, 5),
                (2, 2, 5),
                (4, 6, 6),
                (0, 1, 2),
                (0, 5, 6),
                (1, 4, 4),
                (0, 4, 4),
                (1, 1, 6),
                (2, 2, 4),
                # (2, 3, 6),
                (0, 1, 3),
                (1, 4, 5),
                (0, 1, 4),
                (1, 1, 4),
                (5, 6, 6),
                (0, 1, 5),
                (3, 3, 3),
                (0, 1, 6),
                (0, 2, 3),
                (0, 0, 6),
                (2, 6, 6),
                (2, 3, 3),
                (3, 3, 5),
                (0, 2, 2),
                (0, 0, 5),
                (0, 6, 6),
                (3, 6, 6),
                (5, 5, 5),
                (3, 3, 4),
                (0, 2, 5),
                (0, 0, 4),
                (2, 4, 5),
                (0, 2, 4),
                (0, 0, 3),
                (2, 4, 4),
                (0, 3, 3),
                (3, 3, 6),
                (5, 5, 6),
                (0, 0, 2),
                (1, 2, 2),
                (1, 6, 6),
                (0, 2, 6),
                (0, 0, 1),
                (2, 4, 6),
                (1, 2, 3),
                (2, 2, 6),
                (0, 0, 0),
                (4, 5, 5),
                (0, 3, 6),
                (1, 2, 4),
                (2, 3, 4),
                # (2, 5, 6),
                (1, 1, 1),
                (1, 5, 5),
                (2, 2, 3),
                (1, 2, 5),
                (4, 5, 6),
                (2, 2, 2),
                (4, 4, 6),
                (0, 3, 4),
                (1, 2, 6),
                (1, 1, 3),
                (1, 3, 6),
                (4, 4, 5),
                (0, 3, 5),
                (2, 5, 5),
                (6, 6, 6),
                (1, 1, 2),
                (1, 3, 5),
                (1, 5, 6),
                (3, 4, 4),
                (4, 4, 4),
                # (3, 5, 6),
                (1, 1, 5),
                (1, 3, 4),
                (3, 4, 5),
                (1, 4, 6),
                (3, 5, 5),
                (0, 4, 6),
                (1, 3, 3),
            ]
            # self.trained_combinations = [
            # (0, 5),
            # (1, 5),
            # (2, 5),
            # (3, 5),
            # (4, 5),
            # (5, 5),
            # (0, 4),
            # (1, 4),
            # (2, 4),
            # (3, 4),
            # (4, 4),
            # (0, 3),
            # (1, 3),
            # (2, 3),
            # (3, 3),
            # (0, 2),
            # (1, 2),
            # (2, 2),
            # (0, 1),
            # (1, 1),
            # (0, 0),
            # ]

            # self.mode = "onebyone"
            # self.mode = "single"
            self.mode = "all"
            # self.modeling = "lr"
            # self.modeling = "svm"
            self.modeling = "mlp"
            self.model_combination = "resnet152_resnet152"
            self.hyper_params = {
                "all": [0.002, 180],
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
        choices=["profile", "train", "serve", "background"],
    )
    parser.add_argument("--combination", type=int, nargs="+")
    parser.add_argument(
        "--policy", type=str, default="Abacus", choices=["Abacus", "SJF", "FCFS"]
    )
    parser.add_argument("--load", type=int, default=50)
    args = parser.parse_args()

    run_config = RunConfig(args=args)
    return run_config
