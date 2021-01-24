#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import argparse

parser = argparse.ArgumentParser(description="Abacus")

parser.add_argument("--task", type=str, default="profile", choices=["profile", "train"])

args = parser.parse_args()

# general configurations
args.all_profiled_models = [
    "resnet50",  # 0
    "resnet101",  # 1
    "resnet152",  # 2
    "inception_v3",  # 3
    "vgg16",  # 4
    "vgg19",  # 5
    "bert",  # 6
]

args.models_id = {
    "resnet50": 0,
    "resnet101": 1,
    "resnet152": 2,
    "inception_v3": 3,
    "vgg16": 4,
    "vgg19": 5,
    "bert": 6,
}

args.total_models = 2
args.supported_batchsize = [1, 2, 4, 8, 16]
args.supported_seqlen = [8, 16, 32, 64]

# profiled configurations
args.profiled_combinations = [
    (1, 3),
    (0, 2),
    (2, 5),
    (0, 3),
    (1, 2),
    (3, 3),
    (5, 5),
    (4, 4),
    (1, 5),
    (2, 2),
    (0, 4),
    (1, 1),
    (0, 0),
    (4, 5),
    (1, 4),
    (0, 5),
    (2, 3),
    (3, 5),
    (0, 1),
    (3, 4),
    (2, 4),
    (6, 6),
]
args.total_test = 200
args.test_loop = 100

# prediction model configurations
args.model_select = "mlp"  # linear, svm
args.trained_combinations = []

args.mode = None
args.hyper_params = {}
