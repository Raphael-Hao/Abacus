#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import argparse
def parse_options():
    parser = argparse.ArgumentParser(description="Abacus")

    parser.add_argument("--task", type=str, default="profile", choices=["profile", "train"])

    args = parser.parse_args()

    # general configurations
    args.path="/home/cwh/Project/Lego"

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
    args.trained_combinations = [
        (6, 6),
        (0, 6),
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
    ]

    # args.mode = "onebyone"
    # args.mode = "single"
    args.device = 0
    args.mode = "all"
    args.model_combination = "resnet152_resnet152"
    args.hyper_params = {
        "all": [0.0016, 200],
        "resnet101_inception_v3": [0.001, 80],
        "bert_bert": [0.001, 80],
        "vgg19_bert": [0.001, 80],
        "resnet50_resnet152": [0.001, 80],
        "resnet101_bert": [0.001, 80],
        "resnet152_vgg19": [0.001, 80],
        "resnet50_inception_v3": [0.001, 80],
        "resnet101_resnet152": [0.001, 80],
        "inception_v3_inception_v3": [0.001, 80],
        "vgg19_vgg19": [0.001, 80],
        "vgg16_vgg16": [0.001, 80],
        "resnet101_vgg19": [0.001, 80],
        "inception_v3_bert": [0.001, 80],
        "resnet152_resnet152": [0.001, 80],
        "resnet50_vgg16": [0.001, 80],
        "resnet101_resnet101": [0.001, 80],
        "resnet50_resnet50": [0.001, 80],
        "resnet152_bert": [0.001, 80],
        "vgg16_vgg19": [0.001, 80],
        "resnet101_vgg16": [0.001, 80],
        "resnet50_vgg19": [0.001, 80],
        "resnet152_inception_v3": [0.001, 80],
        "inception_v3_vgg19": [0.001, 80],
        "resnet50_resnet101": [0.001, 80],
        "vgg16_bert": [0.001, 80],
        "resnet50_bert": [0.001, 80],
        "inception_v3_vgg16": [0.001, 80],
        "resnet152_vgg16": [0.001, 80],
    }
    return args
