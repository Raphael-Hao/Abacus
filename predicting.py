#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from lego.utils import gen_model_combinations

from lego.train.predictor import MLPPredictor


if __name__ == "__main__":
    # inception_v3_inception_v3
    all_profiled_models = [
        "vgg16",  # 0
        "vgg19",  # 1
        "resnet50",  # 2
        "resnet101",  # 3
        "resnet152",  # 4
        "inception_v3",  # 5
        "bert_base",  # 6
        "bert_large",  # 7
    ]
    total_models = 2
    trained_combinations = []
    for model_combination in gen_model_combinations(
        all_profiled_models, total_models, trained_combinations
    ):
        data_filename = model_combination[0]
        for model_name in model_combination[1:]:
            data_filename = data_filename + "_" + model_name
        predictor = MLPPredictor(
            lr=0.001, epoch=60, batch_size=8, data_fname=data_filename, split_ratio=0.8
        )
        predictor.train()