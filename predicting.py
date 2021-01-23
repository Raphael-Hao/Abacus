#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import argparse

from lego.utils import gen_model_combinations

from lego.train.predictor import MLPPredictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # inception_v3_inception_v3
    all_profiled_models = [
        "resnet50",  # 0
        "resnet101",  # 1
        "resnet152",  # 2
        "inception_v3",  # 3
        "vgg16",  # 4
        "vgg19",  # 5
        # "bert_base",  # 6
        # "bert_large",  # 7
    ]
    total_models = 2
    trained_combinations = []
    # predictor = MLPPredictor(
    #     lr=0.001, epoch=60, batch_size=64, data_fname=None, split_ratio=0.8
    # )
    # predictor.train()
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