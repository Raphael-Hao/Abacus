#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao


from lego.utils import gen_model_combinations
from lego.config import args
from lego.train.predictor import MLPPredictor, LRPredictor, SVMPredictor


if __name__ == "__main__":

    if args.mode == "all":
        predictor = MLPPredictor(
            lr=0.001, epoch=60, batch_size=64, data_fname=args.mode, split_ratio=0.8
        )
        predictor.train()
        # for model_combination in gen_model_combinations(
        #     args.all_profiled_models, args.total_models, args.trained_combinations
        # ):
    elif args.mode == "single":
        predictor = MLPPredictor(
            epoch=args.hyper_params[args.model_combination][1],
            batch_size=8,
            lr=args.hyper_params[args.model_combination][0],
            data_fname=args.model_combination,
            split_ratio=0.8,
        )
        predictor.train()
    elif args.mode == "onebyone":
        for model_combination in gen_model_combinations(
            args.all_profiled_models, args.total_models, args.trained_combinations
        ):
            data_filename = model_combination[0]
            for model_name in model_combination[1:]:
                data_filename = data_filename + "_" + model_name
            print(data_filename)
            predictor = MLPPredictor(
                lr=args.hyper_params[data_filename][0],
                epoch=args.hyper_params[data_filename][1],
                batch_size=8,
                data_fname=data_filename,
                split_ratio=0.8,
            )
            predictor.train()
    else:
        raise NotImplementedError