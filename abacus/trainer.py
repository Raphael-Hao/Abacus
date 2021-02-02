#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from abacus.utils import gen_model_combinations
from abacus.modeling.predictor import MLPPredictor, LRPredictor, SVMPredictor
from abacus.option import RunConfig

def train_predictor(args:RunConfig):
    if args.mode == "all":
        predictor = MLPPredictor(
            models_id = args.models_id,
            lr=args.hyper_params[args.mode][0],
            epoch=args.hyper_params[args.mode][1],
            batch_size=128,
            data_fname=args.mode,
            split_ratio=0.8,
            device=args.device,
            path=args.path,
        )
        predictor.train()
    elif args.mode == "single":
        predictor = MLPPredictor(
            models_id= args.models_id,
            epoch=args.hyper_params[args.model_combination][1],
            batch_size=8,
            lr=args.hyper_params[args.model_combination][0],
            data_fname=args.model_combination,
            split_ratio=0.8,
            device=args.device,
            path=args.path,
        )
        predictor.train()
    elif args.mode == "onebyone":
        for model_combination in gen_model_combinations(
            args.models_name, args.total_models, args.trained_combinations
        ):
            data_filename = model_combination[0]
            for model_name in model_combination[1:]:
                data_filename = data_filename + "_" + model_name
            print(data_filename)
            predictor = MLPPredictor(
                models_id= args.models_id,
                lr=args.hyper_params[data_filename][0],
                epoch=args.hyper_params[data_filename][1],
                batch_size=8,
                data_fname=data_filename,
                split_ratio=0.8,
                path=args.path,
                device=args.device,
            )
            predictor.train()
    else:
        raise NotImplementedError