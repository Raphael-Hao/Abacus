#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from abacus.utils import gen_model_combinations
from abacus.option import RunConfig


def train_predictor(args: RunConfig):
    predictor = None
    if args.mode == "all":
        if args.modeling == "mlp":
            from abacus.modeling.predictor.mlp import MLPPredictor

            predictor = MLPPredictor(
                models_id=args.models_id,
                lr=args.hyper_params[args.mode][0],
                epoch=args.hyper_params[args.mode][1],
                batch_size=128,
                data_fname=args.mode,
                split_ratio=0.8,
                device=args.device,
                path=args.path,
                total_models=args.total_models,
            )
        elif args.modeling == "lr":
            from abacus.modeling.predictor.lr import LRPredictor

            predictor = LRPredictor(
                models_id=args.models_id,
                epoch=200,
                batch_size=16,
                data_fname="all",
                path=args.path,
                total_models=args.total_models,
            )
        elif args.modeling == "svm":
            from abacus.modeling.predictor.svm import SVMPredictor

            predictor = SVMPredictor(
                models_id=args.models_id,
                epoch=200,
                batch_size=16,
                data_fname="all",
                path=args.path,
                total_models=args.total_models,
            )
        else:
            raise NotImplementedError
        predictor.train(save_result=True, save_model=True, perf=True)
    elif args.mode == "single":
        if args.modeling == "mlp":
            from abacus.modeling.predictor.mlp import MLPPredictor

            predictor = MLPPredictor(
                models_id=args.models_id,
                epoch=args.hyper_params[args.model_combination][1],
                batch_size=8,
                lr=args.hyper_params[args.model_combination][0],
                data_fname=args.model_combination,
                split_ratio=0.8,
                device=args.device,
                path=args.path,
                total_models=args.total_models,
            )
        elif args.modeling == "lr":
            from abacus.modeling.predictor.lr import LRPredictor

            predictor = LRPredictor(
                models_id=args.models_id,
                epoch=200,
                batch_size=16,
                data_fname=args.model_combination,
                path=args.path,
                total_models=args.total_models,
            )
        elif args.modeling == "svm":
            from abacus.modeling.predictor.lr import LRPredictor

            predictor = SVMPredictor(
                models_id=args.models_id,
                epoch=200,
                batch_size=16,
                data_fname=args.model_combination,
                path=args.path,
                total_models=args.total_models,
            )
        else:
            raise NotImplementedError
        predictor.train(save_result=True, save_model=False, perf=False)

    elif args.mode == "onebyone":
        for model_combination in gen_model_combinations(
            args.models_name, args.training_combinations
        ):

            data_filename = model_combination[0]
            for model_name in model_combination[1:]:
                data_filename = data_filename + "_" + model_name
            print(data_filename)
            if args.modeling == "mlp":
                from abacus.modeling.predictor.mlp import MLPPredictor

                predictor = MLPPredictor(
                    models_id=args.models_id,
                    lr=args.hyper_params[data_filename][0],
                    epoch=args.hyper_params[data_filename][1],
                    batch_size=8,
                    data_fname=data_filename,
                    split_ratio=0.8,
                    path=args.path,
                    device=args.device,
                    total_models=args.total_models,
                )
            elif args.modeling == "lr":
                from abacus.modeling.predictor.lr import LRPredictor

                predictor = LRPredictor(
                    models_id=args.models_id,
                    epoch=200,
                    batch_size=16,
                    data_fname=data_filename,
                    path=args.path,
                    total_models=args.total_models,
                )
            elif args.modeling == "svm":
                from abacus.modeling.predictor.svm import SVMPredictor

                predictor = SVMPredictor(
                    models_id=args.models_id,
                    epoch=200,
                    batch_size=16,
                    data_fname=data_filename,
                    path=args.path,
                    total_models=args.total_models,
                )
            else:
                raise NotImplementedError
            predictor.train(save_result=True, save_model=False, perf=False)

    else:
        raise NotImplementedError
