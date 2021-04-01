#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import numpy as np
import pandas as pd
import os
import glob
import torch

from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

from abacus.modeling.dataloader import load_data_for_sklearn
from abacus.modeling.predictor import MultiDNNPredictor
from abacus.utils import gen_model_combinations
from abacus.option import RunConfig
from abacus.option import parse_options

models_id = {
    "resnet50": 0,
    "resnet101": 1,
    "resnet152": 2,
    "inception_v3": 3,
    "vgg16": 4,
    "vgg19": 5,
    "bert": 6,
}
class SVMPredictor(MultiDNNPredictor):
        def __init__(
            self,
            models_id,
            epoch=30,
            batch_size=16,
            data_fname=None,
            split_ratio=0.8,
            path="/home/cwh/Lego",
            total_models=2,
        ):
            super().__init__(
                "svm",
                models_id,
                epoch,
                batch_size,
                data_fname,
                split_ratio,
                path,
                total_models,
            )

        def train(self, if_profile=False):
            trainX, trainY, testX, testY = load_data_for_sklearn(
                self._data_fname, self._split_ratio, self._models_id, self._data_path
            )
            regr = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, max_iter=5000))
            # self._model = regr
            print(trainX)
            print(trainY)
            print(len(trainX))
            print(len(trainY))
            regr.fit(trainX, trainY)

            pred = regr.predict(testX)

            e = pred - testY
            mae = np.average(np.abs(e))
            mape = np.average(np.abs(e) / testY)
            print(mae)
            print(mape)
            self.save_result(combination=self._data_fname, mae=mae, mape=mape)

def train_predictor(args: RunConfig):
    predictor = None
    if args.mode == "all":
        predictor = SVMPredictor(
                    models_id=args.models_id,
                    epoch=200,
                    batch_size=16,
                    data_fname="all",
                    path=args.path,
                    total_models=args.total_models,
                )
        predictor.train()
    elif args.mode == "onebyone":
        for model_combination in gen_model_combinations(
            args.models_name, args.training_combinations
        ):

            data_filename = model_combination[0]
            for model_name in model_combination[1:]:
                data_filename = data_filename + "_" + model_name
                print(data_filename)
                predictor = SVMPredictor(
                        models_id=args.models_id,
                        epoch=200,
                        batch_size=16,
                        data_fname=data_filename,
                        path=args.path,
                        total_models=args.total_models,
                    )
                predictor.train(if_profile=args.profile_predictor)

if __name__ == "__main__":
    run_config = parse_options()
    print(run_config)
    train_predictor(run_config)