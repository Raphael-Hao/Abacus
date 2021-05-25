#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import numpy as np

from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

from abacus.modeling.dataloader import load_data_for_sklearn
from abacus.modeling.predictor import LatencyPredictor
from abacus.option import RunConfig


class LRPredictor(LatencyPredictor):
    def __init__(
        self,
        run_config: RunConfig,
        models_id,
        epoch=30,
        batch_size=16,
        data_fname=None,
        split_ratio=0.8,
        path="/home/cwh/Lego",
        total_models=2,
        mig=0,
    ):
        super().__init__(
            run_config,
            "lr",
            models_id,
            epoch,
            batch_size,
            data_fname,
            split_ratio,
            path,
            total_models,
            mig,
        )
        self.trainX, self.trainY, self.testX, self.testY = load_data_for_sklearn(
            self._data_fname, self._split_ratio, self._models_id, self._data_path
        )

    def train(self, save_result=False, save_model=False, perf=False):
        regr = linear_model.LinearRegression(normalize=True)
        self._model = regr
        print(self.trainX, self.trainY)
        self._model.fit(self.trainX, self.trainY)
        pred = self._model.predict(self.testX)
        e = pred - self.testY
        mae = np.average(np.abs(e))
        mape = np.average(np.abs(e) / self.testY)
        print(mae)
        print(mape)
        if save_result is True:
            self.save_result(combination=self._data_fname, mae=mae, mape=mape)
