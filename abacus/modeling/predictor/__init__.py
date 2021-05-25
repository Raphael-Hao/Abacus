#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import os
import csv
from abacus.option import RunConfig


class LatencyPredictor:
    def __init__(
        self,
        run_config: RunConfig,
        model_select,
        models_id,
        total_epochs,
        batch_size,
        data_fname,
        split_ratio,
        path,
        total_models,
        mig,
    ):
        self._run_config = run_config
        self._model_select = model_select
        self._total_epochs = total_epochs
        self._batch_size = batch_size
        self._data_fname = data_fname
        self._split_ratio = split_ratio
        self._models_id = models_id
        self._path = path
        self._total_models = total_models
        self._mig = mig
        if self._total_models == 2:
            if self._mig == 0:
                self._data_path = os.path.join(self._path, "data/profile/A100/2in7")
                self._save_path = os.path.join(self._path, "model/A100/2in7")
                self._result_path = os.path.join(self._path, "result/A100/2in7")
            elif self._mig == 2:
                self._data_path = os.path.join(self._path, "data/profile/mig/2in4")
                self._save_path = os.path.join(self._path, "model/mig/2in4")
                self._result_path = os.path.join(self._path, "result/mig/2in4")
            else:
                raise NotImplementedError
        elif self._total_models == 3:
            self._data_path = os.path.join(self._path, "data/profile/A100/3in4")
            self._save_path = os.path.join(self._path, "model/A100/3in4")
            self._result_path = os.path.join(self._path, "result/A100/3in4")
        elif self._total_models == 4:
            if self._mig == 0:
                if self._run_config.platform == "":
                    self._data_path = os.path.join(self._path, "data/profile/A100/4in4")
                    self._save_path = os.path.join(self._path, "model/A100/4in4")
                    self._result_path = os.path.join(self._path, "result/A100/4in4")
                self._data_path = os.path.join(self._path, "data/profile/A100/4in4")
                self._save_path = os.path.join(self._path, "model/A100/4in4")
                self._result_path = os.path.join(self._path, "result/A100/4in4")
            elif self._mig == 1:
                self._data_path = os.path.join(self._path, "data/profile/mig/4in4")
                self._save_path = os.path.join(self._path, "model/mig/4in4")
                self._result_path = os.path.join(self._path, "result/mig/4in4")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        os.makedirs(self._save_path, exist_ok=True)
        os.makedirs(self._result_path, exist_ok=True)
        self._result_fname = os.path.join(self._result_path, "results.csv")
        if not os.path.exists(self._result_fname):
            result_file = open(self._result_fname, "w+")
            wr = csv.writer(result_file, dialect="excel")
            wr.writerow(["predictor", "combinations", "mae", "mape"])

    def train(self, save_result=False, save_model=False, perf=False):
        raise NotImplementedError

    def validate(self, save_result=False, save_model=False, perf=False):
        raise NotImplementedError

    def save_result(self, combination, mae, mape):
        result_file = open(self._result_fname, "a+")
        wr = csv.writer(result_file, dialect="excel")
        if combination is None:
            combination = "all"
        wr.writerow([self._model_select, combination, mae, mape])