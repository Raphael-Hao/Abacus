#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import os
import csv


class MultiDNNPredictor:
    def __init__(
        self,
        model_select,
        models_id,
        total_epochs=30,
        batch_size=16,
        data_fname=None,
        split_ratio=0.8,
        path="/home/cwh/Lego",
        total_models=2,
        mig_enabled=False,
    ):
        self._model_select = model_select
        self._total_epochs = total_epochs
        self._batch_size = batch_size
        self._data_fname = data_fname
        self._split_ratio = split_ratio
        self._models_id = models_id
        self._path = path
        self._total_models = total_models
        self._mig_enabled = mig_enabled
        if self._total_models == 2:
            if self._mig_enabled is False:
                self._data_path = os.path.join(self._path, "data/profile/A100/2in7")
                self._save_path = os.path.join(self._path, "model/A100/2in7")
            else:
                self._data_path = os.path.join(self._path, "data/profile/mig/2in4")
                self._save_path = os.path.join(self._path, "model/mig/2in4")
        elif self._total_models == 3:
            self._data_path = os.path.join(self._path, "data/profile/A100/3in4")
            self._save_path = os.path.join(self._path, "model/A100/3in4")
        elif self._total_models == 4:
            if self._mig_enabled is False:
                self._data_path = os.path.join(self._path, "data/profile/A100/4in4")
                self._save_path = os.path.join(self._path, "model/A100/4in4")
            else:
                self._data_path = os.path.join(self._path, "data/profile/mig/4in4")
                self._save_path = os.path.join(self._path, "model/mig/4in4")
        else:
            raise NotImplementedError

        os.makedirs(self._save_path, exist_ok=True)
        self._result_path = os.path.join(self._path, "result")
        os.makedirs(self._result_path, exist_ok=True)
        self._result_fname = os.path.join(self._path, "results.csv")
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