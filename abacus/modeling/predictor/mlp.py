#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import os
import time
import math
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from abacus.modeling.predictor import MultiDNNPredictor
from abacus.modeling.dataloader import load_torch_data
from abacus.modeling.utils import AverageMeter


class MLPregression(nn.Module):
    def __init__(self, first_layer=15):
        super(MLPregression, self).__init__()
        self.hidden1 = nn.Linear(in_features=first_layer, out_features=32, bias=True)
        self.hidden2 = nn.Linear(32, 32)
        self.hidden3 = nn.Linear(32, 32)
        self.predict = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:, 0]


class MLPPredictor(MultiDNNPredictor):
    def __init__(
        self,
        models_id,
        epoch=30,
        batch_size=16,
        lr=0.001,
        lr_schedule_type="cosine",
        data_fname="all",
        split_ratio=0.8,
        path="/home/cwh/Lego",
        device=0,
        total_models=2,
        mig=0,
    ):
        super().__init__(
            "mlp",
            models_id,
            epoch,
            batch_size,
            data_fname,
            split_ratio,
            path,
            total_models,
            mig,
        )
        self._train_loader, self._test_loader = load_torch_data(
            self._data_fname,
            self._batch_size,
            self._split_ratio,
            self._models_id,
            self._data_path,
            self._total_models,
        )
        self._total_batches = len(self._train_loader)
        self._init_lr = lr
        self._lr_schedule_type = lr_schedule_type
        self._device = torch.device("cuda:{}".format(device))
        self._firt_layer = self._total_models * 4 + 7
        self._model = MLPregression(self._firt_layer).to(self._device)
        print(self._model)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._init_lr)
        self._loss_func = nn.MSELoss()

    def train(self, save_result=False, save_model=False, perf=False):

        train_loss_all = []
        for epoch in range(self._total_epochs):
            self._model.train()

            print("total train data: {}".format(self._total_batches))
            train_loss = AverageMeter()

            with tqdm(
                total=self._total_batches, desc="Train Epoch #{}".format(epoch + 1)
            ) as t:
                for i, (input, latency) in enumerate(self._train_loader):
                    new_lr = self.adjust_learning_rate(epoch, i)
                    input, latency = input.to(self._device), latency.to(self._device)
                    output = self._model(input)
                    loss = self._loss_func(output, latency)
                    self._model.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    train_loss.update(loss.item(), input.size(0))
                    t.set_postfix(
                        {
                            "loss": train_loss.avg,
                            "input size": input.size(1),
                            "lr": new_lr,
                        }
                    )
                    t.update(1)
            self.validate()
            train_loss_all.append(train_loss.avg)
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_all, "ro-", label="Train loss")
        plt.legend()
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(
            os.path.join(self._result_path, self._data_fname + "_train.pdf"),
            bbox_inches="tight",
        )
        plt.show()
        self.validate(save_result=save_result, save_model=save_model, perf=perf)

    def validate(self, save_result=False, save_model=False, perf=False):
        origin_latency = None
        predict_latency = None
        self._model.eval()
        total_batches = len(self._test_loader)
        test_loss = AverageMeter()
        with torch.no_grad():
            with tqdm(total=total_batches, desc="Testing ") as t:
                for step, (input, latency) in enumerate(self._test_loader):
                    input, latency = input.to(self._device), latency.to(self._device)
                    pre_y = self._model(input)
                    loss = self._loss_func(pre_y, latency)
                    test_loss.update(loss.item(), input.size(0))
                    t.set_postfix(
                        {
                            "loss": test_loss.avg,
                            "input_size": input.size(1),
                        }
                    )
                    t.update(1)

                    if origin_latency is None or predict_latency is None:
                        origin_latency = latency.cpu().data.numpy()
                        predict_latency = pre_y.cpu().data.numpy()
                    else:
                        origin_latency = np.concatenate(
                            (origin_latency, latency.cpu().data.numpy())
                        )
                        predict_latency = np.concatenate(
                            (predict_latency, pre_y.cpu().data.numpy())
                        )

        mae = np.average(np.abs(predict_latency - origin_latency))
        mape = np.average(np.abs(predict_latency - origin_latency) / origin_latency)
        print("mae: {}, mape: {}".format(mae, mape))
        if save_result is True:
            self.save_result(self._data_fname, mae, mape)
            index = np.argsort(origin_latency)
            plt.figure(figsize=(12, 5))
            plt.plot(
                np.arange(len(origin_latency)),
                origin_latency[index],
                "r",
                label="original y",
            )
            plt.scatter(
                np.arange(len(predict_latency)),
                predict_latency[index],
                s=3,
                c="b",
                label="prediction",
            )
            plt.legend(loc="upper left")
            plt.grid()
            plt.xlabel("index")
            plt.ylabel("y")
            plt.savefig(
                os.path.join(self._result_path, self._data_fname + "_test.pdf"),
                bbox_inches="tight",
            )
        if save_model:
            self.save_model()
        if perf is True:
            self._model.cpu().eval()
            total_cores = os.cpu_count()
            for cores in range(1, total_cores + 1):
                torch.set_num_threads(cores)
                # torch.set_num_interop_threads(cores)
                with torch.no_grad():
                    for bs in range(1, 17):
                        test_input = torch.rand((bs, self._firt_layer))
                        start_time = time.time()
                        for i in range(1000):
                            test_output = self._model(test_input)
                            # print(test_output[0])
                        end_time = time.time() - start_time
                        print(
                            "cores: {}, batch size: {}, Inference time: {} ms".format(
                                cores, bs, end_time
                            )
                        )

    def calc_learning_rate(self, epoch, batch=0):
        if self._lr_schedule_type == "cosine":
            T_total = self._total_epochs * self._total_batches
            T_cur = epoch * self._total_batches + batch
            lr = 0.5 * self._init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self._lr_schedule_type is None:
            lr = self._init_lr
        else:
            raise ValueError("do not support: %s" % self._lr_schedule_type)
        return lr

    def adjust_learning_rate(self, epoch, batch=0):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, batch)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def save_model(self):
        print("saving model......")
        torch.save(
            self._model.state_dict(),
            os.path.join(self._save_path, self._data_fname + ".ckpt"),
        )