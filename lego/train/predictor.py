#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import matplotlib as mpl
from lego.train.dataloader import load_data, load_single_file, load_data_for_sklearn
from lego.train.utils import AverageMeter
from lego.train.models import MLPregression
from lego.train.models import LinearRegression
from tqdm import tqdm
import math

from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn import linear_model

mpl.rcParams["font.family"] = "Times New Roman"


class MultiDNNPredictor:
    def __init__(
        self, total_epochs=30, batch_size=16, data_fname=None, split_ratio=0.8
    ):
        self._total_epochs = total_epochs
        self._batch_size = batch_size
        self._data_fname = data_fname
        self._split_ratio = split_ratio
        self._train_loader, self._test_loader = load_data(
            self._data_fname, self._batch_size, self._split_ratio
        )
        self._total_batches = len(self._train_loader)

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError


class MLPPredictor(MultiDNNPredictor):
    def __init__(
        self,
        epoch=30,
        batch_size=16,
        lr=0.001,
        lr_schedule_type="cosine",
        data_fname=None,
        split_ratio=0.8,
    ):
        super().__init__(epoch, batch_size, data_fname, split_ratio)
        self._init_lr = lr
        self._lr_schedule_type = lr_schedule_type
        self._device = torch.device("cuda:1")
        self._model = MLPregression().to(self._device)
        print(self._model)
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._init_lr)
        self._loss_func = nn.MSELoss()

    def train(self):

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
            train_loss_all.append(train_loss.avg)
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_all, "ro-", label="Train loss")
        plt.legend()
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(self._data_fname + "_train.pdf", bbox_inches="tight")
        plt.show()
        self.validate()

    def validate(self):
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
        print(mae)
        mape = np.average(np.abs(predict_latency - origin_latency) / origin_latency)
        print(mape)
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
        plt.savefig(self._data_fname + "_test.pdf", bbox_inches="tight")
        plt.show()

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


class LRPredictor(MultiDNNPredictor):
    def __init__(
        self,
        epoch=30,
        batch_size=16,
        lr=0.001,
        lr_schedule_type="cosine",
        data_fname=None,
        split_ratio=0.8,
    ):
        super().__init__(epoch, batch_size, data_fname, split_ratio)
        self._device = torch.device("cuda:1")
        self.trainX, self.trainY, self.testX, self.testY = load_data_for_sklearn(data_fname, split_ratio)
        
    def train(self):
        regr = linear_model.LinearRegression()
        self._model = regr

        self._model.fit(self.trainX, self.trainY)
        pred = self._model.predict(self.testX)
        e = pred - self.testY
        print(np.average(np.abs(e)))
        print(np.average(np.abs(e)/self.testY))


class SVMPredictor(MultiDNNPredictor):
    def __init__(
        self,
        epoch=30,
        batch_size=16,
        lr=0.001,
        lr_schedule_type="cosine",
        data_fname=None,
        split_ratio=0.8,
    ):
        super().__init__(epoch, batch_size, data_fname, split_ratio)
        self._device = torch.device("cuda:1")
        self.trainX, self.trainY, self.testX, self.testY = load_data_for_sklearn(data_fname, split_ratio)
        
    def train(self):
        regr = make_pipeline(StandardScaler(),LinearSVR(random_state=0, tol=1e-5, max_iter=10000))
        self._model = regr

        self._model.fit(self.trainX, self.trainY)
        pred = self._model.predict(self.testX)
        e = pred - self.testY
        print(np.average(np.abs(e)))
        print(np.average(np.abs(e)/self.testY))
