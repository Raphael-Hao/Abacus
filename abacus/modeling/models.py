#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import torch.nn as nn
import torch.nn.functional as F

class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        self.hidden1 = nn.Linear(in_features=15, out_features=32, bias=True)
        self.hidden2 = nn.Linear(32, 16)
        # self.hidden3 = nn.Linear(128, 64)
        self.predict = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:, 0]


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.layer = nn.Linear(16, 1)

    def forward(self, x):
        return self.layer(x)