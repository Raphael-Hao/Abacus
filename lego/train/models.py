#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import torch.nn as nn
import torch.nn.functional as F

class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        self.hidden1 = nn.Linear(in_features=14, out_features=128, bias=True)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.predict = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        return output[:, 0]