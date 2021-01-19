#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


class avgpool_fc(nn.Module):
    def __init__(self, avgpool, fc):
        super(avgpool_fc, self).__init__()
        self.avgpool = avgpool
        self.fc = fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class avgpool_classifier(nn.Module):
    def __init__(self, avgpool, classifier):
        super(avgpool_classifier, self).__init__()
        self.avgpool = avgpool
        self.classifier = classifier

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class inception_transform_input(nn.Module):
    def __init__(self):
        super(inception_transform_input, self).__init__()

    def forward(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x


class inception_pair_x_to_aux(nn.Module):
    def __init__(self, x_to_aux):
        super(inception_pair_x_to_aux, self).__init__()
        self.x_to_aux = x_to_aux

    def forward(self, x):
        aux = self.x_to_aux(x)
        return (x, aux)


class inception_pair_x_to_x(nn.Module):
    def __init__(self, x_to_x):
        super(inception_pair_x_to_x, self).__init__()
        self.x_to_x = x_to_x

    def forward(self, *args):
        if isinstance(args[0], tuple):
            x = args[0][0]
            aux = args[0][1]
        else:
            x = args[0]
            aux = args[1]
        x = self.x_to_x(x)
        return (x, aux)


class inception_avgpool_dropout_flatten_fc(nn.Module):
    def __init__(self, avgpool, dropout, fc):
        super(inception_avgpool_dropout_flatten_fc, self).__init__()
        self.avgpool = avgpool
        self.dropout = dropout
        self.fc = fc

    def forward(self, x):
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class inception_output(nn.Module):
    def __init__(self, output):
        super(inception_output, self).__init__()
        self.output = output

    def forward(self, x):
        return self.output(x, None)