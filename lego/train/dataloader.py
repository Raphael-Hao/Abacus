#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import os
import glob


def get_feature_latency(data):
    assert data.shape[1] == 11
    models_id = {
        "vgg16": 0,
        "vgg19": 1,
        "resnet50": 2,
        "resnet101": 3,
        "resnet152": 4,
        "inception_v3": 5,
        "bert_base": 6,
        "bert_large": 7,
    }
    n = data.shape[0]
    feature_data = []
    latency_data = []
    for i in range(n):
        line = data[i]
        model1 = models_id[line[0]]
        model1_record = np.array([int(line[1]), int(line[2]), int(line[3])])
        model2 = models_id[line[4]]
        model2_record = np.array([int(line[5]), int(line[6]), int(line[7])])
        model_feature = np.zeros([len(models_id)])
        model_feature[model1] += 1
        model_feature[model2] += 1
        if model1 <= model2:
            record = np.concatenate((model_feature, model1_record, model2_record))
        else:
            record = np.concatenate((model_feature, model2_record, model1_record))
        feature_data.append(record)
        latency_data.append(float(line[8]))
    feature_data = np.array(feature_data)
    latency_data = np.array(latency_data)
    return feature_data, latency_data


def load_single_file(filepath):
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data)
    return get_feature_latency(data)


def load_data(model_combinatin, batch_size, train_ratio):
    path = "/home/cwh/Project/Lego/data"
    all_feature = None
    all_latency = None
    if model_combinatin == None:
        for filename in glob.glob(os.path.join(path, "*.csv")):
            feature_data, latency_data = load_single_file(os.path.join(path, filename))
            if all_feature is None or all_latency is None:
                all_feature = feature_data
                all_latency = latency_data
            else:
                all_feature = np.concatenate((all_feature, feature_data), axis=0)
                all_latency = np.concatenate((all_latency, latency_data), axis=0)

    elif isinstance(model_combinatin, str):
        filename = model_combinatin + ".csv"
        all_feature, all_latency = load_single_file(os.path.join(path, filename))

    else:
        raise NotImplementedError

    feature_len = len(all_feature)
    latency_len = len(all_latency)
    assert feature_len == latency_len
    train_len = int(feature_len * train_ratio)
    test_len = feature_len - train_len
    all_feature = torch.from_numpy(all_feature.astype(np.float32))
    all_latency = all_latency / 1000
    all_latency = torch.from_numpy(all_latency.astype(np.float32))

    all_dataset = Data.TensorDataset(all_feature, all_latency)
    train_dataset, test_dataset = Data.random_split(all_dataset, [train_len, test_len])
    train_dataloader = Data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = Data.DataLoader(
        dataset=test_dataset, batch_size=128, shuffle=True
    )
    return train_dataloader, test_dataloader


if __name__ == "__main__":

    train_dataloader, test_dataloader = load_data(None, 0.8)
