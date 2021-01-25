#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import os
import glob
from abacus.config import args


def get_feature_latency(data):
    assert data.shape[1] == 13

    n = data.shape[0]
    feature_data = []
    latency_data = []
    for i in range(n):
        line = data[i]
        model1 = args.models_id[line[0]]
        model1_record = np.array(
            [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
        )
        model2 = args.models_id[line[5]]
        model2_record = np.array(
            [int(line[6]), int(line[7]), int(line[8]), int(line[9])]
        )
        model_feature = np.zeros([len(args.models_id)])
        model_feature[model1] += 1
        model_feature[model2] += 1
        if model1 <= model2:
            record = np.concatenate((model_feature, model1_record, model2_record))
        else:
            record = np.concatenate((model_feature, model2_record, model1_record))
        feature_data.append(record)
        latency_data.append(float(line[10]))
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


def load_torch_data(
    model_combinatin, batch_size, train_ratio, data_path="/home/cwh/Lego/data"
):
    all_feature = None
    all_latency = None
    if model_combinatin == "all":
        for filename in glob.glob(os.path.join(data_path, "*.csv")):
            feature_data, latency_data = load_single_file(
                os.path.join(data_path, filename)
            )
            if all_feature is None or all_latency is None:
                all_feature = feature_data
                all_latency = latency_data
            else:
                all_feature = np.concatenate((all_feature, feature_data), axis=0)
                all_latency = np.concatenate((all_latency, latency_data), axis=0)

    else:
        filename = model_combinatin + ".csv"
        all_feature, all_latency = load_single_file(os.path.join(data_path, filename))

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


def load_data_for_sklearn(
    model_combinatin, split_ratio, data_path="/home/cwh/Lego/data"
):
    all_feature = None
    all_latency = None
    if model_combinatin == "all":
        for filename in glob.glob(os.path.join(data_path, "*.csv")):
            feature_data, latency_data = load_single_file(
                os.path.join(data_path, filename)
            )
            if all_feature is None or all_latency is None:
                all_feature = feature_data
                all_latency = latency_data
            else:
                all_feature = np.concatenate((all_feature, feature_data), axis=0)
                all_latency = np.concatenate((all_latency, latency_data), axis=0)

    else:
        filename = model_combinatin + ".csv"
        all_feature, all_latency = load_single_file(os.path.join(data_path, filename))

    X, y = (all_feature, all_latency)
    y = y / 1000
    data = np.concatenate((X, np.reshape(y, (-1, 1))), axis=1)
    np.random.shuffle(data)
    n = data.shape[0]
    split = int(split_ratio * n)
    train = data[:split, :]
    test = data[split:, :]
    trainX = train[:, :16]
    trainY = train[:, 16:].reshape(-1)
    testX = test[:, :16]
    testY = test[:, 16:].reshape(-1)
    return trainX, trainY, testX, testY


if __name__ == "__main__":

    train_dataloader, test_dataloader = load_torch_data(None, 0.8)
