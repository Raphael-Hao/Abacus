#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
# %%
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
import os
import glob


def get_feature_latency(data, models_id, total_models=2):

    n = data.shape[0]
    feature_data = []
    latency_data = []
    for i in range(n):
        line = data[i]
        # print(line)
        model_records = {}
        model_feature = np.zeros([len(models_id)])
        for i in range(total_models):
            model_id = models_id[line[i * 5]]
            model_feature[model_id] += 1
            model_record = np.array(
                [
                    int(line[1 + i * 5]),
                    int(line[2 + i * 5]),
                    int(line[3 + i * 5]),
                    int(line[4 + i * 5]),
                ]
            )
            model_records[model_id] = model_record

        model_records = dict(sorted(model_records.items()))
        # print(model_records)
        for key in model_records:
            model_feature = np.concatenate((model_feature, model_records[key]))

        feature_data.append(model_feature)
        # print(model_feature)
        # print(line[-3])
        latency_data.append(float(line[-3]))
    feature_data = np.array(feature_data)
    latency_data = np.array(latency_data)
    return feature_data, latency_data


def load_single_file(filepath, models_id, total_models=2):
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data)
    return get_feature_latency(data, models_id, total_models)


def load_torch_data(
    model_combinatin,
    batch_size,
    train_ratio,
    models_id,
    data_path="/home/cwh/Lego/data",
    total_models=2,
):
    all_feature = None
    all_latency = None
    if model_combinatin == "all":
        for filename in glob.glob(os.path.join(data_path, "*.csv")):
            # print(filename)
            feature_data, latency_data = load_single_file(
                filename, models_id, total_models
            )
            if all_feature is None or all_latency is None:
                all_feature = feature_data
                all_latency = latency_data
            else:
                all_feature = np.concatenate((all_feature, feature_data), axis=0)
                all_latency = np.concatenate((all_latency, latency_data), axis=0)

    else:
        filename = model_combinatin + ".csv"
        all_feature, all_latency = load_single_file(
            os.path.join(data_path, filename), models_id, total_models
        )

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
    model_combinatin,
    split_ratio,
    models_id,
    data_path="/home/cwh/Lego/data",
    total_models=2,
):
    all_feature = None
    all_latency = None
    if model_combinatin == "all":
        for filename in glob.glob(os.path.join(data_path, "*.csv")):
            feature_data, latency_data = load_single_file(
                filename, models_id, total_models
            )
            if all_feature is None or all_latency is None:
                all_feature = feature_data
                all_latency = latency_data
            else:
                all_feature = np.concatenate((all_feature, feature_data), axis=0)
                all_latency = np.concatenate((all_latency, latency_data), axis=0)

    else:
        filename = model_combinatin + ".csv"
        all_feature, all_latency = load_single_file(
            os.path.join(data_path, filename), models_id, total_models
        )
    X, y = (all_feature, all_latency)
    y = y / 1000
    data = np.concatenate((X, np.reshape(y, (-1, 1))), axis=1)
    np.random.shuffle(data)
    n = data.shape[0]
    split = int(split_ratio * n)
    train = data[:split, :]
    test = data[split:, :]
    trainX = train[:, :-1]
    trainY = train[:, -1:].reshape(-1)
    testX = test[:, :-1]
    testY = test[:, -1:].reshape(-1)

    return trainX, trainY, testX, testY


# %%
if __name__ == "__main__":
    models_id = {
        "resnet50": 0,
        "resnet101": 1,
        "resnet152": 2,
        "inception_v3": 3,
        "vgg16": 4,
        "vgg19": 5,
        "bert": 6,
    }
    train_dataloader, test_dataloader = load_torch_data(
        "inception_v3_vgg19_bert",
        128,
        0.8,
        models_id,
        data_path="/home/cwh/Lego/data/profile/3in4/",
        total_models=3,
    )
    trainX, trainY, testX, testY = load_data_for_sklearn(
        "inception_v3_vgg19_bert",
        0.8,
        models_id,
        data_path="/home/cwh/Lego/data/profile/3in4/",
        total_models=3,
    )
    print(trainX.shape)
    print(trainY)
    print(testX)
    print(testY)

# %%
