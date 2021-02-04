#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
# %%
import numpy as np
import pandas as pd


def get_latency(data):

    n = data.shape[0]
    latency_data = []
    for i in range(n):
        line = data[i]
        latency = float(line[-1])
        if latency == -1:
            continue
        latency_data.append(latency)
    latency_data = np.array(latency_data)
    return latency_data

def load_single_file(filepath):
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data).astype(np.float32)
    return get_latency(data)

# %%

abacus_latency = load_single_file("../data/server/Abacus/resnet50resnet101.csv")
fcfs_latency = load_single_file("../data/server/FCFS/resnet50resnet101.csv")
sjf_latency = load_single_file("../data/server/SJF/resnet50resnet101.csv")
# %%

print(np.percentile(abacus_latency, 99))
print(np.percentile(fcfs_latency, 99))
print(np.percentile(sjf_latency, 99))
abacus_latency = abacus_latency[abacus_latency<60]
fcfs_latency = fcfs_latency[fcfs_latency<60]
sjf_latency = sjf_latency[sjf_latency<60]
print(np.mean(abacus_latency))
print(np.mean(fcfs_latency))
print(np.mean(sjf_latency))

# %%
