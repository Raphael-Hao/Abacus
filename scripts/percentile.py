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
            latency_data.append(150)
        latency_data.append(latency)
    latency_data = np.array(latency_data)
    return latency_data


def load_single_file(filepath):
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    # print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data).astype(np.float32)
    return get_latency(data)


# %% 2in7 in A100

colo_names = [
    "Res50+Res101",
    "Res50+Res152",
    "Res50+IncepV3",
    "Res50+VGG16",
    "Res50+VGG19",
    "Res50+bert",
    "Res101+Res152",
    "Res101+IncepV3",
    "Res101+VGG16",
    "Res101+VGG19",
    "Res101+bert",
    "Res152+IncepV3",
    "Res152+VGG16",
    "Res152+VGG19",
    "Res152+bert",
    "IncepV3+VGG16",
    "IncepV3+VGG19",
    "IncepV3+bert",
    "VGG16+VGG19",
    "VGG16+bert",
    "VGG19+bert",
]

file_names = [
    "resnet50resnet101",
    "resnet50resnet152",
    "resnet50inception_v3",
    "resnet50vgg16",
    "resnet50vgg19",
    "resnet50bert",
    "resnet101resnet152",
    "resnet101inception_v3",
    "resnet101vgg16",
    "resnet101vgg19",
    "resnet101bert",
    "resnet152inception_v3",
    "resnet152vgg16",
    "resnet152vgg19",
    "resnet152bert",
    "inception_v3vgg16",
    "inception_v3vgg19",
    "inception_v3bert",
    "vgg16vgg19",
    "vgg16bert",
    "vgg19bert",
]


qos_target = {
    "resnet50resnet101": 100,
    "resnet50resnet152": 100,
    "resnet50inception_v3": 100,
    "resnet50vgg16": 100,
    "resnet50vgg19": 100,
    "resnet50bert": 100,
    "resnet101resnet152": 100,
    "resnet101inception_v3": 100,
    "resnet101vgg16": 100,
    "resnet101vgg19": 100,
    "resnet101bert": 100,
    "resnet152inception_v3": 100,
    "resnet152vgg16": 100,
    "resnet152vgg19": 100,
    "resnet152bert": 100,
    "inception_v3vgg16": 100,
    "inception_v3vgg19": 100,
    "inception_v3bert": 100,
    "vgg16vgg19": 100,
    "vgg16bert": 100,
    "vgg19bert": 100,
}

data_dir = "../data/server/A100/2in7/"


# %% tail latency caculator

for i in range(len(file_names)):
    # print("--------------{}--------------".format(file_names[i]))
    abacus_latency = load_single_file(data_dir + "Abacus/{}.csv".format(file_names[i]))
    fcfs_latency = load_single_file(data_dir + "FCFS/{}.csv".format(file_names[i]))
    sjf_latency = load_single_file(data_dir + "SJF/{}.csv".format(file_names[i]))
    edf_latency = load_single_file(data_dir + "EDF/{}.csv".format(file_names[i]))

    abacus_tail = np.percentile(abacus_latency, 99)
    fcfs_tail = np.percentile(fcfs_latency, 99)
    sjf_tail = np.percentile(sjf_latency, 99)
    edf_tail = np.percentile(sjf_latency, 99)

    print("Abacus 99%-ile latency: {} for {}".format(abacus_tail, colo_names[i]))
    print("FCFS 99%-ile latency: {} for {}".format(fcfs_tail, colo_names[i]))
    print("SJF 99%-ile latency: {} for {}".format(sjf_tail, colo_names[i]))
    print("EDF 99%-ile latency: {} for {}".format(edf_tail, colo_names[i]))

    # abacus_vio = abacus_latency[abacus_latency < qos_target[file_names[i]]]
    # fcfs_vio = fcfs_latency[fcfs_latency < qos_target[file_names[i]]]
    # sjf_vio = sjf_latency[sjf_latency < qos_target[file_names[i]]]
    # edf_vio = edf_latency[edf_latency < qos_target[file_names[i]]]

    # origin_abacus_load = abacus_latency.shape[0]
    # origin_fcfs_load = fcfs_latency.shape[0]
    # origin_sjf_load = sjf_latency.shape[0]
    # abacus_load = abacus_vio.shape[0]
    # fcfs_load = fcfs_vio.shape[0]
    # sjf_load = sjf_vio.shape[0]

    # print(
    #     "{}, {},{},{}".format(
    #         colo_names[i],
    #         fcfs_load / origin_fcfs_load,
    #         sjf_load / origin_sjf_load,
    #         abacus_load / origin_abacus_load,
    #     )
    # )

# %%
