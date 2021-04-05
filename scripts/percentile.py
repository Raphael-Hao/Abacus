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
            # latency_data.append(150)
            continue
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
    "resnet50resnet152": 150,
    "resnet50inception_v3": 100,
    "resnet50vgg16": 50,
    "resnet50vgg19": 50,
    "resnet50bert": 75,
    "resnet101resnet152": 160,
    "resnet101inception_v3": 150,
    "resnet101vgg16": 90,
    "resnet101vgg19": 80,
    "resnet101bert": 130,
    "resnet152inception_v3": 150,
    "resnet152vgg16": 150,
    "resnet152vgg19": 150,
    "resnet152bert": 150,
    "inception_v3vgg16": 85,
    "inception_v3vgg19": 80,
    "inception_v3bert": 120,
    "vgg16vgg19": 35,
    "vgg16bert": 60,
    "vgg19bert": 60,
}

data_dir = "../results/A100/2in7/"


# %% tail latency caculator

for i in range(21):
    # print("--------------{}--------------".format(file_names[i]))
    abacus_latency = load_single_file(data_dir + "Abacus/{}.csv".format(file_names[i]))
    fcfs_latency = load_single_file(data_dir + "FCFS/{}.csv".format(file_names[i]))
    sjf_latency = load_single_file(data_dir + "SJF/{}.csv".format(file_names[i]))
    edf_latency = load_single_file(data_dir + "EDF/{}.csv".format(file_names[i]))

    abacus_tail = np.percentile(abacus_latency, 99)
    fcfs_tail = np.percentile(fcfs_latency, 99)
    sjf_tail = np.percentile(sjf_latency, 99)
    edf_tail = np.percentile(edf_latency, 99)

    print(
        "Abacus 99%-ile latency: {}, {} queries satified for {}".format(
            abacus_tail, abacus_latency.shape, colo_names[i]
        )
    )
    print(
        "FCFS 99%-ile latency: {}, {} queries satified for {}".format(
            fcfs_tail, fcfs_latency.shape, colo_names[i]
        )
    )
    print(
        "SJF 99%-ile latency: {}, {} queries satified for {}".format(
            sjf_tail, sjf_latency.shape, colo_names[i]
        )
    )
    print(
        "EDF 99%-ile latency: {}, {} queries satified for {}".format(
            edf_tail, edf_latency.shape, colo_names[i]
        )
    )

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

# %% load
