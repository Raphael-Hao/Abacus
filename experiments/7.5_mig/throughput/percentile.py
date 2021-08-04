#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
# %%
import numpy as np
import pandas as pd

data_path = "/state/partition/whcui/repository/project/Abacus/"

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
def select_experiment_data(target, platform, model_config, model_size="normal"):
    if target == "qos" and platform == "A100" and model_config == "2in7":
        colo_names = [
            "Res50+Res101",
            "Res50+Res152",
            "Res50+IncepV3",
            "Res50+VGG16",
            "Res50+VGG19",
            "Res50+Bert",
            "Res101+Res152",
            "Res101+IncepV3",
            "Res101+VGG16",
            "Res101+VGG19",
            "Res101+Bert",
            "Res152+IncepV3",
            "Res152+VGG16",
            "Res152+VGG19",
            "Res152+Bert",
            "IncepV3+VGG16",
            "IncepV3+VGG19",
            "IncepV3+Bert",
            "VGG16+VGG19",
            "VGG16+Bert",
            "VGG19+Bert",
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
        if model_size == "normal":
            qos_target = {
                "resnet50resnet101": 100,
                "resnet50resnet152": 150,
                "resnet50inception_v3": 100,
                "resnet50vgg16": 40,
                "resnet50vgg19": 40,
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
                "inception_v3vgg16": 80,
                "inception_v3vgg19": 80,
                "inception_v3bert": 80,
                "vgg16vgg19": 20,
                "vgg16bert": 60,
                "vgg19bert": 60,
            }
            data_dir = data_path + "data/server/7.2_qos/2in7/"
        elif model_size == "small":
            qos_target = {
                "resnet50resnet101": 65,
                "resnet50resnet152": 95,
                "resnet50inception_v3": 70,
                "resnet50vgg16": 35,
                "resnet50vgg19": 35,
                "resnet50bert": 55,
                "resnet101resnet152": 100,
                "resnet101inception_v3": 100,
                "resnet101vgg16": 55,
                "resnet101vgg19": 65,
                "resnet101bert": 95,
                "resnet152inception_v3": 110,
                "resnet152vgg16": 95,
                "resnet152vgg19": 110,
                "resnet152bert": 110,
                "inception_v3vgg16": 55,
                "inception_v3vgg19": 55,
                "inception_v3bert": 65,
                "vgg16vgg19": 15,
                "vgg16bert": 45,
                "vgg19bert": 50,
            }
            data_dir = data_path + "data/server/7.2_qos/small/"
        else:
            raise NotImplementedError

    elif target == "qos" and platform == "A100" and model_config == "3in4":
        colo_names = [
            "Res101+Res152+VGG19",
            "Res101+Res152+Bert",
            "Res101+VGG19+Bert",
            "Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19",
            "resnet101resnet152bert",
            "resnet101vgg19bert",
            "resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19": 120,
            "resnet101resnet152bert": 150,
            "resnet101vgg19bert": 85,
            "resnet152vgg19bert": 90,
        }

        data_dir = data_path + "data/server/7.4_beyond_pair/qos/3in4/"
    elif target == "qos" and platform == "A100" and model_config == "4in4":
        colo_names = [
            "Res101+Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19bert": 120,
        }

        data_dir = data_path + "data/server/7.4_beyond_pair/qos/4in4/"

    elif target == "qos" and platform == "mig" and model_config == "4in4":
        colo_names = [
            "Res101+Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19bert": 130,
        }

        data_dir = data_path + "data/server/7.5_mig/qos/4in4/"

    elif target == "qos" and platform == "mig" and model_config == "2in4":
        colo_names = [
            "Res101Bert_Res152VGG19",
            "Res101Res152_VGG19Bert",
            "Res101VGG19_Resn152Bert",
        ]

        file_names = [
            "resnet101bert_resnet152vgg19",
            "resnet101resnet152_vgg19bert",
            "resnet101vgg19_resnet152bert",
        ]

        qos_target = {
            "resnet101bert_resnet152vgg19": 130,
            "resnet101resnet152_vgg19bert": 130,
            "resnet101vgg19_resnet152bert": 130,
        }

        data_dir = data_path + "data/server/7.5_mig/qos/2in4/"

    elif target == "qos" and platform == "mig" and model_config == "1in4":
        colo_names = [
            "Res101+Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19bert": 130,
        }

        data_dir = data_path + "data/server/7.5_mig/qos/1in4/"

    elif target == "throughput" and platform == "A100" and model_config == "2in7":
        colo_names = [
            "Res50+Res101",
            "Res50+Res152",
            "Res50+IncepV3",
            "Res50+VGG16",
            "Res50+VGG19",
            "Res50+Bert",
            "Res101+Res152",
            "Res101+IncepV3",
            "Res101+VGG16",
            "Res101+VGG19",
            "Res101+Bert",
            "Res152+IncepV3",
            "Res152+VGG16",
            "Res152+VGG19",
            "Res152+Bert",
            "IncepV3+VGG16",
            "IncepV3+VGG19",
            "IncepV3+Bert",
            "VGG16+VGG19",
            "VGG16+Bert",
            "VGG19+Bert",
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
            "inception_v3vgg16": 80,
            "inception_v3vgg19": 80,
            "inception_v3bert": 80,
            "vgg16vgg19": 30,
            "vgg16bert": 60,
            "vgg19bert": 60,
        }

        data_dir = data_path + "data/server/7.3_throughput/2in7/"
    elif target == "throughput" and platform == "A100" and model_config == "3in4":
        colo_names = [
            "Res101+Res152+VGG19",
            "Res101+Res152+Bert",
            "Res101+VGG19+Bert",
            "Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19",
            "resnet101resnet152bert",
            "resnet101vgg19bert",
            "resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19": 100,
            "resnet101resnet152bert": 100,
            "resnet101vgg19bert": 100,
            "resnet152vgg19bert": 100,
        }

        data_dir = data_path + "data/server/7.4_beyond_pair/throughput/3in4/"
    elif target == "throughput" and platform == "A100" and model_config == "4in4":
        colo_names = [
            "Res101+Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19bert": 100,
        }

        data_dir = data_path + "data/server/7.4_beyond_pair/throughput/4in4/"
    elif target == "throughput" and platform == "mig" and model_config == "4in4":
        colo_names = [
            "Res101+Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19bert": 130,
        }

        data_dir = data_path + "data/server/7.5_mig/throughput/4in4/"

    elif target == "throughput" and platform == "mig" and model_config == "2in4":
        colo_names = [
            "Res101Bert_Res152VGG19",
            "Res101Res152_VGG19Bert",
            "Res101VGG19_Resn152Bert",
        ]

        file_names = [
            "resnet101bert_resnet152vgg19",
            "resnet101resnet152_vgg19bert",
            "resnet101vgg19_resnet152bert",
        ]

        qos_target = {
            "resnet101bert_resnet152vgg19": 130,
            "resnet101resnet152_vgg19bert": 130,
            "resnet101vgg19_resnet152bert": 130,
        }

        data_dir = data_path + "data/server/7.5_mig/throughput/2in4/"

    elif target == "throughput" and platform == "mig" and model_config == "1in4":
        colo_names = [
            "Res101+Res152+VGG19+Bert",
        ]

        file_names = [
            "resnet101resnet152vgg19bert",
        ]

        qos_target = {
            "resnet101resnet152vgg19bert": 130,
        }

        data_dir =data_path + "data/server/7.5_mig/throughput/1in4/"
    else:
        raise NotImplementedError
    return colo_names, file_names, qos_target, data_dir


def data_preprocess(target, platform, model_config, model_size="normal"):
    import csv

    colo_names, file_names, qos_target, data_dir = select_experiment_data(
        target=target,
        platform=platform,
        model_config=model_config,
        model_size=model_size,
    )
    result_filepath = data_dir + "result.csv"
    result_file = open(result_filepath, "w+")
    csv_writer = csv.writer(result_file, dialect="excel")
    result_header = [
        "colocation",
        "FCFS_tail",
        "FCFS_throughput",
        "FCFS_violation",
        "SJF_tail",
        "SJF_throughput",
        "SJF_violation",
        "EDF_tail",
        "EDF_throughput",
        "EDF_violation",
        "Abacus_tail",
        "Abacus_throughput",
        "Abacus_violation",
        # "qos_target",
    ]
    csv_writer.writerow(result_header)

    for i in range(len(colo_names)):
        # print("--------------{}--------------".format(file_names[i]))
        abacus_latency = load_single_file(
            data_dir + "Abacus/{}.csv".format(file_names[i])
        )
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

        abacus_vio = abacus_latency[abacus_latency < qos_target[file_names[i]]].shape[0]
        fcfs_vio = fcfs_latency[fcfs_latency < qos_target[file_names[i]]].shape[0]
        sjf_vio = sjf_latency[sjf_latency < qos_target[file_names[i]]].shape[0]
        edf_vio = edf_latency[edf_latency < qos_target[file_names[i]]].shape[0]

        abacus_load = abacus_latency.shape[0]
        fcfs_load = fcfs_latency.shape[0]
        sjf_load = sjf_latency.shape[0]
        edf_load = edf_latency.shape[0]

        abacus_vio_ratio = 1 - abacus_vio / 1000
        fcfs_vio_ratio = 1 - fcfs_vio / 1000
        sjf_vio_ratio = 1 - sjf_vio / 1000
        edf_vio_ratio = 1 - edf_vio / 1000

        print(
            "throughput improvement: {:.3f}, {:.3f}, {:.3f}".format(
                abacus_load / fcfs_load - 1,
                abacus_load / sjf_load - 1,
                abacus_load / edf_load - 1,
            )
        )
        csv_writer.writerow(
            [
                colo_names[i],
                fcfs_tail / qos_target[file_names[i]],
                fcfs_load,
                fcfs_vio_ratio,
                sjf_tail / qos_target[file_names[i]],
                sjf_load,
                sjf_vio_ratio,
                edf_tail / qos_target[file_names[i]],
                edf_load,
                edf_vio_ratio,
                abacus_tail / qos_target[file_names[i]],
                abacus_load,
                abacus_vio_ratio,
                # qos_target[file_names[i]],
            ]
        )
    result_file.flush()


def small_data_preprocess(target, platform, model_config, model_size="normal"):
    import csv

    colo_names, file_names, qos_target, data_dir = select_experiment_data(
        target=target,
        platform=platform,
        model_config=model_config,
        model_size=model_size,
    )
    result_filepath = data_dir + "result.csv"
    result_file = open(result_filepath, "w+")
    csv_writer = csv.writer(result_file, dialect="excel")
    result_header = [
        "colocation",
        "Abacus_tail",
        "Abacus_throughput",
        "Abacus_violation",
        # "qos_target",
    ]
    csv_writer.writerow(result_header)

    for i in range(len(colo_names)):
        # print("--------------{}--------------".format(file_names[i]))
        abacus_latency = load_single_file(data_dir + "/{}.csv".format(file_names[i]))
        abacus_tail = np.percentile(abacus_latency, 99)

        print(
            "Abacus 99%-ile latency: {}, {} queries satified for {}".format(
                abacus_tail, abacus_latency.shape, colo_names[i]
            )
        )

        abacus_vio = abacus_latency[abacus_latency < qos_target[file_names[i]]].shape[0]

        abacus_load = abacus_latency.shape[0]

        abacus_vio_ratio = 1 - abacus_vio / 1000

        csv_writer.writerow(
            [
                colo_names[i],
                abacus_tail / qos_target[file_names[i]],
                abacus_load,
                abacus_vio_ratio,
            ]
        )
    result_file.flush()


if __name__ == "__main__":
    # A100 qos
    # data_preprocess("qos", "A100", "2in7")
    # data_preprocess("qos", "A100", "3in4")
    # data_preprocess("qos", "A100", "4in4")
    # # A100 throughput
    # data_preprocess("throughput", "A100", "2in7")
    # data_preprocess("throughput", "A100", "3in4")
    # data_preprocess("throughput", "A100", "4in4")
    # # mig qos
    # data_preprocess("qos", "mig", "1in4")
    # data_preprocess("qos", "mig", "2in4")
    # data_preprocess("qos", "mig", "4in4")
    # mig qos
    # data_preprocess("throughput", "mig", "1in4")
    # data_preprocess("throughput", "mig", "2in4")
    # data_preprocess("throughput", "mig", "4in4")
    # small DNN
    small_data_preprocess("qos", "A100", "2in7", "small")