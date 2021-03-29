#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao


#%%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"


def load_single_file(filepath):
    data = pd.read_csv(filepath, header=0)
    data = data.values.tolist()
    total_data_num = len(data)
    print("{} samples loaded from {}".format(total_data_num, filepath))
    data = np.array(data)
    n = data.shape[0]
    std_deviation = []
    for i in range(n):
        line = data[i]
        std_deviation.append(line[-1])
    std_deviation = np.array(std_deviation)
    return std_deviation

def load_all_std_deviation(
    data_path="/state/partition/whcui/repository/project/Abacus/data/profile/A100/2in7",
):
    all_std_deviation = None
    for filename in glob.glob(os.path.join(data_path, "*.csv")):
        print(filename)
        # filepath = os.path.join(data_path, filename)
        # print(filepath)
        data = pd.read_csv(filename, header=0)
        data = data.values.tolist()
        total_data_num = len(data)
        # print("{} samples loaded from {}".format(total_data_num, filepath))
        data = np.array(data)
        n = data.shape[0]
        std_deviation = []
        for i in range(n):
            line = data[i]
            std_deviation.append(line[-1])
        std_deviation_data = np.array(std_deviation)
        if all_std_deviation is None:
            all_std_deviation = std_deviation_data
        else:
            all_std_deviation = np.concatenate((all_std_deviation, std_deviation_data))
    return all_std_deviation.astype(np.float)
def load_all_lantency(
    data_path="/state/partition/whcui/repository/project/Abacus/data/profile/A100/2in7",
):
    all_std_deviation = None
    for filename in glob.glob(os.path.join(data_path, "*.csv")):
        print(filename)
        # filepath = os.path.join(data_path, filename)
        # print(filepath)
        data = pd.read_csv(filename, header=0)
        data = data.values.tolist()
        total_data_num = len(data)
        # print("{} samples loaded from {}".format(total_data_num, filepath))
        data = np.array(data)
        n = data.shape[0]
        std_deviation = []
        for i in range(n):
            line = data[i]
            std_deviation.append(line[-2])
        std_deviation_data = np.array(std_deviation)
        if all_std_deviation is None:
            all_std_deviation = std_deviation_data
        else:
            all_std_deviation = np.concatenate((all_std_deviation, std_deviation_data))
    return all_std_deviation.astype(np.float)
all_latency = load_all_lantency()
all_std_deviation = load_all_std_deviation()

#%%
print(all_latency.shape)
print(np.percentile(all_std_deviation, 80))
print(np.percentile(all_std_deviation, 90))
print(np.percentile(all_std_deviation, 99))
print(np.mean(all_std_deviation))
print(np.max(all_std_deviation))
print(np.min(all_std_deviation))

#%%
print(np.percentile(all_latency, 80))
print(np.percentile(all_latency, 90))
print(np.percentile(all_latency, 99))
print(np.mean(all_latency))
print(np.max(all_latency))
print(np.min(all_latency))

# %%

import seaborn as sns

sns.ecdfplot(all_std_deviation)
sns.ecdfplot(all_latency)
plt.savefig("std_deviation.pdf", bbox_inches="tight")
# %%

import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"


def load_all_unpred_latency(
    data_path="/state/partition/whcui/repository/project/Abacus/data/profile/mps",
):
    all_unpred_latency = None
    for filename in glob.glob(os.path.join(data_path, "*.csv")):
        filepath = os.path.join(data_path, filename)
        data = pd.read_csv(filepath, header=0)
        data = data.values.tolist()
        total_data_num = len(data)
        print("{} samples loaded from {}".format(total_data_num, filepath))
        data = np.array(data)
        n = data.shape[0]
        unpred_latency = []
        for i in range(n):
            line = data[i]
            unpred_latency.append(line[-1].astype(float)/1000)
        unpred_latency = np.array([unpred_latency])
        if all_unpred_latency is None:
            all_unpred_latency = unpred_latency
        else:
            all_unpred_latency = np.concatenate(
                (all_unpred_latency, unpred_latency), axis=0
            )
    return all_unpred_latency.astype(np.float)
all_unpred_latency = load_all_unpred_latency()

# %%
import seaborn as sns
sns.ecdfplot(all_unpred_latency[0])
sns.ecdfplot(all_unpred_latency[1])
sns.ecdfplot(all_unpred_latency[2])
sns.ecdfplot(all_unpred_latency[3])
sns.ecdfplot(all_unpred_latency[4])
sns.ecdfplot(all_unpred_latency[5])
sns.ecdfplot(all_unpred_latency[6])
print(np.max(all_unpred_latency[0]))
print(np.max(all_unpred_latency[1]))
print(np.max(all_unpred_latency[2]))
print(np.max(all_unpred_latency[3]))
print(np.max(all_unpred_latency[4]))
print(np.max(all_unpred_latency[5]))
print(np.max(all_unpred_latency[6]))
print(np.min(all_unpred_latency[0]))
print(np.min(all_unpred_latency[1]))
print(np.min(all_unpred_latency[2]))
print(np.min(all_unpred_latency[3]))
print(np.min(all_unpred_latency[4]))
print(np.min(all_unpred_latency[5]))
print(np.min(all_unpred_latency[6]))
print(np.sum(all_unpred_latency > 30))
plt.savefig("unpredictable_latency.pdf", bbox_inches="tight")
# %%

import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"