#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

#%%
import numpy as np
import glob
import pandas as pd
import os

def load_all_std_deviation(
    data_path="../data/profile/A100/3in4",
):
    all_std_deviation = None
    for filename in glob.glob(os.path.join(data_path, "*.csv")):
        # print(filename)
        # filepath = os.path.join(data_path, filename)
        data = pd.read_csv(filename, header=0)
        data = data.values.tolist()
        total_data_num = len(data)
        print("{} samples loaded from {}".format(total_data_num, filename))
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
    data_path="../data/profile/A100/3in4",
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


#%% 2in7
all_std_deviation = load_all_std_deviation("../data/profile/A100/2in7") / 1000
all_latency = load_all_lantency("../data/profile/A100/2in7") / 1000

#%% 3in4
all_std_deviation = load_all_std_deviation("../data/profile/A100/3in4") / 1000
all_latency = load_all_lantency("../data/profile/A100/3in4") / 1000

#%% 4in4
all_std_deviation = load_all_std_deviation("../data/profile/A100/4in4") / 1000
all_latency = load_all_lantency("../data/profile/A100/4in4") / 1000

#%% 2in4
all_std_deviation = load_all_std_deviation("../data/profile/mig/2in4") / 1000
all_latency = load_all_lantency("../data/profile/mig/2in4") / 1000

# %%

print("{} samples are collected".format(len(all_std_deviation)))
print("80%-ile std: {}".format(np.percentile(all_std_deviation, 80)))
print("90%-ile std: {}".format(np.percentile(all_std_deviation, 90)))
print("99%-ile std: {}".format(np.percentile(all_std_deviation, 99)))
print("mean std: {}".format(np.mean(all_std_deviation)))
print("max std: {}".format(np.max(all_std_deviation)))
print("min std: {}".format(np.min(all_std_deviation)))


# %%
print("80%-ile e2e: {}".format(np.percentile(all_latency, 80)))
print("90%-ile std: {}".format(np.percentile(all_latency, 90)))
print("99%-ile std: {}".format(np.percentile(all_latency, 99)))
print("mean std: {}".format(np.mean(all_latency)))
print("max std: {}".format(np.max(all_latency)))
print("min std: {}".format(np.min(all_latency)))

# %%
print("ratio of average e2e to std: {}".format(np.mean(all_std_deviation / all_latency)))

# %%
import seaborn as sns
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "Times New Roman"

# %%
fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
ax.set_xlabel("Standard deviation of latencies(ms)", fontsize=32)
ax.set_ylabel("PDF", fontsize=32)
ax.tick_params("x", labelsize=28)
ax.tick_params("y", labelsize=28)
# ax.set_xlim(0,5)
# sns.ecdfplot(all_std_deviation)
sns.kdeplot(all_std_deviation, shade=True)
plt.tight_layout()
# sns.distplot(all_std_deviation, shade=True)
plt.savefig("../figure/std_deviation.pdf", bbox_inches="tight")
plt.show()

# %%

fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
import matplotlib.ticker as mtick

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel("End-to-end latencies(ms)", fontsize=32)
ax.set_ylabel("CDF", fontsize=32)
ax.tick_params("x", labelsize=28)
ax.tick_params("y", labelsize=28)

sns.ecdfplot(all_latency)
plt.tight_layout()

plt.savefig("../figure/profiled_latency.pdf", bbox_inches="tight")
plt.show(block=False)

# %%

fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
import matplotlib.ticker as mtick

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel(
    "End-to-end latencies / Standard deviation of latencies (ms)", fontsize=32
)
ax.set_ylabel("CDF", fontsize=32)
ax.tick_params("x", labelsize=28)
ax.tick_params("y", labelsize=28)

sns.ecdfplot(all_latency, label="e2e", legend=True, linewidth=3)
sns.ecdfplot(all_std_deviation, label="std", legend=True, linewidth=3)
plt.legend(ncol=2, loc="upper right", fontsize=28)
plt.tight_layout()
plt.savefig("../figure/latency_cdf.pdf", bbox_inches="tight")
plt.show(block=False)


# %%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"


def load_all_unpred_latency(
    data_path="../data/profile/mps",
):
    all_unpred_latency = None
    for filename in glob.glob(os.path.join(data_path, "*.csv")):
        # filepath = os.path.join(data_path, filename)
        data = pd.read_csv(filename, header=0)
        data = data.values.tolist()
        total_data_num = len(data)
        print("{} samples loaded from {}".format(total_data_num, filename))
        data = np.array(data)
        n = data.shape[0]
        unpred_latency = []
        for i in range(n):
            line = data[i]
            unpred_latency.append(line[-1].astype(float) / 1000)
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

fig = plt.figure(figsize=(10, 3))
ax = fig.add_subplot(111)
import matplotlib.ticker as mtick

ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel("Latency(ms)", fontsize=20)
ax.set_ylabel("CDF", fontsize=20)
ax.tick_params("x", labelsize=18)
ax.tick_params("y", labelsize=18)
sns.ecdfplot(all_unpred_latency[5], label="Resnet50", linewidth=2)
# sns.ecdfplot(all_unpred_latency[1],label = "Resnet101")
sns.ecdfplot(all_unpred_latency[3], label="Resnet152", linewidth=2)
sns.ecdfplot(all_unpred_latency[6], label="Inception_v3", linewidth=2)
sns.ecdfplot(all_unpred_latency[2], label="VGG16", linewidth=2)
sns.ecdfplot(all_unpred_latency[0], label="VGG19", linewidth=2)
sns.ecdfplot(all_unpred_latency[4], label="Bert", linewidth=2)
plt.legend(fontsize=16, ncol=2, loc=(0.5, 0), frameon=False)
plt.savefig("../figure/unpredictable_latency.pdf", bbox_inches="tight")
plt.show()

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
# %%
