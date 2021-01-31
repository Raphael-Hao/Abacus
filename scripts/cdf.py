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
    data_path="/home/cwh/Lego/data/profile/2in7",
):
    all_std_deviation = None
    for filename in glob.glob(os.path.join(data_path, "*.csv")):
        std_deviation_data= load_single_file(
            os.path.join(data_path, filename)
        )
        if all_std_deviation is None:
            all_std_deviation = std_deviation_data
        else:
            all_std_deviation = np.concatenate((all_std_deviation, std_deviation_data))
    return all_std_deviation.astype(np.float)


all_std_deviation = load_all_std_deviation()
#%%

print(np.max(all_std_deviation))
print(np.min(all_std_deviation))

# %%

import seaborn as sns
sns.ecdfplot(all_std_deviation)

# %%
