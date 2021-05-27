#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /cluster_abacus.py
# \brief: 
# Author: raphael hao

# %%
import pandas as pd
import numpy as np
import glob
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Times New Roman"

loads = [
    8357,
    9005,
    8908,
    8771,
    8624,
    8393,
    10345,
    9490,
    9388,
    9498,
    9108,
    9337,
    10606,
    10682,
    10338,
    10069,
    9691,
    8936,
    8723,
    8951,
    8796,
    9293,
    9207,
    8999,
    8779,
    8731,
    9265,
    9325,
    9206,
    8913,
    8840,
    8752,
    8958,
    8712,
    8974,
    8430,
    9704,
    9170,
    8902,
    8954,
    8668,
    8986,
    8846,
    8640,
    8437,
    8944,
    9248,
    8851,
    8725,
    8645,
    8627,
    8929,
    8809,
    8850,
    8823,
    8873,
    9179,
    8522,
    8737,
    8851,
    8689,
    8538,
    8702,
    8683,
    8726,
    8780,
    10904,
    9764,
    9295,
    9504,
    9509,
    9663,
    10498,
    10480,
    10450,
    10264,
    10107,
    9409,
    8987,
    8920,
    8719,
    8863,
    8931,
    9015,
    9117,
    8812,
    9545,
    9038,
    8698,
    9091,
    8579,
    9014,
    8794,
    8621,
    8876,
    8839,
    9782,
    9011,
    8772,
    9180,
    8875,
    9124,
    8721,
    8875,
    8732,
    8770,
    9435,
    8944,
    8914,
    8793,
    8701,
    9013,
    8768,
    8887,
    8621,
    9190,
    9231,
    9021,
    8781,
    8905,
]

abacus_src_path = "/home/whcui/project/Abacus/results/cluster/Abacus"
abacus_raw_file = "/home/whcui/project/Abacus/results/cluster/Abacus_raw.csv"
abacus_clean_file = "/home/whcui/project/Abacus/results/cluster/Abacus_clean.csv"

clock_src_path = "/home/whcui/project/Abacus/results/cluster/Clock"
clock_raw_file = "/home/whcui/project/Abacus/results/cluster/Clock_raw.csv"
clock_clean_file = "/home/whcui/project/Abacus/results/cluster/Clock_clean.csv"

# %% merge abacus
def merge_abacus(src_path,dst_file):
    file_cnt = 0
    for filename in glob.glob(src_path+ "/**/*.csv", recursive=True):
        print("file: {} loaded".format(filename))
        df = pd.read_csv(filename,header=0)
        if file_cnt ==0:
            df.to_csv(dst_file, index=False)
            file_cnt+=1
        else:
            df.to_csv(dst_file, index=False, header=False, mode="a+")

merge_abacus(src_path=abacus_src_path,dst_file=abacus_raw_file)

# %% merge clock
def merge_clock(src_path,dst_file):
    file_cnt = 0
    for filename in glob.glob(src_path+ "/*.csv", recursive=True):
        print("file: {} loaded".format(filename))
        df = pd.read_csv(filename,header=0)
        if file_cnt ==0:
            df.to_csv(dst_file, index=False)
            file_cnt+=1
        else:
            df.to_csv(dst_file, index=False, header=False, mode="a+")

merge_clock(src_path=clock_src_path,dst_file=clock_raw_file)

# %% preprocess abacus data
def abacus_preprocess(src_name,dst_name):
    data = pd.read_csv(src_name, header=0)
    print(len(data))
    data_resorted = data.sort_values("query_id")
    data_removed = data_resorted[data_resorted.query_id != 0]
    data_removed = data_removed.reset_index(drop=True)
    print(data_removed)
    start_index = 0
    total_query = len(data_removed)
    total_load = sum(loads)
    for i in range(120):
        # print("load id: {}".format(i))
        end_index = int(start_index + total_query * loads[i] / total_load)
        # print("end_index: {}".format(end_index))
        if i == 119:
            end_index = total_query
        data_removed.loc[start_index:end_index, "load_id"] = i
        start_index = end_index
    print(data_removed)
    data_removed.to_csv(dst_name, index=False)
abacus_preprocess(abacus_raw_file,abacus_clean_file)

#%% preprocess clock data
def clock_preprocess(src_name, dst_name):
    data = pd.read_csv(src_name, header=0)
    data_resorted = data.sort_values("query_id")
    data_removed = data_resorted[data_resorted.query_id != 0]
    data_removed = data_removed.reset_index(drop=True)
    data_removed.to_csv(dst_name, index=False)
clock_preprocess(clock_raw_file,clock_clean_file)

# %%

def get_abacus_data(src_df):
    src_df = src_df[src_df["latency"]!=-1]
    src_df = src_df[src_df["latency"] < 105]
    return src_df

def get_load(src_df):
    load_df = src_df[src_df["latency"]!=-1]
    return len(load_df)/60*16


def get_clock_data(src_df):
    src_df = src_df[src_df["latency"]!=-1]
    src_df = src_df[src_df["latency"] < 103]
    return src_df

def get_abacus_latency(file_name):
    data = pd.read_csv(file_name,header=0)
    data_plot = []
    for i in range(120):
        df = data[data["load_id"] == i]
        cur_load = get_load(df)
        df = get_abacus_data(df)
        cur_lat_99 = df.latency.quantile(0.99)
        cur_lat_mean = df.latency.mean()
        print("99%-ile: {}, mean: {}, load: {}".format(cur_lat_99, cur_lat_mean, cur_load))
        if i == 0:
            data_plot = np.array([[cur_lat_99,cur_lat_mean,cur_load]])
        else:
            data_plot = np.append(data_plot,np.array([[cur_lat_99,cur_lat_mean,cur_load]]), axis = 0)
    return data_plot
def get_clock_latency(file_name):
    data = pd.read_csv(file_name,header=0)
    data_plot = []
    for i in range(120):
        if i== 0:
            df = data[data["load_id"] == 1]
        elif i == 1:
            df = data[data["load_id"] == 2]
        else:
            df = data[data["load_id"] == i]
        df = get_clock_data(df)
        cur_load = get_load(df)
        cur_lat_99 = df.latency.quantile(0.99)
        cur_lat_mean = df.latency.mean()
        print("99%-ile: {}, mean: {}, load: {}".format(cur_lat_99, cur_lat_mean, cur_load))
        if i == 0:
            data_plot = np.array([[cur_lat_99,cur_lat_mean,cur_load]])
        else:
            data_plot = np.append(data_plot,np.array([[cur_lat_99,cur_lat_mean,cur_load]]), axis = 0)
    return data_plot

abacus_data=get_abacus_latency(abacus_clean_file)
# print(abacus_data)
clock_data=get_clock_latency(clock_clean_file)
# print(clock_data)
# %%

x = [i for i in range(120)]

abacus_99 = abacus_data[:,0]
abacus_mean = abacus_data[:,1]
abacus_load = abacus_data[:,2]

clock_99 = clock_data[:,0]
clock_mean = clock_data[:,1]
clock_load = clock_data[:,2]

fig,axs=plt.subplots(3,1, figsize=(10,6))

axs[0].plot(x,loads)

axs[1].plot(x,abacus_99)
axs[1].plot(x,clock_99)
axs[2].plot(x,abacus_mean)
axs[2].plot(x,clock_mean)
axs[0].plot(x,abacus_load)
axs[0].plot(x,clock_load)

axs[0].set_xlim(0,119)
axs[0].set_ylim(7000,11000)
axs[1].set_xlim(0,119)
# axs[1].set_ylim(0,102)
axs[2].set_xlim(0,119)

plt.show()
plt.tight_layout()
plt.savefig("../figure/large_scale.png", bbox_inches="tight")
# %%
