#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: xxx

from matplotlib import pyplot as plt
import pandas as pd
import argparse
import numpy as np
import os

data_dir = "/state/partition/whcui/repository/project/Abacus/data/server/"
fig_path = "/state/partition/whcui/repository/project/Abacus/figure/"
# Set
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.direction"] = "in"

# Co-location, global
co_location = []

# parser = argparse.ArgumentParser()
# parser.add_argument("--mixed_option", default="2in7")  # mixed option
# parser.add_argument("--device", default="A100")  # device option
# parser.add_argument("--target", default="qos")  # target option
# args = parser.parse_args()


# target_option: 0 for qos, 1 for throughput, 2 for qos small
def csv_reader(file_path, target_option):
    global co_location

    data = pd.read_csv(file_path)
    if target_option == 0:
        # qos
        co_location = list(data["colocation"])
        FCFS_latency = list(data["FCFS_tail"])
        SJF_latency = list(data["SJF_tail"])
        EDF_latency = list(data["EDF_tail"])
        Abacus_latency = list(data["Abacus_tail"])
        # regularize the co_location list
        for i in range(len(co_location)):
            co_location[i] = "(" + co_location[i].replace("+", ",") + ")"
        return co_location, FCFS_latency, SJF_latency, EDF_latency, Abacus_latency
    if target_option == 1:
        # throughput
        co_location = list(data["colocation"])
        FCFS_latency = list(data["FCFS_throughput"])
        SJF_latency = list(data["SJF_throughput"])
        EDF_latency = list(data["EDF_throughput"])
        Abacus_latency = list(data["Abacus_throughput"])
        # regularize the co_location list
        for i in range(len(co_location)):
            co_location[i] = "(" + co_location[i].replace("+", ",") + ")"
        return co_location, FCFS_latency, SJF_latency, EDF_latency, Abacus_latency
    if target_option == 2:
        # qos
        co_location = list(data["colocation"])
        Abacus_latency = list(data["Abacus_tail"])
        # regularize the co_location list
        for i in range(len(co_location)):
            co_location[i] = "(" + co_location[i].replace("+", ",") + ")"
        return co_location, Abacus_latency


def qos_main():
    global co_location

    # csv reader
    co_location, FCFS_latency, SJF_latency, EDF_latency, Abacus_latency = csv_reader(
        data_dir + "7.2_qos/2in7/result.csv",
        0,
    )
    # Plot
    x = np.arange(len(co_location))
    FCFS_group = np.array(FCFS_latency)
    SJF_group = np.array(SJF_latency)
    EDF_group = np.array(EDF_latency)
    Abacus_group = np.array(Abacus_latency)

    # Size
    total_width, n = 0.85, 4
    width = total_width / n
    x = x + (total_width - width) / 4

    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 3.5)

    # Bar
    ax.bar(
        x,
        FCFS_group,
        width=width * 0.7,
        color="white",
        edgecolor="dimgrey",
        hatch="////",
        linewidth=1.0,
        label="FCFS",
        zorder=10,
    )
    ax.bar(
        x + width,
        SJF_group,
        width=width * 0.7,
        color="white",
        edgecolor="silver",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="SJF",
        zorder=10,
    )
    ax.bar(
        x + 2 * width,
        EDF_group,
        width=width * 0.7,
        color="white",
        edgecolor="lightseagreen",
        hatch="////",
        linewidth=1.0,
        label="EDF",
        zorder=10,
    )
    ax.bar(
        x + 3 * width,
        Abacus_group,
        width=width * 0.7,
        color="white",
        edgecolor="tomato",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="Abacus",
        zorder=10,
    )

    # Legend
    ax.legend(
        ncol=4,
        loc="best",
        fontsize=3,
        markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "roman"},
    )
    ax.grid(
        axis="y",
        color="red",
        zorder=0,
    )

    # Ticks
    x = x + (total_width - width) / 3
    y = [0.0, 0.5, 1.0, 1.5]
    plt.xticks(x, co_location, rotation=15, fontsize=9, weight="bold")
    plt.yticks(y, fontsize=10, weight="bold")
    plt.xlim(0, 21)
    plt.ylim(0, None)

    plt.ylabel("Normalized 99%-ile Latency", weight="bold")

    # Path
    file_path = fig_path + "7.2_qos.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()


def throughput_main():
    global co_location

    # csv reader
    (
        co_location,
        FCFS_throughput,
        SJF_throughput,
        EDF_throughput,
        Abacus_throughput,
    ) = csv_reader(
        data_dir + "7.3_throughput/2in7/result.csv",
        1,
    )
    # Plot
    x = np.arange(len(co_location))
    FCFS_group = np.array(FCFS_throughput) / 10
    SJF_group = np.array(SJF_throughput) / 10
    EDF_group = np.array(EDF_throughput) / 10
    Abacus_group = np.array(Abacus_throughput) / 10

    # Size
    total_width, n = 0.85, 4
    width = total_width / n
    x = x + (total_width - width) / 4

    fig, ax = plt.subplots()
    fig.set_size_inches(23.4, 3.5)

    # Bar
    ax.bar(
        x,
        FCFS_group,
        width=width * 0.7,
        color="white",
        edgecolor="dimgrey",
        hatch="////",
        linewidth=1.0,
        label="FCFS",
        zorder=10,
    )
    ax.bar(
        x + width,
        SJF_group,
        width=width * 0.7,
        color="white",
        edgecolor="silver",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="SJF",
        zorder=10,
    )
    ax.bar(
        x + 2 * width,
        EDF_group,
        width=width * 0.7,
        color="white",
        edgecolor="lightseagreen",
        hatch="////",
        linewidth=1.0,
        label="EDF",
        zorder=10,
    )
    ax.bar(
        x + 3 * width,
        Abacus_group,
        width=width * 0.7,
        color="white",
        edgecolor="tomato",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="Abacus",
        zorder=10,
    )

    # Legend
    ax.legend(
        ncol=4,
        loc="best",
        fontsize=3,
        markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "roman"},
    )
    ax.grid(
        axis="y",
        color="thistle",
        zorder=0,
    )

    # Ticks
    x = x + (total_width - width) / 3
    y = [0, 20, 40, 60, 80, 100]
    plt.xticks(x, co_location, rotation=15, fontsize=9, weight="bold")
    plt.yticks(y, fontsize=10, weight="bold")
    plt.xlim(0, 21)
    plt.ylim(0, None)

    plt.ylabel("QPS", weight="bold")

    # Path
    file_path = fig_path + "7.3_throughput.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()


def qos_small_main():
    global co_location

    # csv reader
    co_location, Abacus_latency = csv_reader(data_dir + "7.2_qos/small/result.csv", 2)
    # Plot
    x = np.arange(len(co_location))
    Abacus_group = np.array(Abacus_latency)

    # Size
    total_width, n = 0.85, 4
    width = total_width / n
    x = x + (total_width - width) / 4

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 3.5)

    # Bar
    ax.bar(
        x,
        Abacus_group,
        width=width * 3,
        color="white",
        edgecolor="tomato",
        hatch="////",
        linewidth=1.0,
        label="Abacus",
        zorder=10,
    )

    # Ticks
    x = x + (total_width - width) / 3
    y = [0.0, 0.25, 0.5, 0.75, 1.0]
    plt.xticks(x, co_location, rotation=25, fontsize=9, weight="bold")
    plt.yticks(y, fontsize=10, weight="bold")
    plt.xlim(-.5, 21)
    plt.ylim(0, None)

    plt.ylabel("Normalized 99%-ile Latency", weight="bold")

    # Path
    file_path = fig_path + "small_dnn.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()


def qos_main_3_in_4():
    cur_co_location = []

    # First add the 4in4 option
    (
        cur_co_location,
        FCFS_latency,
        SJF_latency,
        EDF_latency,
        Abacus_latency,
    ) = csv_reader(data_dir + "7.4_beyond_pair/qos/4in4/result.csv", 0)

    # csv reader
    (
        tmp_co_location,
        tmp_FCFS_latency,
        tmp_SJF_latency,
        tmp_EDF_latency,
        tmp_Abacus_latency,
    ) = csv_reader(
        data_dir + "7.4_beyond_pair/qos/3in4/result.csv",
        0,
    )

    cur_co_location = cur_co_location + tmp_co_location
    FCFS_latency = FCFS_latency + tmp_FCFS_latency
    SJF_latency = SJF_latency + tmp_SJF_latency
    EDF_latency = EDF_latency + tmp_EDF_latency
    Abacus_latency = Abacus_latency + tmp_Abacus_latency

    # Plot
    x = np.arange(len(cur_co_location))
    FCFS_group = np.array(FCFS_latency)
    SJF_group = np.array(SJF_latency)
    EDF_group = np.array(EDF_latency)
    Abacus_group = np.array(Abacus_latency)

    # Size
    total_width, n = 0.85, 4
    width = total_width / n
    x = x + (total_width - width) / 4

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 3.5)

    # Bar
    ax.bar(
        x,
        FCFS_group,
        width=width * 0.7,
        color="white",
        edgecolor="dimgrey",
        hatch="////",
        linewidth=1.0,
        label="FCFS",
        zorder=10,
    )
    ax.bar(
        x + width,
        SJF_group,
        width=width * 0.7,
        color="white",
        edgecolor="silver",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="SJF",
        zorder=10,
    )
    ax.bar(
        x + 2 * width,
        EDF_group,
        width=width * 0.7,
        color="white",
        edgecolor="lightseagreen",
        hatch="////",
        linewidth=1.0,
        label="EDF",
        zorder=10,
    )
    ax.bar(
        x + 3 * width,
        Abacus_group,
        width=width * 0.7,
        color="white",
        edgecolor="tomato",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="Abacus",
        zorder=10,
    )

    # Legend
    ax.legend(
        ncol=4,
        loc="best",
        fontsize=3,
        markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "roman"},
    )
    ax.grid(
        axis="y",
        color="red",
        zorder=0,
    )

    # Ticks
    x = x + (total_width - width) / 3
    y = [0.0, 0.5, 1.0, 1.5]
    plt.xticks(x, cur_co_location, rotation=15, fontsize=9, weight="bold")
    plt.yticks(y, fontsize=10, weight="bold")
    plt.xlim(0, 5)
    plt.ylim(0, None)

    plt.ylabel("Normalized 99%-ile Latency", weight="bold")

    # Path
    file_path = fig_path + "beyond_pair_qos.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()


def throughput_main_3_in_4():
    cur_co_location = []

    # First add the 4in4 option
    (
        cur_co_location,
        FCFS_throughput,
        SJF_throughput,
        EDF_throughput,
        Abacus_throughput,
    ) = csv_reader(data_dir + "7.4_beyond_pair/throughput/4in4/result.csv", 1)

    # csv reader
    (
        tmp_co_location,
        tmp_FCFS_throughput,
        tmp_SJF_throughput,
        tmp_EDF_throughput,
        tmp_Abacus_throughput,
    ) = csv_reader(
        data_dir + "7.4_beyond_pair/throughput/3in4/result.csv",
        1,
    )

    cur_co_location = cur_co_location + tmp_co_location
    FCFS_throughput = FCFS_throughput + tmp_FCFS_throughput
    SJF_throughput = SJF_throughput + tmp_SJF_throughput
    EDF_throughput = EDF_throughput + tmp_EDF_throughput
    Abacus_throughput = Abacus_throughput + tmp_Abacus_throughput

    # Plot
    x = np.arange(len(cur_co_location))
    FCFS_group = np.array(FCFS_throughput) / 10
    SJF_group = np.array(SJF_throughput) / 10
    EDF_group = np.array(EDF_throughput) / 10
    Abacus_group = np.array(Abacus_throughput) / 10

    # Size
    total_width, n = 0.85, 4
    width = total_width / n
    x = x + (total_width - width) / 4

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 3.5)

    # Bar
    ax.bar(
        x,
        FCFS_group,
        width=width * 0.7,
        color="white",
        edgecolor="dimgrey",
        hatch="////",
        linewidth=1.0,
        label="FCFS",
        zorder=10,
    )
    ax.bar(
        x + width,
        SJF_group,
        width=width * 0.7,
        color="white",
        edgecolor="silver",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="SJF",
        zorder=10,
    )
    ax.bar(
        x + 2 * width,
        EDF_group,
        width=width * 0.7,
        color="white",
        edgecolor="lightseagreen",
        hatch="////",
        linewidth=1.0,
        label="EDF",
        zorder=10,
    )
    ax.bar(
        x + 3 * width,
        Abacus_group,
        width=width * 0.7,
        color="white",
        edgecolor="tomato",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="Abacus",
        zorder=10,
    )

    # Legend
    ax.legend(
        ncol=4,
        loc="best",
        fontsize=3,
        markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "roman"},
    )
    ax.grid(
        axis="y",
        color="red",
        zorder=0,
    )

    # Ticks
    x = x + (total_width - width) / 3
    y = [0, 20, 40, 60, 80, 100]
    plt.xticks(x, cur_co_location, rotation=15, fontsize=9, weight="bold")
    plt.yticks(y, fontsize=10, weight="bold")
    plt.xlim(0, 5)
    plt.ylim(0, None)

    plt.ylabel("QPS", weight="bold")

    # Path
    file_path = fig_path + "beyond_pair_throughput.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()


def qos_main_mig():
    cur_co_location = []

    # First add the 1in4 option
    (
        cur_co_location,
        FCFS_latency,
        SJF_latency,
        EDF_latency,
        Abacus_latency,
    ) = csv_reader(data_dir + "7.5_mig/qos/1in4/result.csv", 0)
    SJF_latency[0], Abacus_latency[0] = 0, 0
    # Then add the 2in4 option
    (
        tmp_co_location,
        tmp_FCFS_latency,
        tmp_SJF_latency,
        tmp_EDF_latency,
        tmp_Abacus_latency,
    ) = csv_reader(data_dir + "7.5_mig/qos/2in4/result.csv", 0)
    cur_co_location = cur_co_location + tmp_co_location
    FCFS_latency = FCFS_latency + tmp_FCFS_latency
    SJF_latency = SJF_latency + tmp_SJF_latency
    EDF_latency = EDF_latency + tmp_EDF_latency
    Abacus_latency = Abacus_latency + tmp_Abacus_latency
    # Then add the 4in4 option
    (
        tmp_co_location,
        tmp_FCFS_latency,
        tmp_SJF_latency,
        tmp_EDF_latency,
        tmp_Abacus_latency,
    ) = csv_reader(data_dir + "7.5_mig/qos/4in4/result.csv", 0)
    cur_co_location = cur_co_location + tmp_co_location
    FCFS_latency = FCFS_latency + tmp_FCFS_latency
    SJF_latency = SJF_latency + tmp_SJF_latency
    EDF_latency = EDF_latency + tmp_EDF_latency
    Abacus_latency = Abacus_latency + tmp_Abacus_latency

    # Plot
    x = np.arange(len(cur_co_location))
    FCFS_group = np.array(FCFS_latency)
    SJF_group = np.array(SJF_latency)
    EDF_group = np.array(EDF_latency)
    Abacus_group = np.array(Abacus_latency)

    # Size
    total_width, n = 0.85, 4
    width = total_width / n
    x = x + (total_width - width) / 4

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 3.5)

    # Bar
    ax.bar(
        x,
        FCFS_group,
        width=width * 0.7,
        color="white",
        edgecolor="dimgrey",
        hatch="////",
        linewidth=1.0,
        label="FCFS",
        zorder=10,
    )
    ax.bar(
        x + width,
        SJF_group,
        width=width * 0.7,
        color="white",
        edgecolor="silver",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="SJF",
        zorder=10,
    )
    ax.bar(
        x + 2 * width,
        EDF_group,
        width=width * 0.7,
        color="white",
        edgecolor="lightseagreen",
        hatch="////",
        linewidth=1.0,
        label="EDF",
        zorder=10,
    )
    ax.bar(
        x + 3 * width,
        Abacus_group,
        width=width * 0.7,
        color="white",
        edgecolor="tomato",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="Abacus",
        zorder=10,
    )

    # Legend
    ax.legend(
        ncol=4,
        loc="best",
        fontsize=3,
        markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "roman"},
    )
    ax.grid(
        axis="y",
        color="red",
        zorder=0,
    )

    # Ticks
    x = x + (total_width - width) / 3
    y = [0.0, 0.5, 1.0, 1.5]
    plt.xticks(x, cur_co_location, rotation=15, fontsize=9, weight="bold")
    plt.yticks(y, fontsize=10, weight="bold")
    plt.xlim(0, 5)
    plt.ylim(0, None)

    plt.ylabel("Normalized 99%-ile Latency", weight="bold")

    # Path
    file_path = fig_path + "mig_qos.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()


def throughput_main_mig():
    cur_co_location = []

    # First add the 1in4 option
    (
        cur_co_location,
        FCFS_throughput,
        SJF_throughput,
        EDF_throughput,
        Abacus_throughput,
    ) = csv_reader(
        data_dir + "7.5_mig/throughput/1in4/result.csv", 1
    )
    SJF_throughput[0], Abacus_throughput[0] = 0, 0
    # Then add the 2in4 option
    (
        tmp_co_location,
        tmp_FCFS_throughput,
        tmp_SJF_throughput,
        tmp_EDF_throughput,
        tmp_Abacus_throughput,
    ) = csv_reader(
        data_dir + "7.5_mig/throughput/2in4/result.csv", 1
    )
    cur_co_location = cur_co_location + tmp_co_location
    FCFS_throughput = FCFS_throughput + tmp_FCFS_throughput
    SJF_throughput = SJF_throughput + tmp_SJF_throughput
    EDF_throughput = EDF_throughput + tmp_EDF_throughput
    Abacus_throughput = Abacus_throughput + tmp_Abacus_throughput
    # Then add the 4in4 option
    (
        tmp_co_location,
        tmp_FCFS_throughput,
        tmp_SJF_throughput,
        tmp_EDF_throughput,
        tmp_Abacus_throughput,
    ) = csv_reader(
        data_dir + "7.5_mig/throughput/4in4/result.csv", 1
    )
    cur_co_location = cur_co_location + tmp_co_location
    FCFS_throughput = FCFS_throughput + tmp_FCFS_throughput
    SJF_throughput = SJF_throughput + tmp_SJF_throughput
    EDF_throughput = EDF_throughput + tmp_EDF_throughput
    Abacus_throughput = Abacus_throughput + tmp_Abacus_throughput

    # Plot
    x = np.arange(len(cur_co_location))
    FCFS_group = np.array(FCFS_throughput) / 10
    SJF_group = np.array(SJF_throughput) / 10
    EDF_group = np.array(EDF_throughput) / 10
    Abacus_group = np.array(Abacus_throughput) / 10

    # Size
    total_width, n = 0.85, 4
    width = total_width / n
    x = x + (total_width - width) / 4

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 3.5)

    # Bar
    ax.bar(
        x,
        FCFS_group,
        width=width * 0.7,
        color="white",
        edgecolor="dimgrey",
        hatch="////",
        linewidth=1.0,
        label="FCFS",
        zorder=10,
    )
    ax.bar(
        x + width,
        SJF_group,
        width=width * 0.7,
        color="white",
        edgecolor="silver",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="SJF",
        zorder=10,
    )
    ax.bar(
        x + 2 * width,
        EDF_group,
        width=width * 0.7,
        color="white",
        edgecolor="lightseagreen",
        hatch="////",
        linewidth=1.0,
        label="EDF",
        zorder=10,
    )
    ax.bar(
        x + 3 * width,
        Abacus_group,
        width=width * 0.7,
        color="white",
        edgecolor="tomato",
        hatch="\\\\\\\\",
        linewidth=1.0,
        label="Abacus",
        zorder=10,
    )

    # Legend
    ax.legend(
        ncol=4,
        loc="best",
        fontsize=3,
        markerscale=3,
        labelspacing=0.1,
        edgecolor="black",
        shadow=False,
        fancybox=False,
        handlelength=0.8,
        handletextpad=0.6,
        columnspacing=0.8,
        prop={"weight": "roman"},
    )
    ax.grid(
        axis="y",
        color="red",
        zorder=0,
    )

    # Ticks
    x = x + (total_width - width) / 3
    y = [0, 20, 40, 60, 80, 100]
    plt.xticks(x, cur_co_location, rotation=15, fontsize=9, weight="bold")
    plt.yticks(y, fontsize=10, weight="bold")
    plt.xlim(0, 5)
    plt.ylim(0, None)

    plt.ylabel("QPS", weight="bold")

    # Path
    file_path = fig_path + "mig_throughput.pdf"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path, bbox_inches="tight")
    # plt.show()

