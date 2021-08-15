#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /plot.py
# \brief: 
# Author: raphael hao

# %%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

fname = "../../data/modeling/2in7results.csv"
data = np.array(pd.read_csv(fname, header=0).values.tolist())

x_0 = [i - 0.3 for i in range(22)]
x_1 = [i for i in range(22)]
x_2 = [i + 0.3 for i in range(22)]
x_3 = [21.6]
mlp_error = data[0:22, 3].astype(np.float32)
print(np.mean(mlp_error[0:21]))
lr_error = data[23:45, 3].astype(np.float32)
print(np.mean(lr_error[0:21]))
svm_error = data[45:, 3].astype(np.float32)
print(np.mean(svm_error[0:28]))
all_mlp_error = data[22, 3].astype(np.float32)
print(mlp_error)

print("Co-location pairs")
colo_names = [
    "(Res50,Res101)",
    "(Res50,Res152)",
    "(Res50,IncepV3)",
    "(Res50,VGG16)",
    "(Res50,VGG19)",
    "(Res50,bert)",
    "(Res101,Res152)",
    "(Res101,IncepV3)",
    "(Res101,VGG16)",
    "(Res101,VGG19)",
    "(Res101,Bert)",
    "(Res152,IncepV3)",
    "(Res152,VGG16)",
    "(Res152,VGG19)",
    "(Res152,bert)",
    "(IncepV3,VGG16)",
    "(IncepV3,VGG19)",
    "(IncepV3,bert)",
    "(VGG16,VGG19)",
    "(VGG16,bert)",
    "(VGG19,bert)",
    "all",
]


print(colo_names)
#%%
mpl.rcParams["hatch.linewidth"] = 1.0
fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
ax.bar(
    x_0,
    lr_error,
    0.3,
    label="Linear Regression",
    color="none",
    hatch="////////",
    edgecolor="#4B4B4B",
)
ax.bar(
    x_1,
    svm_error,
    0.3,
    label="SVM",
    color="none",
    hatch="\\\\\\\\\\\\\\\\",
    edgecolor="#9F9F9F",
)
ax.bar(
    x_2, mlp_error, 0.3, label="MLP", color="none", hatch="////////", edgecolor="red"
)
ax.bar(
    x_3,
    all_mlp_error,
    0.3,
    color="none",
    hatch="\\\\\\\\\\\\\\\\",
    label="Cross Validation",
    edgecolor="#138D8C",
)
ax.set_xlim(-0.5, 21.75)

plt.legend(ncol=4, loc="upper center", fontsize=16)
plt.ylabel("Prediction Error", fontsize=18)
plt.xticks(x_1, colo_names, rotation=30, fontsize=14)
y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4]
ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.tight_layout()
plt.savefig("../../figure/prediction_error.png", bbox_inches="tight")
plt.show()
# %%
