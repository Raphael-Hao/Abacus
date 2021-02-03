#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
# %%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"

fname = "../data/modeling/results.csv"
data = np.array(pd.read_csv(fname, header=0).values.tolist())

fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
x_0 = [i - 0.3 for i in range(29)]
x_1 = [i for i in range(29)]
x_2 = [i + 0.3 for i in range(29)]
x_3 = [28.6]
mlp_error = data[0:29, 3].astype(np.float32)
print(np.mean(mlp_error[0:28]))
lr_error = data[30:59, 3].astype(np.float32)
print(np.mean(lr_error[0:28]))
svm_error = data[59:88, 3].astype(np.float32)
print(np.mean(svm_error[0:28]))
all_mlp_error = data[29, 3].astype(np.float32)
print(mlp_error)

ax.bar(x_0, lr_error, 0.3)
ax.bar(x_1, svm_error, 0.3)
ax.bar(x_2, mlp_error, 0.3)
ax.bar(x_3, all_mlp_error, 0.3)
ax.set_xlim(-1, 29)
plt.savefig("prediction_error.pdf", bbox_inches="tight")
# %%
