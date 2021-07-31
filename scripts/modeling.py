#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
# %%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Times New Roman"

fname = "../data/modeling/2in7results.csv"
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
    "all",
]


print(colo_names)
#%%

fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
ax.bar(
    x_0, lr_error, 0.3, label="Linear Regression", color="#ffeda0", edgecolor="black"
)
ax.bar(x_1, svm_error, 0.3, label="SVM", color="#feb24c", edgecolor="black")
ax.bar(x_2, mlp_error, 0.3, label="MLP", color="#f03b20", edgecolor="black")
ax.bar(x_3, all_mlp_error, 0.3, label="Cross Validation", edgecolor="black")
ax.set_xlim(-1, 22)

plt.legend(ncol=4, loc="upper center",fontsize=16)
plt.ylabel("Prediction Error",fontsize=18)
plt.xticks(x_1, colo_names, rotation=30,fontsize=14)
y_ticks = [0.0, 0.1,0.2,0.3]
ax.set_yticklabels(y_ticks, rotation=0, fontsize=14)
plt.tight_layout()
plt.savefig("../figure/prediction_error.pdf", bbox_inches="tight")

# %%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


mpl.rcParams["font.family"] = "Times New Roman"

fname = "../data/server/qos_rate.csv"
data = np.array(pd.read_csv(fname, header=None).values.tolist())

x_1= data[:,0]
y_1 = data[:,1]/100

fname = "../data/modeling/bs_latency.csv"
data = np.array(pd.read_csv(fname, header=None).values.tolist())

x_2 = data[:,0]
y_2 = data[:,1]

# %%
fig, axs = plt.subplots(1, 2, figsize=(10,3))
axs[1].plot(x_1, y_1, color="teal")
axs[0].plot(x_2, y_2, color="teal")

axs[1].set_xlim(1, 8)
axs[0].set_xlim(1, 16)
axs[1].set_xlabel("Searching Ways", fontsize=20)
axs[1].set_ylabel("QoS Guarantee(%)", fontsize=20)

import matplotlib.ticker as mtick
axs[1].yaxis.set_major_formatter(mtick.PercentFormatter(1,decimals=1))


# axs[0].set_ylim(0.12,0.25)
axs[0].set_ylim(0.12,0.25)
axs[1].tick_params(labelsize=18)
axs[0].tick_params(labelsize=18)

# labels = [item.get_text() for item in axs[1].get_yticklabels()]
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# empty_string_labels = ['']*len(labels)
# axs[1].set_yticklabels(empty_string_labels)
# plt.text(0, 20, "(a)")
axs[0].set_title("(a)",fontsize=20)
axs[1].set_title("(b)",fontsize=20)
axs[0].set_xlabel("Batch size", fontsize=20)
axs[0].set_ylabel("Latency(ms)", fontsize=20)
plt.tight_layout()
plt.savefig("../figure/bs_core_latency.pdf",bbox_inches="tight")


# %%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Times New Roman"

fname = "../data/server/tail_th.csv"
data = np.array(pd.read_csv(fname, header=0).values.tolist())

x_0 = [i - 0.25 for i in range(21)]
x_1 = [i for i in range(21)]
x_2 = [i + 0.25 for i in range(21)]

# %%
qos_target = data[0:21,1].astype(np.float32)
fcfs_tail = data[0:21, 2].astype(np.float32)/qos_target
sjf_tail = data[0:21, 3].astype(np.float32)/qos_target
abacus_tail = data[0:21, 4].astype(np.float32)/qos_target

print("Co-location pairs")
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


print(colo_names)
#%%
# #A1BAD4 #FFFEC6 #B7CBA3
fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
ax.bar(
    x_0, fcfs_tail, 0.25, label="FCFS", color="#A1BAD4", edgecolor="black"
)
ax.bar(x_1, sjf_tail, 0.25, label="SJF", color="#FFFEC6", edgecolor="black")
ax.bar(x_2, abacus_tail, 0.25, label="Abacus", color="#B7CBA3", edgecolor="black")
plt.axhline(y=1, color='r', linestyle='--')
ax.set_xlim(-0.5, 20.5)

ax.tick_params(labelsize=14)
plt.legend(ncol=4, loc="upper right",fontsize=16)
plt.ylabel("Normalized Tail Latency",fontsize=18)
ax.yaxis.set_label_coords(-0.02, 0.2)
plt.xticks(x_1, colo_names, rotation=30,fontsize=14)
plt.tight_layout()
plt.savefig("../figure/tail_latency.pdf", bbox_inches="tight")
plt.show()
# %%
import numpy as np
import glob
import pandas as pd
import os
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Times New Roman"

fname = "../data/server/tail_th.csv"
data = np.array(pd.read_csv(fname, header=0).values.tolist())

x_0 = [i - 0.25 for i in range(21)]
x_1 = [i for i in range(21)]
x_2 = [i + 0.25 for i in range(21)]

fcfs_th = data[0:21, 5].astype(np.float32)/10
sjf_th = data[0:21, 6].astype(np.float32)/10
abacus_th = data[0:21, 7].astype(np.float32)/10

print("Co-location pairs")
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


print(colo_names)
#%%
# #A1BAD4 #FFFEC6 #B7CBA3

fig = plt.figure(figsize=(16, 3))
ax = fig.add_subplot(111)
ax.bar(
    x_0, fcfs_th, 0.25, label="FCFS", color="#A1BAD4", edgecolor="black"
)
ax.bar(x_1, sjf_th, 0.25,label="SJF", color="#FFFEC6", edgecolor="black")
ax.bar(x_2, abacus_th, 0.25, label="Abacus", color="#B7CBA3", edgecolor="black")
# plt.axhline(y=1, color='r', linestyle='--')
ax.set_xlim(-0.5, 20.5)
ax.tick_params(labelsize=14)

plt.legend(ncol=4, loc="upper center",fontsize=16)
plt.ylabel("QPS",fontsize=18)
plt.xticks(x_1, colo_names, rotation=30,fontsize=14)
plt.tight_layout()
plt.savefig("../figure/throughput.pdf", bbox_inches="tight")
abacus_fcfs_improve = (abacus_th-fcfs_th)/fcfs_th
abacus_sjf_improve = (abacus_th-sjf_th)/sjf_th
print(np.mean(abacus_fcfs_improve))
print(np.mean(abacus_sjf_improve))
print(np.max(abacus_fcfs_improve))
print(np.max(abacus_sjf_improve))
# %%

import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

mpl.rcParams["font.family"] = "Times New Roman"


fname = "../data/modeling/core_latency.csv"
data = np.array(pd.read_csv(fname, header=None).values.tolist())

x_1 = data[:,0]
y_1 = data[:,1]


fig = plt.figure(figsize=(8, 3))

axs = fig.add_subplot(111)

axs.plot(x_1, y_1,color="teal")


axs.set_xlim(1, 28)

axs.set_xlabel("CPU cores", fontsize=20)
axs.set_ylabel("Latency(ms)", fontsize=20)
axs.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# axs.set_ylim(1,1)
axs.tick_params(labelsize=18)
plt.tight_layout()


plt.savefig("../figure/latency_core.pdf",bbox_inches="tight")

# %%

import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

mpl.rcParams["font.family"] = "Times New Roman"
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

fname = "../data/server/qos_ratio.csv"
data = np.array(pd.read_csv(fname, header=None).values.tolist())

y_1 = data[:,1].astype(np.float32)
y_2 = data[:,2].astype(np.float32)
y_3 = data[:,3].astype(np.float32)
z_1 = (y_3-y_1)
z_2 = (y_3-y_2)
print(np.mean(z_1))
print(np.mean(z_2))
print(np.max(z_1))
print(np.max(z_2))

x_1 = [i for i in range(1, 22)]
#%%

fig = plt.figure(figsize=(8, 4))

axs = fig.add_subplot(111)

axs.plot(x_1, y_1,label ="FCFS")
axs.plot(x_1, y_2,label="SJF")
axs.plot(x_1, y_3,color="red", label="Abacus")

# axs.plot(x_1, y_1,color="teal")
axs.set_ylim(0,1.2)

axs.set_xlim(1, 20)

# axs.set_xlabel("", fontsize=20)
axs.set_ylabel("QoS Guarantee(%)", fontsize=18)
# axs.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axs.yaxis.set_label_coords(-0.09, 0.2)
import matplotlib.ticker as mtick
axs.yaxis.set_major_formatter(mtick.PercentFormatter(1))
# axs.set_ylim(0.12,0.25)
# axs.yaxis.labelpad = 20
axs.tick_params(labelsize=16)
x_1 = [i for i in range(0, 21)]
plt.xticks(x_1, colo_names, rotation=90,fontsize=14)
plt.tight_layout()
plt.legend(ncol=4, loc="lower center",fontsize=16)
plt.savefig("../figure/qos_violation_ratio.pdf",bbox_inches="tight")
# %%

