#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /plot.py
# \brief:
# Author: raphael hao

#%%
from percentile import data_preprocess

data_preprocess("qos", "mig", "1in4")
data_preprocess("qos", "mig", "2in4")
data_preprocess("qos", "mig", "4in4")
#%%
from plotter import qos_main_mig

qos_main_mig()

# %%
