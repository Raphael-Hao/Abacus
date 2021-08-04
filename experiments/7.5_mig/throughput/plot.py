#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /plot.py
# \brief:
# Author: raphael hao

#%%
from percentile import data_preprocess

data_preprocess("throughput", "mig", "1in4")
data_preprocess("throughput", "mig", "2in4")
data_preprocess("throughput", "mig", "4in4")
#%%
from plotter import throughput_main_mig

throughput_main_mig()
# %%
