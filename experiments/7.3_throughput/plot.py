#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /plot.py
# \brief:
# Author: raphael hao

#%%
from percentile import data_preprocess

data_preprocess("throughput", "A100", "2in7")
#%%
from plotter import throughput_main

throughput_main()

# %%
