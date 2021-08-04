#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /plot.py
# \brief:
# Author: raphael hao

#%%
from percentile import data_preprocess

data_preprocess("qos", "A100", "2in7")
#%%
from plotter import qos_main

qos_main()
# %%

#%%
from percentile import small_data_preprocess

small_data_preprocess("qos", "A100", "2in7", "small")
# %%

from plotter import qos_small_main

qos_small_main()
# %%
