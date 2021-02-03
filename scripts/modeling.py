#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"

fname = "../data/modeling/result.csv"
data = pd.read_csv(fname, header=0)
