#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao
import torch
from test.server import TestScheduler

if __name__ == "__main__":
    torch.set_printoptions(linewidth=200)
    test_scheduler = TestScheduler()
    test_scheduler.test_query_feature()