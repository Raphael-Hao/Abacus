#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /abacus.py
# \brief:
# Author: raphael hao
from abacus.loadbalancer.base import LoadBalancer


class AbacusLoadBalancer(LoadBalancer):
    def __init__(self) -> None:
        super().__init__()