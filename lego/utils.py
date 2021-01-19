#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

import sys
import time


def timestamp(name, stage):
    print("TIMESTAMP, %s, %s, %f" % (name, stage, time.time()), file=sys.stderr)
