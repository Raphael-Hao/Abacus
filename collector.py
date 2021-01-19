#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: raphael hao

from lego.network.resnet import resnet101
from torch.cuda.streams import Stream
import torch.nn as nn
import torch
import torch.multiprocessing as mp
from lego.network.resnet import BasicBlock, Bottleneck
import datetime

cuda = torch.device("cuda")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
streams = [Stream(0) for _ in range(2)]

inputs = [torch.rand(1, 3, 224, 224).half().cuda() for _ in range(2)]
weights = [
    torch.nn.Parameter(torch.rand(64, 3, 3, 3).half(), requires_grad=False).cuda()
    for _ in range(2)
]

models = [resnet101().half().cuda() for _ in range(2)]
# models = [torch.nn.Conv2d(3, 64, 3).cuda() for _ in range(2)]


for model in models:
    model.eval()


for i in range(2):
    with torch.cuda.stream(streams[i]):
        for j in range(10):
            # torch.conv2d(inputs[i], weight=weights[i])
            models[i](inputs[i])

start_event = torch.cuda.Event(enable_timing=True)
end_events = [torch.cuda.Event(enable_timing=False) for _ in range(2)]
end_event = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()

# single stream

print("starting: {}".format(datetime.datetime.now()))
start_event.record(torch.cuda.default_stream())
for i in range(2):
    with torch.cuda.stream(streams[0]):
        for j in range(10):
            # torch.conv2d(inputs[i], weight=weights[i])
            models[i](inputs[i])
            # print(
            #     "stream: {}, cnt: {}, executing: {}".format(0, i, datetime.datetime.now())
            # )
end_events[0].record(streams[0])
torch.cuda.default_stream().wait_event(end_events[0])
end_event.record(torch.cuda.default_stream())
print("before sync: {}".format(datetime.datetime.now()))
end_event.synchronize()
print("after sync: {}".format(datetime.datetime.now()))
elapsed_time_ms = start_event.elapsed_time(end_event)
print("single stream: {}".format(elapsed_time_ms/20))


# multi stream
print("starting: {}".format(datetime.datetime.now()))
start_event.record(torch.cuda.default_stream())
for i in range(2):
    with torch.cuda.stream(streams[i]):
        for j in range(10):
            # torch.conv2d(inputs[i], weight=weights[i])
            models[i](inputs[i])
            # print(
            #     "stream: {}, cnt: {}, executing: {}".format(0, i, datetime.datetime.now())
            # )
end_events[0].record(streams[0])
end_events[1].record(streams[1])
torch.cuda.default_stream().wait_event(end_events[0])
torch.cuda.default_stream().wait_event(end_events[1])
end_event.record(torch.cuda.default_stream())
print("before sync: {}".format(datetime.datetime.now()))
end_event.synchronize()
print("after sync: {}".format(datetime.datetime.now()))
elapsed_time_ms = start_event.elapsed_time(end_event)
print("multi stream: {}".format(elapsed_time_ms/20))



if __name__ == "__main_":
    pass
