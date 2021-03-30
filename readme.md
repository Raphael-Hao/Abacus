# Abacus
[TOC]

This repository contains the source code for a research paper.

# What is Abacus
Abacus is a runtime system that runs multiple DNN queries simultaneously with stable and predictable latency. Abacus enables deterministic operator overlap to enforce the latency predictability. Abacus is comprised of an overlap-aware latency predictor, a headroom-based query controller, and segmental model executors. The latency predictor is able to precisely predict the latencies of queries when the operator overlap is determined. The query controller determines the appropriate operator overlap to guarantee the QoS of all the DNN services on a GPU. The model executors run the operators as needed to support the deterministic operator overlap. Our evaluation using seven popular DNNs on an Nvidia A100 GPU shows that Abacus significantly reduces the QoS violation and improves the throughput compared with state-of-the-art solutions.

# Environment Preparation
- Hardware&software requirements
  1. Hardware Requirements
     1. CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
     2. Memroy: 252G
     3. NVIDIA Ampere 100

  2. Software Requirements
     1. Ubuntu 20.04.1 (Kernel 5.8.0)
     2. GPU Driver: 460.39
     3. CUDA 11.2
     4. CUDNN 8.1
     5. Anaconda3-2020.7
     6. Pytorch 1.8.0

- Preparing Python environment
  1. Install Anaconda3 as the Python runtime
    ```shell
    $ cd ~ && mkdir -p .local && cd .local
    $ wget -O Anaconda3-2020.07-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
    $ chmod +x Anaconda3-2020.07-Linux-x86_64.sh
    $ ./Anaconda3-2020.07-Linux-x86_64.sh -b -p ../.local/anaconda3
    ```
  2. Activate conda and create python environment with essential dependencies
    ```shell
    $ eval "$($HOME/.local/anaconda3/bin/conda shell.zsh hook)"
    $ # cd into Abacus repository
    $ conda create --name Abacus
    $ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
    $ pip install -r requirements.txt
    ```
  3. Start MPS server for Abacus
    ```shell
    $ # cd into Abacus repository
    $ chmod +x mps.sh
    $ ./mps.sh start
    ```
# Getting Started

The following sections step through the things required to run Abacus

## Profiling
Abacus needs to profile the essential data for training a precise overlap-aware latency predictor.
```shell
$ python main.py --task profile
```
## Training Predictor

```
$ python main.py --task train
```

## Online Serving

```
$ python main.py --task serve
```

# Evaluation
| Experiment Name/ Section/ Paragraph | Related Figures | Experiment |
| ----------------------------------- | --------------- | ---------- |
|   7.2                                  |                 |            |
|   7.3                                  |                 |            |
|   7.4                                  |                 |            |
|   7.5                                  |                 |            |
|   7.6                                  |                 |            |
