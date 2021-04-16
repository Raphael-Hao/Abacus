# Abacus

This repository contains the source code for a research paper.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Abacus](#abacus)
  - [What is Abacus](#what-is-abacus)
  - [Environment Preparation](#environment-preparation)
  - [Getting Started](#getting-started)
    - [Profiling](#profiling)
    - [Training Predictor](#training-predictor)
    - [Online Serving](#online-serving)
  - [Evaluation](#evaluation)
    - [Ensuring QoS](#ensuring-qos)
    - [Improving Peak Throughput](#improving-peak-throughput)
    - [Beyongd Pair-wise Co-location](#beyongd-pair-wise-co-location)
    - [Integrating with MIGs](#integrating-with-migs)
    - [Effectiveness of Multi-way Search](#effectiveness-of-multi-way-search)

<!-- /code_chunk_output -->

## What is Abacus

**Abacus** is a runtime system that runs multiple DNN queries simultaneously with stable and predictable latency. **Abacus** enables deterministic operator overlap to enforce the latency predictability. **Abacus** is comprised of an overlap-aware latency predictor, a headroom-based query controller, and segmental model executors. The latency predictor is able to precisely predict the latencies of queries when the operator overlap is determined. The query controller determines the appropriate operator overlap to guarantee the QoS of all the DNN services on a GPU. The model executors run the operators as needed to support the deterministic operator overlap. Our evaluation using seven popular DNNs on an Nvidia A100 GPU shows that **Abacus** significantly reduces the QoS violation and improves the throughput compared with state-of-the-art solutions.

## Environment Preparation

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

  3. Nvidia GPU related affairs: switch MIG and MPS

  ```shell
  $ chmod +x nvidia_utils.sh
  $ # switch mig, 0: disable, 1: enable
  $ ./nvidia_utils.sh --mig 0/1
  $ # create GPU instances
  $ ./nvidia_utils.sh --cgi 1/2/4
  $ # delete GPU instances
  $ ./nvidia_utils.sh --dgi
  $ # switch mps with mig disabled, 0: disable, 1: enable
  $ ./nvidia_utils.sh --mps 0/1
  $ # switch mps with mig enabled, 0: disable, 1: enable
  $ ./nvidia_utils.sh --mps 0/1 --mig 1/2/4
  ```

## Getting Started

The following sections step through the things required to run **Abacus**

### Profiling

**Abacus** needs to profile the essential data for training a precise overlap-aware latency predictor.

We first profiling the data without MPS enabled and MIG disabled.

- Profiling for pair-wise co-location on a dedicated A100.
  ```shell
  $ python main.py --task profile --model_num 2
  ```
- Profiling for triplet-wise co-location on a dedicated A100.
  ```shell
  $ python main.py --task profile --model_num 3
  ```
- Profiling for quadruplet-wise co-location on a dedicated A100.
  ```shell
  $ python main.py --task profile --model_num 4
  ```

We then profiling the data without MPS and MIG both enabled.

- Profiling for pair-wise co-location on a _MIG 2g.10gb_ of A100.
  ```shell
  $ python main.py --task profile --model_num 2 --mig 2
  ```
- Profiling for quadruplet-wise co-location on a _MIG 4g.10gb_ of A100.
  ```shell
  $ python main.py --task profile --model_num 4 --mig 1
  ```

### Training Predictor

After obataining all the profiling data, we train the latency predictor for each cases.

- Training the predictor for pair-wise co-location on a dedicated A100.
  ```shell
  $ python main.py --task train --model_num 2 --mode all --modeling mlp
  ```
- Training the predictor for triplet-wise co-location on a dedicated A100.
  ```shell
  $ python main.py --task train --model_num 3 --mode all --modeling mlp
  ```
- Training the predictor for quadruplet-wise co-location on a dedicated A100.
  ```shell
  $ python main.py --task train --model_num 4 --mode all --modeling mlp
  ```
- Training the predictor for pair-wise co-location on a _MIG 2g.10gb_ of A100.
  ```shell
  $ python main.py --task train --model_num 2 --mode all --modeling mlp --mig 2
  ```
- Training the predictor for quadruplet-wise co-location on a _MIG 4g.20gb_ of A100.
  ```shell
  $ python main.py --task train --model_num 4 --mode all --modeling mlp --mig 1
  ```

### Online Serving

After profiling and training, we can serve multiple DNN services with **Abacus**

- Testing **Abacus** for pair-wise co-location on a dedicated A100, take the setup of `model id: 0, 1; qos_target: 50ms; total tested queries: 1000; search ways: 2` as an example.
  ```shell
  $ python main.py --task serve --model_num 2 --comb 0 1 --policy Abacus --load 50 --qos 50 --queries 1000 --thld 5 --ways 2 --abandon
  ```
- Testing **Abacus** for triplet-wise co-location on a dedicated A100, take the setup of `model id: 0, 1, 3; qos_target: 50ms; total tested queries: 1000; search ways: 2` as an example.
  ```shell
  $ python main.py --task serve --model_num 2 --comb 0 1 2 --policy Abacus --load 50 --qos 50 --queries 1000 --thld 5 --ways 2 --abandon
  ```
- Testing **Abacus** for quadruplet-wise co-location on a dedicated A100, take the setup of `model id: 0, 1, 2, 3; qos_target: 50ms; total tested queries: 1000; search ways: 2` as an example.
  ```shell
  $ python main.py --task serve --model_num 2 --comb 0 1 2 3 --policy Abacus --load 50 --qos 50 --queries 1000 --thld 5 --ways 2 --abandon
  ```
- Testing **Abacus** for pair-wise co-location on a _MIG 2g.10gb_ of A100, take the setup of `model id: 0, 1; qos_target: 50ms; total tested queries: 1000; search ways: 2` as an example.
  ```shell
  $ python main.py --task serve --model_num 2 --comb 0 1 --policy Abacus --load 50 --qos 50 --queries 1000 --thld 5 --ways 2 --abandon --mig 2
  ```
- Testing **Abacus** for pair-wise co-location on a _MIG 4g.20gb_ of A100, take the setup of `model id: 0, 1, 2, 3; qos_target: 50ms; total tested queries: 1000; search ways: 2` as an example.
  ```shell
  $ python main.py --task serve --model_num 2 --comb 0 1 --policy Abacus --load 50 --qos 50 --queries 1000 --thld 5 --ways 2 --abandon --mig 1
  ```

## Evaluation

All evaluations are conducted in the root directory of **Abacus**. The following table presents the detailed information of each experiments, including the corresponding figures in paper and the shell scripts for running the experiment.

|  Experiment Name/ Section/ Paragraph  | Related Figures |     Scripts Location     |
| :-----------------------------------: | :-------------: | :----------------------: |
|           7.2 Ensuring QoS            | Figure 13 & 14  |     experiment/0_qos     |
|     7.3 Improving Peak Throughput     |    Figure 15    | experiment/1_throughput  |
|   7.4 Beyongd Pair-wise Co-location   | Figure 16 & 17  | experiment/2_beyond_pair |
|       7.5 Integrating with MIGs       | Figure 18 & 19  |     experiment/3_mig     |
| 7.6 Effectiveness of Multi-way Search |    Figure 20    |  experiment/3_multiway   |

### Ensuring QoS

- We first evaluate the QoS guarantee in terms of tail latency and QoS violation ratio.

  ```shell
  $ ./experiments/0_qos/2in7.sh
  ```

  The following figure depicts the 99%-ile latency for each pair-wise co-location.

  |                         <img src="figure/2in7_qos.png" width="1000">                          |
  | :-------------------------------------------------------------------------------------------: |
  | **End-to-end 99%-ile latency of each pair-wise co-location with FCFS, SJF, EDF, and Abacus.** |

  The following figure depicts the QoS violation ratio for each pair-wise co-location.

  |                    <img src="figure/qos_violation_ratio.png" width="400">                    |
  | :------------------------------------------------------------------------------------------: |
  | **QoS violation ratio of each pair-wise co-location with<br /> FCFS, SJF, EDF, and Abacus.** |

### Improving Peak Throughput

- We then evaluate the peak throughput under the load that exceeds the hardware limit.

  ```shell
  $ ./experiments/1_throughput/2in7.sh
  ```

  The following figure depicts the peak throughput for each pair-wise co-location.

  |                             <img src="figure/2in7_throughput.png" width="1000"/>                             |
  | :----------------------------------------------------------------------------------------------------------: |
  | **The peak throughput of each co-location pair with FCFS, SJF, EDF, and Abacus while guaranteeing the QoS.** |

### Beyongd Pair-wise Co-location

- We also evaluate **Abacus** beyond pair-wise co-location.

  ```shell
  $ ./experiments/2_beyond_pair/qos/3in4.sh
  $ ./experiments/2_beyond_pair/qos/4in4.sh
  $ ./experiments/2_beyond_pair/throughput/3in4.sh
  $ ./experiments/2_beyond_pair/throughput/3in4.sh
  ```

  The following two figures presents the performance of **Abacus** for triplet-wise and quadruplet-wise co-location in terms of QoS guarantee and peak throughput.

  |                               <img src="figure/3in4_qos.png" width="580">                               |                          <img src="figure/3in4_throughput.png" width="580">                          |
  | :-----------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------: |
  | **The 99%-ile latency in triplets- and quadruplets- wise deployments with FCFS, SJF, EDF, and Abacus.** | **Peak throughputs in triplets- and quadruplets- wise deployments with FCFS, SJF, EDF, and Abacus.** |

### Integrating with MIGs

- We also evaluate **Abacus** with MIG enabled.

  ```shell
  $ ./experiments/mig/qos/1in4.sh
  $ ./experiments/mig/qos/2in4.sh
  $ ./experiments/mig/qos/4in4.sh
  $ ./experiments/mig/throughput/1in4.sh
  $ ./experiments/mig/throughput/2in4.sh
  $ ./experiments/mig/throughput/3in4.sh
  ```

  The following two figures presents the performance of **Abacus** for *MIG 1g.10gb* and *MIG 2g.20gb* in terms of QoS guarantee and peak throughput.


|          <img src="figure/mig_qos.png" width="580">           |       <img src="figure/mig_throughput.png" width="580">        |
| :-----------------------------------------------------------: | :------------------------------------------------------------: |
| **The 99%-ile latency of the co-located services with MIGs.** | **The peak throughputs of the co-located services with MIGs.** |

### Effectiveness of Multi-way Search

- We also evaluate the effectiveness of multi-way search


|                     <img src="figure/bs_core_latency.png" width="360">                      |
| :-----------------------------------------------------------------------------------------: |
| **Duration of determining an appropriate operator<br /> group with different search ways.** |
