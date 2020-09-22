<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Bayesian Methods
================

This folder contains examples related to Bayesian Methods.

We currently have *Stochastic Gradient Langevin Dynamics (SGLD)* [<cite>(Welling and Teh, 2011)</cite>](http://www.icml-2011.org/papers/398_icmlpaper.pdf)
and *Bayesian Dark Knowledge (BDK)* [<cite>(Balan, Rathod, Murphy and Welling, 2015)</cite>](http://papers.nips.cc/paper/5965-bayesian-dark-knowledge).

**sgld.ipynb** shows how to use MXNet to repeat the toy experiment in the original SGLD paper.

**bdk.ipynb** shows how to use MXNet to implement the DistilledSGLD algorithm in Bayesian Dark Knowledge.

**bdk_demo.py** contains scripts (more than the notebook) related to Bayesian Dark Knowledge. Use `python bdk_demo.py -d 1 -l 2 -t 50000` to run classification on MNIST. 

View parameters we can use with the following command.

```shell
python bdk_demo.py -h


usage: bdk_demo.py [-h] [-d DATASET] [-l ALGORITHM] [-t TRAINING] [--gpu GPU]

Examples in the paper [NIPS2015]Bayesian Dark Knowledge and [ICML2011]Bayesian
Learning via Stochastic Gradient Langevin Dynamics

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset to use. 0 --> TOY, 1 --> MNIST, 2 -->
                        Synthetic Data in the SGLD paper
  -l ALGORITHM, --algorithm ALGORITHM
                        Type of algorithm to use. 0 --> SGD, 1 --> SGLD,
                        other-->DistilledSGLD
  -t TRAINING, --training TRAINING
                        Number of training samples
  --gpu GPU             0 to use GPU, not set to use CPU
```
