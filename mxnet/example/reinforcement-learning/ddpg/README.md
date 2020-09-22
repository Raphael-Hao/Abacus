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

# mx-DDPG
MXNet Implementation of DDPG

## /!\ This example depends on RLLAB which is deprecated /!\

# Introduction

This is the MXNet implementation of [DDPG](https://arxiv.org/abs/1509.02971). It is tested in the rllab cart pole environment against rllab's native implementation and achieves comparably similar results. You can substitute with this anywhere you use rllab's DDPG with minor modifications.

# Dependency

* rllab

# Usage

To run the algorithm, 

```python
python run.py
```

The implementation relies on rllab for environments and logging and the hyperparameters could be set in ```run.py```.

# References

1. [Lillicrap, Timothy P., et al. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 (2015)](https://arxiv.org/abs/1509.02971).
2. rllab. URL: [https://github.com/openai/rllab](https://github.com/openai/rllab)
