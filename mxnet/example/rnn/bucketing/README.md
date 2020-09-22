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

RNN Example
===========
This folder contains RNN examples using high level mxnet.rnn interface.

--------------

## Gluon Implementation

You can check this improved [Gluon implementation](http://gluon-nlp.mxnet.io/model_zoo/language_model/index.html#word-language-model) in gluon-nlp, the largest LSTM model reaches a perplexity of 65.62.

--------------


## Data
1) Review the license for the Sherlock Holmes dataset and ensure that you agree to it. Then uncomment the lines in the 'get_sherlockholmes_data.sh' script that download the dataset.

2) Run `get_sherlockholmes_data.sh` to download Sherlock Holmes data.

## Python

- Generate the Sherlock Holmes language model by using LSTM:

  For Python2 (CPU support): can take 2+ hours on AWS-EC2-p2.16xlarge

      $ python  [lstm_bucketing.py](lstm_bucketing.py) 

  For Python3 (CPU support): can take 2+ hours on AWS-EC2-p2.16xlarge

      $ python3 [lstm_bucketing.py](lstm_bucketing.py) 

  Assuming your machine has 4 GPUs and you want to use all the 4 GPUs:

  For Python2 (GPU support only): can take 50+ minutes on AWS-EC2-p2.16xlarge

      $ python [cudnn_rnn_bucketing.py](cudnn_rnn_bucketing.py) --gpus 0,1,2,3

  For Python3 (GPU support only): can take 50+ minutes on AWS-EC2-p2.16xlarge

      $ python3 [cudnn_rnn_bucketing.py](cudnn_rnn_bucketing.py) --gpus 0,1,2,3

- To run the mixed precision inference for the trained model, you should use the `--dtype`.

  This uses AMP conversion API for bucketing module to convert to a mixed precision module.

    $ python [cudnn_rnn_bucketing.py](cudnn_rnn_bucketing.py) --gpus 0 --model-prefix saved_rnn_model --load-epoch 12 --test --dtype float16


### Performance Note:

More ```MXNET_GPU_WORKER_NTHREADS``` may lead to better performance. For setting ```MXNET_GPU_WORKER_NTHREADS```, please refer to [Environment Variables](https://mxnet.apache.org/api/faq/env_var).

