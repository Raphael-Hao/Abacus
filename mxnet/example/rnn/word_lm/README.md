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

Word Level Language Modeling
===========
This example trains a multi-layer LSTM on Sherlock Holmes language modeling benchmark.

The following techniques have been adopted for SOTA results:
- [LSTM for LM](https://arxiv.org/pdf/1409.2329.pdf)
- [Weight tying](https://arxiv.org/abs/1608.05859) between word vectors and softmax output embeddings

## Prerequisite
The example requires MXNet built with CUDA.

## Data
The Sherlock Holmes data is a copyright free copy of Sherlock Holmes from[(Project Gutenberg)](http://www.gutenberg.org/cache/epub/1661/pg1661.txt):

## Usage
Example runs and the results:

```
python train.py --tied --nhid 650 --emsize 650 --dropout 0.5        # Test ppl of 44.26
```

```
usage: train.py [-h] [--data DATA] [--emsize EMSIZE] [--nhid NHID]
                [--nlayers NLAYERS] [--lr LR] [--clip CLIP] [--epochs EPOCHS]
                [--batch_size BATCH_SIZE] [--dropout DROPOUT] [--tied]
                [--bptt BPTT] [--log-interval LOG_INTERVAL] [--seed SEED]

Sherlock Holmes LSTM Language Model

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping by global norm
  --epochs EPOCHS       upper epoch limit
  --batch_size BATCH_SIZE
                        batch size
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --bptt BPTT           sequence length
  --log-interval LOG_INTERVAL
                        report interval
  --seed SEED           random seed
```


