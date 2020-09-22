---
layout: page_api
title: Model API *Deprecated*
permalink: /api/scala/docs/tutorials/model
is_tutorial: true
tag: scala
---
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

# MXNet Scala Model API

The model API provides a simplified way to train neural networks using common best practices.
It's a thin wrapper built on top of the [ndarray](ndarray) and [symbolic](symbol)
modules that make neural network training easy.

Topics:

* [Train the Model](#train-the-model)
* [Save the Model](#save-the-model)
* [Periodic Checkpoint](#periodic-checkpointing)
* [Multiple Devices](#use-multiple-devices)
* [Model API Reference]({{'/api/scala/docs/api/#org.apache.mxnet.Model'|relative_url}})

## Train the Model

To train a model, perform two steps: configure the model using the symbol parameter,
then call ```model.Feedforward.create``` to create the model.
The following example creates a two-layer neural network.

```scala
    // configure a two layer neuralnetwork
    val data = Symbol.Variable("data")
    val fc1 = Symbol.api.FullyConnected(data, num_hidden = 128, name = "fc1")
    val act1 = Symbol.api.Activation(Some(fc1), "relu", "relu1")
    val fc2 = Symbol.api.FullyConnected(Some(act1), num_hidden = 64, name = "fc2")
    val softmax = Symbol.api.SoftmaxOutput(Some(fc2), name = "sm")

    // Construct the FeedForward model and fit on the input training data
    val model = FeedForward.newBuilder(softmax)
      .setContext(Context.cpu())
      .setNumEpoch(num_epoch)
      .setOptimizer(new SGD(learningRate = 0.01f, momentum = 0.9f, wd = 0.0001f))
      .setTrainData(trainDataIter)
      .setEvalData(valDataIter)
      .build()
```
You can also use the `scikit-learn-style` construct and `fit` function to create a model.

```scala
    // create a model using sklearn-style two-step way
    val model = new FeedForward(softmax,
                                numEpoch = numEpochs,
                                argParams = argParams,
                                auxParams = auxParams,
                                beginEpoch = beginEpoch,
                                epochSize = epochSize)

  model.fit(trainData = train)
```
For more information, see [API Reference]({{'/api/scala/docs/api/#package'|relative_url}}).

## Save the Model

After the job is done, save your work.
We also provide `save` and `load` functions. You can use the `load` function to load a model checkpoint from a file.

```scala
    // checkpoint the model data into file,
    // save a model to modelPrefix-symbol.json and modelPrefix-0100.params
    val modelPrefix: String = "checkpt"
    val num_epoch = 100
    Model.saveCheckpoint(modelPrefix, epoch + 1, symbol, argParams, auxStates)

    // load model back
    val model_loaded = FeedForward.load(modelPrefix, num_epoch)
```
The advantage of these two `save` and `load` functions is that they are language agnostic.
You should be able to save and load directly into cloud storage, such as Amazon S3 and HDFS.

##  Periodic Checkpointing

We recommend checkpointing your model after each iteration.
To do this, use ```EpochEndCallback``` to add a ```Model.saveCheckpoint(<parameters>)``` checkpoint callback to the function after each iteration .

```scala
    // modelPrefix-symbol.json will be saved for symbol.
    // modelPrefix-epoch.params will be saved for parameters.
    // Checkpoint the model into file. Can specify parameters.
    // For more information, check API doc.
    val modelPrefix: String = "checkpt"
    val checkpoint: EpochEndCallback =
    if (modelPrefix == null) null
    else new EpochEndCallback {
      override def invoke(epoch: Int, symbol: Symbol,
                         argParams: Map[String, NDArray],
                         auxStates: Map[String, NDArray]): Unit = {
       Model.saveCheckpoint(modelPrefix, epoch + 1, symbol, argParams, auxParams)
            }
           }

    // Load model checkpoint from file. Returns symbol, argParams, auxParams.
    val (_, argParams, _) = Model.loadCheckpoint(modelPrefix, num_epoch)

```
You can load the model checkpoint later using ```Model.loadCheckpoint(modelPrefix, num_epoch)```.

## Use Multiple Devices

Set ```ctx``` to the list of devices that you want to train on. You can create a list of devices in any way you want.

```scala
    val devices = Array(Context.gpu(0), Context.gpu(1))

    val model = new FeedForward(ctx = devices,
             symbol = network,
             numEpoch = numEpochs,
             optimizer = optimizer,
             epochSize = epochSize,
             ...)
```
Training occurs in parallel on the GPUs that you specify.

## Next Steps
* See [Symbolic API](symbol) for operations on NDArrays that assemble neural networks from layers.
* See [IO Data Loading API](io) for parsing and loading data.
* See [NDArray API](ndarray) for vector/matrix/tensor operations.
* See [KVStore API](kvstore) for multi-GPU and multi-host distributed training.