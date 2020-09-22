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

# Using pre-trained models in MXNet


In this tutorial we will see how to use multiple pre-trained models with Apache MXNet. First, let's download three image classification models from the Apache MXNet [Gluon model zoo](https://mxnet.apache.org/api/python/gluon/model_zoo.html).
* **DenseNet-121** ([research paper](https://arxiv.org/abs/1608.06993)), improved state of the art on [ImageNet dataset](http://image-net.org/challenges/LSVRC) in 2016.
* **MobileNet** ([research paper](https://arxiv.org/abs/1704.04861)), MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks, suitable for mobile applications.
* **ResNet-18** ([research paper](https://arxiv.org/abs/1512.03385v1)), the -152 version is the 2015 winner in multiple categories.

Why would you want to try multiple models? Why not just pick the one with the best accuracy? As we will see later in the tutorial, even though these models have been trained on the same dataset and optimized for maximum accuracy, they do behave slightly differently on specific images. In addition, prediction speed and memory footprints can vary, and that is an important factor for many applications. By trying a few pretrained models, you have an opportunity to find a model that can be a good fit for solving your business problem.


```python
import json

import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np
%matplotlib inline
```

## Loading the model

The [Gluon Model Zoo](https://mxnet.apache.org/api/python/gluon/model_zoo.html) provides a collection of off-the-shelf models. You can get the ImageNet pre-trained model by using `pretrained=True`.
If you want to train on your own classification problem from scratch, you can get an untrained network with a specific number of classes using the `classes` parameter: for example `net = vision.resnet18_v1(classes=10)`. However note that you cannot use the `pretrained` and `classes` parameter at the same time. If you want to use pre-trained weights as initialization of your network except for the last layer, have a look at the last section of this tutorial.

We can specify the *context* where we want to run the model: the default behavior is to use a CPU context. There are two reasons for this:
* First, this will allow you to test the notebook even if your machine is not equipped with a GPU :)
* Second, we're going to predict a single image and we don't have any specific performance requirements. For production applications where you'd want to predict large batches of images with the best possible throughput, a GPU could definitely be the way to go.
* If you want to use a GPU, make sure you have pip installed the right version of mxnet, or you will get an error when using the `mx.gpu()` context. Refer to the [install instructions](/get_started)


```python
# We set the context to CPU, you can switch to GPU if you have one and installed a compatible version of MXNet
ctx = mx.cpu()
```


```python
# We can load three the three models
densenet121 = vision.densenet121(pretrained=True, ctx=ctx)
mobileNet = vision.mobilenet0_5(pretrained=True, ctx=ctx)
resnet18 = vision.resnet18_v1(pretrained=True, ctx=ctx)
```

We can look at the description of the MobileNet network for example, which has a relatively simple yet deep architecture


```python
print(mobileNet)
```


Let's have a closer look at the first convolution layer:


```python
print(mobileNet.features[0].params)
```

The first layer applies **`16`** different convolutional masks, of size **`InputChannels x 3 x 3`**. For the first convolution, there are **`3`** input channels, the `R`, `G`, `B` channels of the input image. That gives us the weight matrix of shape **`16 x 3 x 3 x 3`**. There is no bias applied in this convolution.

Let's have a look at the output layer now:


```python
print(mobileNet.output)
```


Did you notice the shape of layer? The weight matrix is **1000 x 512**. This layer contains 1,000 neurons: each of them will store an activation representative of the probability of the image belonging to a specific category. Each neuron is also fully connected to all 512 neurons in the previous layer.

OK, enough exploring! Now let's use these models to classify our own images.

## Loading the data
All three models have been pre-trained on the ImageNet data set which includes over 1.2 million pictures of objects and animals sorted in 1,000 categories.
We get the imageNet list of labels. That way we have the mapping so when the model predicts for example category index `4`, we know it is predicting `hammerhead, hammerhead shark`


```python
mx.test_utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/image_net_labels.json')
categories = np.array(json.load(open('image_net_labels.json', 'r')))
print(categories[4])
```

Get a test image


```python
filename = mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/onnx/images/dog.jpg?raw=true', fname='dog.jpg')
```

If you want to use your own image for the test, copy the image to the same folder that contains the notebook and change the following line:


```python
filename = 'dog.jpg'
```

Load the image as a NDArray


```python
image = mx.image.imread(filename)
plt.imshow(image.asnumpy())
```

Neural network expects input in a specific format. Usually images comes in the `Width x Height x Channels` format. Where channels are the RGB channels.
This network accepts images in the `BatchSize x 3 x 224 x 224`. `224 x 224` is the image resolution, that's how the model was trained. `3` is the number of channels : Red, Green and Blue (in this order). In this case we use a `BatchSize` of `1` since we are predicting one image at a time.
Here are the transformation steps:
* Read the image: this will return a NDArray shaped as (image height, image width, 3), with the three channels in RGB order.
* Resize the shorter edge of the image 224.
* Crop, using a size of 224x224 from the center of the image.
* Shift the mean and standard deviation of our color channels to match the ones of the dataset the network has been trained on.
* Transpose the array from (Height, Width, 3) to (3, Height, Width).
* Add a fourth dimension, the batch dimension.



```python
def transform(image):
    resized = mx.image.resize_short(image, 224) #minimum 224x224 images
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped.astype(np.float32)/255,
                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
                                      std=mx.nd.array([0.229, 0.224, 0.225]))
    # the network expect batches of the form (N,3,224,224)
    transposed = normalized.transpose((2,0,1))  # Transposing from (224, 224, 3) to (3, 224, 224)
    batchified = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified
```

## Testing the different networks
We run the image through each pre-trained network. The models output a *NDArray* holding 1,000 activation values, which we convert to probabilities using the `softmax()` function, corresponding to the 1,000 categories it has been trained on. The output prediction NDArray has only one row since batch size is equal to 1


```python
predictions = resnet18(transform(image)).softmax()
print(predictions.shape)
```


We then take the top `k` predictions for our image, here the top `3`.


```python
top_pred = predictions.topk(k=3)[0].asnumpy()
```

And we print the categories predicted with their corresponding probabilities:


```python
for index in top_pred:
    probability = predictions[0][int(index)]
    category = categories[int(index)]
    print("{}: {:.2f}%".format(category, probability.asscalar()*100))
```


Let's turn this into a function. Our parameters are an image, a model, a list of categories and the number of top categories we'd like to print.


```python
def predict(model, image, categories, k):
    predictions = model(transform(image)).softmax()
    top_pred = predictions.topk(k=k)[0].asnumpy()
    for index in top_pred:
        probability = predictions[0][int(index)]
        category = categories[int(index)]
        print("{}: {:.2f}%".format(category, probability.asscalar()*100))
    print('')
```

### DenseNet121


```python
%%time
predict(densenet121, image, categories, 3)
```


### MobileNet


```python
%%time
predict(mobileNet, image, categories, 3)
```

### Resnet-18


```python
%%time
predict(resnet18, image, categories, 3)
```

As you can see, pre-trained networks produce slightly different predictions, and have different run-time. In this case, MobileNet is almost **5 times faster** than DenseNet!

## Fine-tuning pre-trained models

You can replace the output layer of your pre-trained model to fit the right number of classes for your own image classification task like this, for example for 10 classes:


```python
NUM_CLASSES = 10
with resnet18.name_scope():
    resnet18.output = gluon.nn.Dense(NUM_CLASSES)
```


```python
print(resnet18.output)
```


Now you can train your model on your new data using the pre-trained weights as initialization. This is called transfer learning and it has proved to be very useful especially in the cases where you only have access to a small dataset. Your network will have already learned how to perform general pattern detection and feature extraction on the larger dataset.
You can learn more about transfer learning and fine-tuning with MXNet in these tutorials:
- [Transferring knowledge through fine-tuning](http://gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html)
- [Fine Tuning an ONNX Model](/api/python/docs/tutorials/packages/onnx/fine_tuning_gluon.html)


That's it! Explore the model zoo, have fun with pre-trained models!

<!-- INSERT SOURCE DOWNLOAD BUTTONS -->
