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

FCN-xs EXAMPLE
--------------
This folder contains an example implementation for Fully Convolutional Networks (FCN) in MXNet.  
The example is based on the [FCN paper](https://arxiv.org/abs/1411.4038) by long et al. of UC Berkeley.

## Sample results
![fcn-xs pasval_voc result](https://github.com/dmlc/web-data/blob/master/mxnet/image/fcnxs-example-result.jpg)

We have trained a simple fcn-xs model, the hyper-parameters are below:

| model   | lr (fixed) | epoch |
| ------- | ---------: | ----: |
| fcn-32s |      1e-10 |    31 |
| fcn-16s |      1e-12 |    27 |
| fcn-8s  |      1e-14 |    19 |

(```when using the newest mxnet, you'd better using larger learning rate, such as 1e-4, 1e-5, 1e-6 instead, because the newest mxnet will do gradient normalization in SoftmaxOutput```)

The training dataset size is only 2027, and the validation dataset size is 462.  

## Training the model

### Step 1: setup pre-requisites

- Install python package `Pillow` (required by `image_segment.py`).
```shell
pip install --upgrade Pillow
```
- Setup your working directory. Assume your working directory is `~/train_fcn_xs`, and MXNet is built as `~/mxnet`. Copy example scripts into the working directory.
```shell
cp ~/mxnet/example/fcn-xs/* .
```
### Step 2: Download the vgg16fc model and training data
* vgg16fc model: you can download the ```VGG_FC_ILSVRC_16_layers-symbol.json``` and ```VGG_FC_ILSVRC_16_layers-0074.params``` from [baidu yun](http://pan.baidu.com/s/1bgz4PC), [dropbox](https://www.dropbox.com/sh/578n5cxej7ofd6m/AACuSeSYGcKQDi1GoB72R5lya?dl=0).  
this is the fully convolution style of the origin
[VGG_ILSVRC_16_layers.caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel), and the corresponding [VGG_ILSVRC_16_layers_deploy.prototxt](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-vgg_ilsvrc_16_layers_deploy-prototxt), the vgg16 model has [license](http://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
* Training data: download the ```VOC2012.rar```  [robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), and extract it into ```.\VOC2012```
* Mapping files: download ```train.lst```, ```val.lst``` from [baidu yun](http://pan.baidu.com/s/1bgz4PC) into the ```.\VOC2012``` directory

Once you completed all these steps, your working directory should contain a ```.\VOC2012``` directory, which contains the following: ```JPEGImages folder```, ```SegmentationClass folder```, ```train.lst```, ```val.lst```

#### Step 3: Train the fcn-xs model
* Based on your hardware, configure CPU or GPU for training by parameter ```--gpu```. It is recommended to use GPU due to the computational complexity and data load. 
View parameters we can use with the following command.
```shell
python fcn_xs.py -h


usage: fcn_xs.py [-h] [--model MODEL] [--prefix PREFIX] [--epoch EPOCH]
                 [--init-type INIT_TYPE] [--retrain] [--gpu GPU]

Convert vgg16 model to vgg16fc model.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.
  --prefix PREFIX       The prefix(include path) of vgg16 model with mxnet
                        format.
  --epoch EPOCH         The epoch number of vgg16 model.
  --init-type INIT_TYPE
                        the init type of fcn-xs model, e.g. vgg16, fcnxs
  --retrain             true means continue training.
  --gpu GPU             0 to use GPU, not set to use CPU
```

* It is recommended to train fcn-32s and fcn-16s before training the fcn-8s model

To train the fcn-32s model, run the following:

```shell
python -u fcn_xs.py --model=fcn32s --prefix=VGG_FC_ILSVRC_16_layers --epoch=74 --init-type=vgg16
```
* In the fcn_xs.py, you may need to change the directory ```root_dir```, ```flist_name```, ``fcnxs_model_prefix``` for your own data.
* When you train fcn-16s or fcn-8s model, you should change the code in ```run_fcnxs.sh``` corresponding, such as when train fcn-16s, comment out the fcn32s script, then it will like this:
```shell
 python -u fcn_xs.py --model=fcn16s --prefix=FCN32s_VGG16 --epoch=31 --init-type=fcnxs
```
* The output log may look like this(when training fcn-8s):
```c++
INFO:root:Start training with gpu(3)
INFO:root:Epoch[0] Batch [50]   Speed: 1.16 samples/sec Train-accuracy=0.894318
INFO:root:Epoch[0] Batch [100]  Speed: 1.11 samples/sec Train-accuracy=0.904681
INFO:root:Epoch[0] Batch [150]  Speed: 1.13 samples/sec Train-accuracy=0.908053
INFO:root:Epoch[0] Batch [200]  Speed: 1.12 samples/sec Train-accuracy=0.912219
INFO:root:Epoch[0] Batch [250]  Speed: 1.13 samples/sec Train-accuracy=0.914238
INFO:root:Epoch[0] Batch [300]  Speed: 1.13 samples/sec Train-accuracy=0.912170
INFO:root:Epoch[0] Batch [350]  Speed: 1.12 samples/sec Train-accuracy=0.912080
```

## Using the pre-trained model for image segmentation
To try out the pre-trained model, follow these steps:
* Download the pre-trained symbol and weights from [yun.baidu](http://pan.baidu.com/s/1bgz4PC). You should download these files: ```FCN8s_VGG16-symbol.json``` and ```FCN8s_VGG16-0019.params```
* Run the segmentation script, providing it your input image path: ```python image_segmentaion.py --input <your JPG image path>```
* The segmented output ```.png``` file will be generated in the working directory

## Tips
* This example runs full image size training, so there is no need to resize or crop input images to the same size. Accordingly, batch_size during training is set to 1.
* The fcn-xs model is based on vgg16 model, with some crop, deconv, element-sum layer added, so the model is quite big, moreover, the example is using whole image size training, if the input image is large(such as 700*500), then memory consumption may be high. Due to that, I suggest you use GPU with at least 12GB memory for training.
* If you don't have access to GPU with 12GB memory for training, I suggest you change the ```cut_off_size``` to a small value when constructing the FileIter, example below:  
```python
train_dataiter = FileIter(
      root_dir             = "./VOC2012",
      flist_name           = "train.lst",
      cut_off_size         = 400,
      rgb_mean             = (123.68, 116.779, 103.939),
      )
```

