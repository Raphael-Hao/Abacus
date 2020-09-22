# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#-------------------------------------------------------------------------------
#  Template configuration for compiling mxnet
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of mxnet. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#-------------------------------------------------------------------------------

#---------------------
# For cross compilation we only explictily set a compiler when one is not already present.
#--------------------

ifndef CC
export CC = gcc
endif
ifndef CXX
export CXX = g++
endif
ifndef NVCC
export NVCC = nvcc
endif

# whether compile with options for MXNet developer
DEV = 0

# whether compile with debug
DEBUG = 0

# whether to turn on segfault signal handler to log the stack trace
USE_SIGNAL_HANDLER = 1

# the additional link flags you want to add
ADD_LDFLAGS = -L${CROSS_ROOT}/lib -L/usr/lib/aarch64-linux-gnu/

# the additional compile flags you want to add
ADD_CFLAGS = -I${CROSS_ROOT}/include -I/usr/include/aarch64-linux-gnu/

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------

# whether use CUDA during compile
USE_CUDA = 1

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable, leave it as NONE
# USE_CUDA_PATH = /usr/local/cuda
USE_CUDA_PATH = /usr/local/cuda-9.0/targets/aarch64-linux

# whether to enable CUDA runtime compilation
ENABLE_CUDA_RTC = 0

# whether use CuDNN R3 library
USE_CUDNN = 1

#whether to use NCCL library
USE_NCCL = 0
#add the path to NCCL library
USE_NCCL_PATH = NONE

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 0
# Add OpenCV include path, in which the directory `opencv2` exists
USE_OPENCV_INC_PATH = NONE
# Add OpenCV shared library path, in which the shared library exists
USE_OPENCV_LIB_PATH = NONE

#whether use libjpeg-turbo for image decode without OpenCV wrapper
USE_LIBJPEG_TURBO = 0
#add the path to libjpeg-turbo library
USE_LIBJPEG_TURBO_PATH = NONE

# use openmp for parallelization
USE_OPENMP = 1

# whether use MKL-DNN library
USE_MKLDNN = 0

# whether use NNPACK library
USE_NNPACK = 0

# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
# in default use atlas for linux while apple for osx
UNAME_S := $(shell uname -s)
USE_BLAS = openblas

# whether use lapack during compilation
# only effective when compiled with blas versions openblas/apple/atlas/mkl
USE_LAPACK = 1

# path to lapack library in case of a non-standard installation
USE_LAPACK_PATH =

# add path to intel library, you may need it for MKL, if you did not add the path
# to environment variable
USE_INTEL_PATH = NONE

# If use MKL only for BLAS, choose static link automatically to allow python wrapper
ifeq ($(USE_BLAS), mkl)
USE_STATIC_MKL = 1
else
USE_STATIC_MKL = NONE
endif

#----------------------------
# Settings for power and arm arch
#----------------------------
USE_SSE=0

# Turn off F16C instruction set support
USE_F16C=0

#----------------------------
# distributed computing
#----------------------------

# whether or not to enable multi-machine supporting
USE_DIST_KVSTORE = 0

# whether or not allow to read and write HDFS directly. If yes, then hadoop is
# required
USE_HDFS = 0

# path to libjvm.so. required if USE_HDFS=1
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server

# whether or not allow to read and write AWS S3 directly. If yes, then
# libcurl4-openssl-dev is required, it can be installed on Ubuntu by
# sudo apt-get install -y libcurl4-openssl-dev
USE_S3 = 0

#----------------------------
# performance settings
#----------------------------
# Use operator tuning
USE_OPERATOR_TUNING = 1

# Use gperftools if found
# Disable because of #8968
USE_GPERFTOOLS = 0

# path to gperftools (tcmalloc) library in case of a non-standard installation
USE_GPERFTOOLS_PATH =

# Use JEMalloc if found, and not using gperftools
USE_JEMALLOC = 1

# path to jemalloc library in case of a non-standard installation
USE_JEMALLOC_PATH =

#----------------------------
# additional operators
#----------------------------

# path to folders containing projects specific operators that you don't want to put in src/operators
EXTRA_OPERATORS =

#----------------------------
# other features
#----------------------------

# Create C++ interface package
USE_CPP_PACKAGE = 0

# Use int64_t type to represent the total number of elements in the tensor
# This will cause performance degradation reported in issue #14496
# Set to 1 for large tensor with tensor size greater than INT32_MAX i.e. 2147483647
# Note: the size of each dimension is still bounded by INT32_MAX
USE_INT64_TENSOR_SIZE = 0

#----------------------------
# plugins
#----------------------------

# whether to use caffe integration. This requires installing caffe.
# You also need to add CAFFE_PATH/build/lib to your LD_LIBRARY_PATH
# CAFFE_PATH = $(HOME)/caffe
# MXNET_PLUGINS += plugin/caffe/caffe.mk

# WARPCTC_PATH = $(HOME)/warp-ctc
# MXNET_PLUGINS += plugin/warpctc/warpctc.mk

# whether to use sframe integration. This requires build sframe
# git@github.com:dato-code/SFrame.git
# SFRAME_PATH = $(HOME)/SFrame
# MXNET_PLUGINS += plugin/sframe/plugin.mk
