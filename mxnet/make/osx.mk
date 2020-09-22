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
# choice of compiler
#--------------------

export CC = gcc
export CXX = g++
export NVCC = nvcc

# whether compile with options for MXNet developer
DEV = 0

# whether compile with debug
DEBUG = 0

# the additional link flags you want to add
ADD_LDFLAGS =

# the additional compile flags you want to add
ADD_CFLAGS =

# whether to build operators written in TVM
USE_TVM_OP = 0

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------

# whether use CUDA during compile
USE_CUDA = 0

# add the path to CUDA library to link and compile flag
# if you have already add them to environment variable, leave it as NONE
# USE_CUDA_PATH = /usr/local/cuda
USE_CUDA_PATH = NONE

# whether to enable CUDA runtime compilation
ENABLE_CUDA_RTC = 1

# whether use CUDNN R3 library
USE_CUDNN = 0

# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 1
# Add OpenCV include path, in which the directory `opencv2` exists
USE_OPENCV_INC_PATH = NONE
# Add OpenCV shared library path, in which the shared library exists
USE_OPENCV_LIB_PATH = NONE

# use openmp for parallelization
# apple-clang by default does not have openmp built-in
USE_OPENMP = 0

# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas, apple
USE_BLAS = apple

# whether use lapack during compilation
# only effective when compiled with blas versions openblas/apple/atlas/mkl
USE_LAPACK = 1

# by default, disable lapack when using MKL
# switch on when there is a full installation of MKL available (not just MKL_ML)
ifeq ($(USE_BLAS), mkl)
USE_LAPACK = 0
endif

# add path to intel library, you may need it for MKL, if you did not add the path
# to environment variable
USE_INTEL_PATH = NONE

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
# additional operators
#----------------------------

# path to folders containing projects specific operators that you don't want to put in src/operators
EXTRA_OPERATORS =

#----------------------------
# other features
#----------------------------

# Create C++ interface package
USE_CPP_PACKAGE = 0

# Use int64_t type to represent the total number of elements in a tensor
# This will cause performance degradation reported in issue #14496
# Set to 1 for large tensor with tensor size greater than INT32_MAX i.e. 2147483647
# Note: the size of each dimension is still bounded by INT32_MAX
USE_INT64_TENSOR_SIZE = 0

#----------------------------
# plugins
#----------------------------

# whether to use torch integration. This requires installing torch.
# TORCH_PATH = $(HOME)/torch
# MXNET_PLUGINS += plugin/torch/torch.mk
