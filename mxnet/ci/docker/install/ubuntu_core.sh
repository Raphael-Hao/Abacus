#!/usr/bin/env bash

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

# build and install are separated so changes to build don't invalidate
# the whole docker cache for the image

set -ex
# FIXME(larroy) enable in a different PR
#perl -pi -e 's/archive.ubuntu.com/us-west-2.ec2.archive.ubuntu.com/' /etc/apt/sources.list
apt-get update || true

# Avoid interactive package installers such as tzdata.
export DEBIAN_FRONTEND=noninteractive

apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    curl \
    git \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libjemalloc-dev \
    libhdf5-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    libturbojpeg \
    libzmq3-dev \
    libtinfo-dev \
    zlib1g-dev \
    libedit-dev \
    libxml2-dev \
    ninja-build \
    software-properties-common \
    sudo \
    unzip \
    vim-nox \
    wget

# Use libturbojpeg package as it is correctly compiled with -fPIC flag
# https://github.com/HaxeFoundation/hashlink/issues/147
ln -s /usr/lib/x86_64-linux-gnu/libturbojpeg.so.0.1.0 /usr/lib/x86_64-linux-gnu/libturbojpeg.so


# CMake 3.13.2+ is required
mkdir /opt/cmake && cd /opt/cmake
wget -nv https://cmake.org/files/v3.13/cmake-3.13.5-Linux-x86_64.sh
sh cmake-3.13.5-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
rm cmake-3.13.5-Linux-x86_64.sh
cmake --version
