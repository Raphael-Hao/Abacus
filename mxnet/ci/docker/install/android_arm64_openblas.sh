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
pushd .
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make -j$(nproc) TARGET=ARMV8 ARM_SOFTFP_ABI=1 HOSTCC=gcc NOFORTRAN=1 libs
# Can't be run (utility not compiled for the target platform)
#make install
cp *.h /usr/include
cp libopenblas.a /usr/local/lib
popd
