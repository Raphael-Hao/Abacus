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
"""Create helper functions to load mnist dataset and toy dataset"""
from __future__ import print_function
import os
import ssl
import numpy


def load_mnist(training_num=50000):
    """Load mnist dataset"""
    data_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'mnist.npz')
    if not os.path.isfile(data_path):
        from six.moves import urllib
        origin = (
            'https://github.com/sxjscience/mxnet/raw/master/example/bayesian-methods/mnist.npz'
        )
        print('Downloading data from %s to %s' % (origin, data_path))
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(origin, context=ctx) as u, open(data_path, 'wb') as f:
            f.write(u.read())
        print('Done!')
    dat = numpy.load(data_path)
    X = (dat['X'][:training_num] / 126.0).astype('float32')
    Y = dat['Y'][:training_num]
    X_test = (dat['X_test'] / 126.0).astype('float32')
    Y_test = dat['Y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test


def load_toy():
    training_data = numpy.loadtxt('toy_data_train.txt')
    testing_data = numpy.loadtxt('toy_data_test_whole.txt')
    X = training_data[:, 0].reshape((training_data.shape[0], 1))
    Y = training_data[:, 1].reshape((training_data.shape[0], 1))
    X_test = testing_data[:, 0].reshape((testing_data.shape[0], 1))
    Y_test = testing_data[:, 1].reshape((testing_data.shape[0], 1))
    return X, Y, X_test, Y_test


def load_synthetic(theta1, theta2, sigmax, num=20):
    flag = numpy.random.randint(0, 2, (num,))
    X = flag * numpy.random.normal(theta1, sigmax, (num,)) \
        + (1.0 - flag) * numpy.random.normal(theta1 + theta2, sigmax, (num,))
    return X
