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

import mxnet as mx
import numpy as np
import json

def test_default_init():
    data = mx.sym.Variable('data')
    sym = mx.sym.LeakyReLU(data=data, act_type='prelu')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10,10))])
    mod.init_params()
    assert (list(mod.get_params()[0].values())[0].asnumpy() == 0.25).all()

def test_variable_init():
    data = mx.sym.Variable('data')
    gamma = mx.sym.Variable('gamma', init=mx.init.One())
    sym = mx.sym.LeakyReLU(data=data, gamma=gamma, act_type='prelu')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10,10))])
    mod.init_params()
    assert (list(mod.get_params()[0].values())[0].asnumpy() == 1).all()

def test_aux_init():
    data = mx.sym.Variable('data')
    sym = mx.sym.BatchNorm(data=data, name='bn')
    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=[('data', (10, 10, 3, 3))])
    mod.init_params()
    assert (mod.get_params()[1]['bn_moving_var'].asnumpy() == 1).all()
    assert (mod.get_params()[1]['bn_moving_mean'].asnumpy() == 0).all()

def test_rsp_const_init():
    def check_rsp_const_init(init, val):
        shape = (10, 10)
        x = mx.symbol.Variable("data", stype='csr')
        weight = mx.symbol.Variable("weight", shape=(shape[1], 2),
                                    init=init, stype='row_sparse')
        dot = mx.symbol.sparse.dot(x, weight)
        mod = mx.mod.Module(dot, label_names=None)
        mod.bind(data_shapes=[('data', shape)])
        mod.init_params()
        assert (list(mod.get_params()[0].values())[0].asnumpy() == val).all()

    check_rsp_const_init(mx.initializer.Constant(value=2.), 2.)
    check_rsp_const_init(mx.initializer.Zero(), 0.)
    check_rsp_const_init(mx.initializer.One(), 1.)

def test_bilinear_init():
    bili = mx.init.Bilinear()
    bili_weight = mx.ndarray.empty((1,1,4,4))
    bili._init_weight(None, bili_weight)
    bili_1d = np.array([[1/float(4), 3/float(4), 3/float(4), 1/float(4)]])
    bili_2d = bili_1d * np.transpose(bili_1d)
    assert (bili_2d == bili_weight.asnumpy()).all()

def test_const_init_dumps():
    shape = tuple(np.random.randint(1, 10, size=np.random.randint(1, 5)))
    # test NDArray input
    init = mx.init.Constant(mx.nd.ones(shape))
    val = init.dumps()
    assert val == json.dumps([init.__class__.__name__.lower(), init._kwargs])
    # test scalar input
    init = mx.init.Constant(1)
    assert init.dumps() == '["constant", {"value": 1}]'
    # test numpy input
    init = mx.init.Constant(np.ones(shape))
    val = init.dumps()
    assert val == json.dumps([init.__class__.__name__.lower(), init._kwargs])


if __name__ == '__main__':
    test_variable_init()
    test_default_init()
    test_aux_init()
    test_rsp_const_init()
    test_bilinear_init()
    test_const_init_dumps()
