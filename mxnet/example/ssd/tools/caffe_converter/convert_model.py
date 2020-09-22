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

from __future__ import print_function
import argparse
import sys
import caffe_parser
import mxnet as mx
import numpy as np
from convert_symbol import convert_symbol

def convert_model(prototxt_fname, caffemodel_fname, output_prefix=None):
    """Convert caffe model

    Parameters
    ----------

    prototxt_fname : str
         Filename of the prototxt model definition
    caffemodel_fname : str
         Filename of the binary caffe model
    output_prefix : str, optinoal
         If given, then save the converted MXNet into output_prefx+'.json' and
         output_prefx+'.params'

    Returns
    -------
    sym : Symbol
         Symbol convereted from prototxt
    arg_params : list of NDArray
         Argument parameters
    aux_params : list of NDArray
         Aux parameters
    input_dim : tuple
         Input dimension
    """
    sym, input_dim = convert_symbol(prototxt_fname)
    arg_shapes, _, aux_shapes = sym.infer_shape(data=tuple(input_dim))
    arg_names = sym.list_arguments()
    aux_names = sym.list_auxiliary_states()
    arg_shape_dic = dict(zip(arg_names, arg_shapes))
    aux_shape_dic = dict(zip(aux_names, aux_shapes))
    arg_params = {}
    aux_params = {}
    first_conv = True

    layers, names = caffe_parser.read_caffemodel(prototxt_fname, caffemodel_fname)
    layer_iter = caffe_parser.layer_iter(layers, names)
    layers_proto = caffe_parser.get_layers(caffe_parser.read_prototxt(prototxt_fname))

    for layer_name, layer_type, layer_blobs in layer_iter:
        if layer_type == 'Convolution' or layer_type == 'InnerProduct'  \
           or layer_type == 4 or layer_type == 14 or layer_type == 'PReLU' \
           or layer_type == 'Deconvolution' or layer_type == 39  or layer_type == 'Normalize':
            if layer_type == 'PReLU':
                assert (len(layer_blobs) == 1)
                wmat = layer_blobs[0].data
                weight_name = layer_name + '_gamma'
                arg_params[weight_name] = mx.nd.zeros(wmat.shape)
                arg_params[weight_name][:] = wmat
                continue
            if layer_type == 'Normalize':
                assert (len(layer_blobs) == 1)
                weight_name = layer_name + '_scale'
                wmat = layer_blobs[0].data
                arg_params[weight_name] = mx.nd.zeros((1, len(wmat), 1, 1))
                arg_params[weight_name][:] = np.array(list(wmat)).reshape((1, len(wmat), 1, 1))
                continue
            wmat_dim = []
            if getattr(layer_blobs[0].shape, 'dim', None) is not None:
                if len(layer_blobs[0].shape.dim) > 0:
                    wmat_dim = layer_blobs[0].shape.dim
                else:
                    wmat_dim = [layer_blobs[0].num, layer_blobs[0].channels,
                                layer_blobs[0].height, layer_blobs[0].width]
            else:
                wmat_dim = list(layer_blobs[0].shape)
            wmat = np.array(layer_blobs[0].data).reshape(wmat_dim)

            channels = wmat_dim[1]
            if channels == 3 or channels == 4:  # RGB or RGBA
                if first_conv:
                    # Swapping BGR of caffe into RGB in mxnet
                    wmat[:, [0, 2], :, :] = wmat[:, [2, 0], :, :]

            assert(wmat.flags['C_CONTIGUOUS'] is True)
            sys.stdout.write('converting layer {0}, wmat shape = {1}'.format(
                layer_name, wmat.shape))
            if len(layer_blobs) == 2:
                bias = np.array(layer_blobs[1].data)
                bias = bias.reshape((bias.shape[0], 1))
                assert(bias.flags['C_CONTIGUOUS'] is True)
                bias_name = layer_name + "_bias"

                if bias_name not in arg_shape_dic:
                    print(bias_name + ' not found in arg_shape_dic.')
                    continue
                bias = bias.reshape(arg_shape_dic[bias_name])
                arg_params[bias_name] = mx.nd.zeros(bias.shape)
                arg_params[bias_name][:] = bias
                sys.stdout.write(', bias shape = {}'.format(bias.shape))

            sys.stdout.write('\n')
            sys.stdout.flush()
            wmat = wmat.reshape((wmat.shape[0], -1))
            weight_name = layer_name + "_weight"

            if weight_name not in arg_shape_dic:
                print(weight_name + ' not found in arg_shape_dic.')
                continue
            wmat = wmat.reshape(arg_shape_dic[weight_name])
            arg_params[weight_name] = mx.nd.zeros(wmat.shape)
            arg_params[weight_name][:] = wmat


            if first_conv and (layer_type == 'Convolution' or layer_type == 4):
                first_conv = False

        elif layer_type == 'Scale':
            if 'scale' in layer_name:
                bn_name = layer_name.replace('scale', 'bn')
            elif 'sc' in layer_name:
                bn_name = layer_name.replace('sc', 'bn')
            else:
                assert False, 'Unknown name convention for bn/scale'

            gamma = np.array(layer_blobs[0].data)
            beta = np.array(layer_blobs[1].data)
            # beta = np.expand_dims(beta, 1)
            beta_name = '{}_beta'.format(bn_name)
            gamma_name = '{}_gamma'.format(bn_name)

            beta = beta.reshape(arg_shape_dic[beta_name])
            gamma = gamma.reshape(arg_shape_dic[gamma_name])
            arg_params[beta_name] = mx.nd.zeros(beta.shape)
            arg_params[gamma_name] = mx.nd.zeros(gamma.shape)
            arg_params[beta_name][:] = beta
            arg_params[gamma_name][:] = gamma

            assert gamma.flags['C_CONTIGUOUS'] is True
            assert beta.flags['C_CONTIGUOUS'] is True
            print('converting scale layer, beta shape = {}, gamma shape = {}'.format(
                beta.shape, gamma.shape))
        elif layer_type == 'BatchNorm':
            bn_name = layer_name
            mean = np.array(layer_blobs[0].data)
            var = np.array(layer_blobs[1].data)
            rescale_factor = layer_blobs[2].data[0]
            if rescale_factor != 0:
                rescale_factor = 1 / rescale_factor
            mean_name = '{}_moving_mean'.format(bn_name)
            var_name = '{}_moving_var'.format(bn_name)
            mean = mean.reshape(aux_shape_dic[mean_name])
            var = var.reshape(aux_shape_dic[var_name])
            aux_params[mean_name] = mx.nd.zeros(mean.shape)
            aux_params[var_name] = mx.nd.zeros(var.shape)
            # Get the original epsilon
            for idx, layer in enumerate(layers_proto):
                if layer.name == bn_name:
                    bn_index = idx
            eps_caffe = layers_proto[bn_index].batch_norm_param.eps
            # Compensate for the epsilon shift performed in convert_symbol
            eps_symbol = float(sym.attr_dict()[bn_name + '_moving_mean']['eps'])
            eps_correction = eps_caffe - eps_symbol
            # Fill parameters
            aux_params[mean_name][:] = mean * rescale_factor
            aux_params[var_name][:] = var * rescale_factor + eps_correction
            assert var.flags['C_CONTIGUOUS'] is True
            assert mean.flags['C_CONTIGUOUS'] is True
            print('converting batchnorm layer, mean shape = {}, var shape = {}'.format(
                mean.shape, var.shape))

            fix_gamma = layers_proto[bn_index+1].type != 'Scale'
            if fix_gamma:
                gamma_name = '{}_gamma'.format(bn_name)
                gamma = np.array(np.ones(arg_shape_dic[gamma_name]))
                beta_name = '{}_beta'.format(bn_name)
                beta = np.array(np.zeros(arg_shape_dic[beta_name]))
                arg_params[beta_name] = mx.nd.zeros(beta.shape)
                arg_params[gamma_name] = mx.nd.zeros(gamma.shape)
                arg_params[beta_name][:] = beta
                arg_params[gamma_name][:] = gamma
                assert gamma.flags['C_CONTIGUOUS'] is True
                assert beta.flags['C_CONTIGUOUS'] is True

        else:
            print('\tskipping layer {} of type {}'.format(layer_name, layer_type))
            assert len(layer_blobs) == 0

    if output_prefix is not None:
        model = mx.mod.Module(symbol=sym, label_names=None)
        model.bind(data_shapes=[('data', tuple(input_dim))])
        model.init_params(arg_params=arg_params, aux_params=aux_params)
        model.save_checkpoint(output_prefix, 0)

    return sym, arg_params, aux_params, input_dim

def main():
    parser = argparse.ArgumentParser(
        description='Caffe prototxt to mxnet model parameter converter.')
    parser.add_argument('prototxt', help='The prototxt filename')
    parser.add_argument('caffemodel', help='The binary caffemodel filename')
    parser.add_argument('save_model_name', help='The name of the output model prefix')
    args = parser.parse_args()

    convert_model(args.prototxt, args.caffemodel, args.save_model_name)
    print ('Saved model successfully to {}'.format(args.save_model_name))

if __name__ == '__main__':
    main()
