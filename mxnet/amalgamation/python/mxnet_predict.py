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

# coding: utf-8
# pylint: disable=invalid-name, too-many-arguments
"""Lightweight API for mxnet prediction.

This is for prediction only, use mxnet python package instead for most tasks.
"""
from __future__ import absolute_import

import os
import sys
from array import array
import ctypes
import logging
import numpy as np

# pylint: disable= no-member
_DTYPE_NP_TO_MX = {
    None: -1,
    np.float32: 0,
    np.float64: 1,
    np.float16: 2,
    np.uint8: 3,
    np.int32: 4,
    np.int8: 5,
    np.int64: 6,
}

_DTYPE_MX_TO_NP = {
    -1: None,
    0: np.float32,
    1: np.float64,
    2: np.float16,
    3: np.uint8,
    4: np.int32,
    5: np.int8,
    6: np.int64,
}

__all__ = ["Predictor", "load_ndarray_file"]


py_str = lambda x: x.decode('utf-8')


def c_str_array(strings):
    """Create ctypes const char ** from a list of Python strings.

    Parameters
    ----------
    strings : list of string
        Python strings.

    Returns
    -------
    (ctypes.c_char_p * len(strings))
        A const char ** pointer that can be passed to C API.
    """
    arr = (ctypes.c_char_p * len(strings))()
    arr[:] = [s.encode('utf-8') for s in strings]
    return arr


def c_str(string):
    """"Convert a python string to C string."""
    if not isinstance(string, str):
        string = string.decode('ascii')
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Create ctypes array from a python array."""
    return (ctype * len(values))(*values)

def c_array_buf(ctype, buf):
    """Create ctypes array from a Python buffer."""
    return (ctype * len(buf)).from_buffer(buf)



def _find_lib_path():
    """Find mxnet library."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    amalgamation_lib_path = os.path.join(curr_path, '../../lib/libmxnet_predict.so')
    if os.path.exists(amalgamation_lib_path) and os.path.isfile(amalgamation_lib_path):
        lib_path = [amalgamation_lib_path]
        return lib_path
    else:
        logging.info('Cannot find libmxnet_predict.so. Will search for MXNet library using libinfo.py then.')
        try:
            from mxnet.libinfo import find_lib_path
            lib_path = find_lib_path()
            return lib_path
        except ImportError:
            libinfo_path = os.path.join(curr_path, '../../python/mxnet/libinfo.py')
            if os.path.exists(libinfo_path) and os.path.isfile(libinfo_path):
                libinfo = {'__file__': libinfo_path}
                exec(compile(open(libinfo_path, "rb").read(), libinfo_path, 'exec'), libinfo, libinfo)
                lib_path = libinfo['find_lib_path']()
                return lib_path
            else:
                raise RuntimeError('Cannot find libinfo.py at %s.' % libinfo_path)


def _load_lib():
    """Load libary by searching possible path."""
    lib_path = _find_lib_path()
    lib = ctypes.cdll.LoadLibrary(lib_path[0])
    # DMatrix functions
    lib.MXGetLastError.restype = ctypes.c_char_p
    return lib


def _check_call(ret):
    """Check the return value of API."""
    if ret != 0:
        raise RuntimeError(py_str(_LIB.MXGetLastError()))


def _monitor_callback_wrapper(callback):
    """A wrapper for the user-defined handle."""
    def callback_handle(name, array, _):
        """ ctypes function """
        callback(name, array)
    return callback_handle

_LIB = _load_lib()
# type definitions
mx_uint = ctypes.c_uint
mx_int = ctypes.c_int
mx_float = ctypes.c_float
mx_float_p = ctypes.POINTER(mx_float)
PredictorHandle = ctypes.c_void_p
NDListHandle = ctypes.c_void_p

devstr2type = {'cpu': 1, 'gpu': 2, 'cpu_pinned': 3}

class Predictor(object):
    """A predictor class that runs prediction.

    Parameters
    ----------
    symbol_json_str : str
        Path to the symbol file.

    param_raw_bytes : str, bytes
        The raw parameter bytes.

    input_shapes : dict of str to tuple
        The shape of input data

    dev_type : str, optional
        The device type of the predictor.

    dev_id : int, optional
        The device id of the predictor.

    type_dict : Dict of str->numpy.dtype
        Input type dictionary, name->dtype
    """
    def __init__(self, symbol_file,
                 param_raw_bytes, input_shapes,
                 dev_type="cpu", dev_id=0, type_dict=None):
        dev_type = devstr2type[dev_type]
        indptr = [0]
        sdata = []
        keys = []
        for k, v  in input_shapes.items():
            if not isinstance(v, tuple):
                raise ValueError("Expect input_shapes to be dict str->tuple")
            keys.append(c_str(k))
            sdata.extend(v)
            indptr.append(len(sdata))
        handle = PredictorHandle()
        param_raw_bytes = bytearray(param_raw_bytes)
        ptr = (ctypes.c_char * len(param_raw_bytes)).from_buffer(param_raw_bytes)

        # data types
        num_provided_arg_types = 0
        # provided type argument names
        provided_arg_type_names = ctypes.POINTER(ctypes.c_char_p)()
        # provided types
        provided_arg_type_data = ctypes.POINTER(mx_uint)()
        if type_dict is not None:
            provided_arg_type_names = []
            provided_arg_type_data = []
            for k, v in type_dict.items():
                v = np.dtype(v).type
                if v in _DTYPE_NP_TO_MX:
                    provided_arg_type_names.append(k)
                    provided_arg_type_data.append(_DTYPE_NP_TO_MX[v])
            num_provided_arg_types = mx_uint(len(provided_arg_type_names))
            provided_arg_type_names = c_str_array(provided_arg_type_names)
            provided_arg_type_data = c_array_buf(ctypes.c_int, array('i', provided_arg_type_data))

        _check_call(_LIB.MXPredCreateEx(
            c_str(symbol_file),
            ptr, len(param_raw_bytes),
            ctypes.c_int(dev_type), ctypes.c_int(dev_id),
            mx_uint(len(indptr) - 1),
            c_array(ctypes.c_char_p, keys),
            c_array(mx_uint, indptr),
            c_array(mx_uint, sdata),
            num_provided_arg_types,
            provided_arg_type_names,
            provided_arg_type_data,
            ctypes.byref(handle)))
        self.type_dict = type_dict
        self.handle = handle

    def __del__(self):
        _check_call(_LIB.MXPredFree(self.handle))

    def forward(self, **kwargs):
        """Perform forward to get the output.

        Parameters
        ----------
        **kwargs
            Keyword arguments of input variable name to data.

        Examples
        --------
        >>> predictor.forward(data=mydata)
        >>> out = predictor.get_output(0)
        """
        if self.type_dict and len(self.type_dict) != len(kwargs.items()):
            raise ValueError("number of kwargs should be same as len of type_dict" \
                             "Please check your forward pass inputs" \
                             "or type_dict passed to Predictor instantiation")

        for k, v in kwargs.items():
            if not isinstance(v, np.ndarray):
                raise ValueError("Expect numpy ndarray as input")
            if self.type_dict and k in self.type_dict:
                v = np.asarray(v, dtype=self.type_dict[k], order='C')
            else:
                v = np.asarray(v, dtype=np.float32, order='C')
            _check_call(_LIB.MXPredSetInput(
                self.handle, c_str(k),
                v.ctypes.data_as(mx_float_p),
                mx_uint(v.size)))
        _check_call(_LIB.MXPredForward(self.handle))

    def reshape(self, input_shapes):
        """Change the input shape of the predictor.

        Parameters
        ----------
        input_shapes : dict of str to tuple
            The new shape of input data.

        Examples
        --------
        >>> predictor.reshape({'data':data_shape_tuple})
        """
        indptr = [0]
        sdata = []
        keys = []
        for k, v  in input_shapes.items():
            if not isinstance(v, tuple):
                raise ValueError("Expect input_shapes to be dict str->tuple")
            keys.append(c_str(k))
            sdata.extend(v)
            indptr.append(len(sdata))

        new_handle = PredictorHandle()
        _check_call(_LIB.MXPredReshape(
            mx_uint(len(indptr) - 1),
            c_array(ctypes.c_char_p, keys),
            c_array(mx_uint, indptr),
            c_array(mx_uint, sdata),
            self.handle,
            ctypes.byref(new_handle)))
        _check_call(_LIB.MXPredFree(self.handle))
        self.handle = new_handle

    def get_output(self, index):
        """Get the index-th output.

        Parameters
        ----------
        index : int
            The index of output.

        Returns
        -------
        out : numpy array.
            The output array.
        """
        pdata = ctypes.POINTER(mx_uint)()
        ndim = mx_uint()
        out_type = mx_int()
        _check_call(_LIB.MXPredGetOutputShape(
            self.handle, index,
            ctypes.byref(pdata),
            ctypes.byref(ndim)))
        _check_call(_LIB.MXPredGetOutputType(
            self.handle, index,
            ctypes.byref(out_type)))
        shape = tuple(pdata[:ndim.value])
        data = np.empty(shape, dtype=_DTYPE_MX_TO_NP[out_type.value])
        _check_call(_LIB.MXPredGetOutput(
            self.handle, mx_uint(index),
            data.ctypes.data_as(mx_float_p),
            mx_uint(data.size)))
        return data

    def set_monitor_callback(self, callback, monitor_all=False):
        cb_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)
        self._monitor_callback = cb_type(_monitor_callback_wrapper(callback))
        _check_call(_LIB.MXPredSetMonitorCallback(self.handle,
                                                  self._monitor_callback,
                                                  None,
                                                  ctypes.c_int(monitor_all)))


def load_ndarray_file(nd_bytes):
    """Load ndarray file and return as list of numpy array.

    Parameters
    ----------
    nd_bytes : str or bytes
        The internal ndarray bytes

    Returns
    -------
    out : dict of str to numpy array or list of numpy array
        The output list or dict, depending on whether the saved type is list or dict.
    """
    handle = NDListHandle()
    olen = mx_uint()
    nd_bytes = bytearray(nd_bytes)
    ptr = (ctypes.c_char * len(nd_bytes)).from_buffer(nd_bytes)
    _check_call(_LIB.MXNDListCreate(
        ptr, len(nd_bytes),
        ctypes.byref(handle), ctypes.byref(olen)))
    keys = []
    arrs = []

    for i in range(olen.value):
        key = ctypes.c_char_p()
        cptr = mx_float_p()
        pdata = ctypes.POINTER(mx_uint)()
        ndim = mx_uint()
        _check_call(_LIB.MXNDListGet(
            handle, mx_uint(i), ctypes.byref(key),
            ctypes.byref(cptr), ctypes.byref(pdata), ctypes.byref(ndim)))
        shape = tuple(pdata[:ndim.value])
        dbuffer = (mx_float * np.prod(shape)).from_address(ctypes.addressof(cptr.contents))
        ret = np.frombuffer(dbuffer, dtype=np.float32).reshape(shape)
        ret = np.array(ret, dtype=np.float32)
        keys.append(py_str(key.value))
        arrs.append(ret)
    _check_call(_LIB.MXNDListFree(handle))

    if len(keys) == 0 or len(keys[0]) == 0:
        return arrs
    else:
        return {keys[i] : arrs[i] for i in range(len(keys))
  }
