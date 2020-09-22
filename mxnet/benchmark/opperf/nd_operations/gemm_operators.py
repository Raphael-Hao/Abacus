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
from benchmark.opperf.utils.benchmark_utils import run_performance_test
from benchmark.opperf.utils.common_utils import merge_map_list
from benchmark.opperf.rules.default_params import MX_OP_MODULE
"""Performance benchmark tests for MXNet NDArray GEMM Operators.

1. dot
2. batch_dot
3. khatri_rao

TODO
3. As part of default tests, following needs to be added:
    3.1 Sparse dot. (csr, default) -> row_sparse
    3.2 Sparse dot. (csr, row_sparse) -> default
    3.3 With Transpose of lhs
    3.4 With Transpose of rhs
4. 1D array: inner product of vectors
"""


def run_gemm_operators_benchmarks(ctx=mx.cpu(), dtype='float32', profiler='native', warmup=25, runs=100):
    """Runs benchmarks with the given context and precision (dtype)for all the GEMM
    operators (dot, batch_dot, khatri_rao) in MXNet.

    Parameters
    ----------
    ctx: mx.ctx
        Context to run benchmarks
    dtype: str, default 'float32'
        Precision to use for benchmarks
    profiler: str, default 'native'
        Type of Profiler to use (native/python)
    warmup: int, default 25
        Number of times to run for warmup
    runs: int, default 100
        Number of runs to capture benchmark results

    Returns
    -------
    Dictionary of results. Key -> Name of the operator, Value -> Benchmark results.

    """
    # Benchmark tests for dot operator
    dot_benchmark_res = run_performance_test(
        [getattr(MX_OP_MODULE, "dot")], run_backward=True,
        dtype=dtype, ctx=ctx,
        inputs=[{"lhs": (1024, 1024),
                 "rhs": (1024, 1024)},
                {"lhs": (1000, 10),
                 "rhs": (1000, 10),
                 "transpose_b": True},
                {"lhs": (1000, 1),
                 "rhs": (100, 1000),
                 "transpose_a": True,
                 "transpose_b": True}],
        warmup=warmup, runs=runs, profiler=profiler)
    # Benchmark tests for batch_dot operator
    batch_dot_benchmark_res = run_performance_test(
        [getattr(MX_OP_MODULE, "batch_dot")], run_backward=True,
        dtype=dtype, ctx=ctx,
        inputs=[{"lhs": (32, 1024, 1024),
                 "rhs": (32, 1024, 1024)},
                {"lhs": (32, 1000, 10),
                 "rhs": (32, 1000, 10),
                 "transpose_b": True},
                {"lhs": (32, 1000, 1),
                 "rhs": (32, 100, 1000),
                 "transpose_a": True,
                 "transpose_b": True}],
        warmup=warmup, runs=runs, profiler=profiler)
    # Operator khatri_rao is not yet implemented for GPU
    khatri_rao_benchmark_res = []
    if ctx != mx.gpu():
        # Benchmark tests for khatri_rao operator
        khatri_rao_benchmark_res = run_performance_test(
            [getattr(MX_OP_MODULE, "khatri_rao")], run_backward=False,
            dtype=dtype, ctx=ctx,
            inputs=[{"args": [(32, 32), (32, 32)]},
                    {"args": [(64, 64), (64, 64)]}],
            warmup=warmup, runs=runs, profiler=profiler)

    # Prepare combined results for GEMM operators
    mx_gemm_op_results = merge_map_list(dot_benchmark_res + batch_dot_benchmark_res + khatri_rao_benchmark_res)
    return mx_gemm_op_results
