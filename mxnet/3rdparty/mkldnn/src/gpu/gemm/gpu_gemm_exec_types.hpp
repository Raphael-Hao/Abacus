/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_GEMM_GPU_GEMM_EXEC_TYPES_HPP
#define GPU_GEMM_GPU_GEMM_EXEC_TYPES_HPP

#include "common/memory_storage.hpp"
#include "common/stream.hpp"

#define DNNL_ARG_A DNNL_ARG_SRC
#define DNNL_ARG_B DNNL_ARG_WEIGHTS
#define DNNL_ARG_C DNNL_ARG_DST

namespace dnnl {
namespace impl {
namespace gpu {

#define GEMM_CTX_ARG_STORAGE(argument) \
    (ctx.args().argument ? *(ctx.args().argument) \
                         : dnnl::impl::memory_storage_t::empty_storage())

struct gemm_exec_args_t {
    memory_storage_t *a = nullptr;
    memory_storage_t *b = nullptr;
    memory_storage_t *c = nullptr;
    memory_storage_t *a_zero_point = nullptr;
    memory_storage_t *b_zero_point = nullptr;
    memory_storage_t *c_zero_point = nullptr;
    memory_storage_t *bias = nullptr;
    memory_storage_t *output_scales = nullptr;
};

struct gemm_exec_ctx_t {
    gemm_exec_ctx_t(stream_t *stream, const gemm_exec_args_t &args,
            gemm_desc_t *gemm_desc = nullptr)
        : stream_(stream), args_(args), gemm_desc_(gemm_desc) {}

    stream_t *stream() const { return stream_; }
    const gemm_exec_args_t &args() const { return args_; }
    const gemm_desc_t *desc() const { return gemm_desc_; }

private:
    stream_t *stream_;
    gemm_exec_args_t args_;
    gemm_desc_t *gemm_desc_ = nullptr;
};

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
