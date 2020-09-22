/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GEMM_BASED_COMMON_HPP
#define GEMM_BASED_COMMON_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"

#include "cpu/cpu_isa_traits.hpp"

#include "cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {
namespace gemm_based {

struct params_t {
    // indicates if an auxiliary array for intermediate computations is not
    // required
    bool dst_is_acc_;

    // indicates if output scales from attributes are applied
    // by gemm (alpha parameter) or post-op kernel (pp_kernel_)
    bool gemm_applies_output_scales_ = false;

    // sum post-op scaling factor that is fused into gemm
    float gemm_beta_ = 0.f;

    // indicates if a special post processing kernel
    // should be invoked after gemm
    bool has_pp_kernel_ = false;

    // an attribute for post processing kernel
    primitive_attr_t pp_attr_;

    // auxiliary functions

    // returns gemm alpha parameter (a single value for now)
    float get_gemm_alpha(const float *primitive_scales) const {
        return gemm_applies_output_scales_ ? primitive_scales[0] : 1.f;
    }

    // returns scaling factors for post processing kernel
    const float *get_post_processing_scales(
            const float *primitive_scales) const {
        return gemm_applies_output_scales_ ? pp_attr_.output_scales_.scales_
                                           : primitive_scales;
    }
};

inline void book_acc_scratchpad(
        matmul_pd_t &pd, const params_t &params, size_t sizeof_acc_data) {
    bool is_runtime_dims
            = utils::one_of(DNNL_RUNTIME_DIM_VAL, pd.batch(), pd.M(), pd.N());
    if (!params.dst_is_acc_ && !is_runtime_dims) {
        auto scratchpad = pd.scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                sizeof_acc_data
                        * nstl::min(pd.batch(), (dim_t)dnnl_get_max_threads())
                        * pd.M() * pd.N());
    }
}

} // namespace gemm_based
} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
