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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"
#include "simple_q10n.hpp"
#include "type_helpers.hpp"

#include "ref_binary.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

typedef struct {
    alg_kind_t alg;
    bool do_sum;
    float sum_scale;
    const std::unique_ptr<ref_eltwise_scalar_fwd_t> &eltwise_ker;
    const scales_t *scales;
    bool do_scale_src0;
    bool do_scale_src1;
} params_t;

template <typename data_t>
data_t compute_alg(data_t x, data_t y, alg_kind_t alg) {
    data_t d = 0;
    if (alg == alg_kind::binary_add) {
        d = x + y;
    } else if (alg == alg_kind::binary_mul) {
        d = x * y;
    } else if (alg == alg_kind::binary_max) {
        d = nstl::max(x, y);
    } else if (alg == alg_kind::binary_min) {
        d = nstl::min(x, y);
    } else {
        assert(!"not supported operation!");
    }
    return d;
}

template <typename src0_data_t, typename src1_data_t, typename dst_data_t>
typename utils::enable_if<nstl::is_integral<src0_data_t>::value>::type
perform_op(
        dst_data_t *dst, src0_data_t x, src1_data_t y, const params_t &params) {
    float x_f = (float)x;
    float y_f = (float)y;
    float dst_f = (float)dst[0];

    if (params.do_scale_src0) x_f *= params.scales[0].scales_[0];
    if (params.do_scale_src1) y_f *= params.scales[1].scales_[0];

    float acc = compute_alg<float>(x_f, y_f, params.alg);
    float scaled_dst = params.do_sum ? params.sum_scale * dst_f : 0.f;
    acc += scaled_dst;
    if (params.eltwise_ker) acc = params.eltwise_ker->compute_scalar(acc);
    dst[0] = qz_a1b0<float, dst_data_t>()(acc);
}

template <typename src0_data_t, typename src1_data_t, typename dst_data_t>
typename utils::enable_if<!nstl::is_integral<src0_data_t>::value>::type
perform_op(
        dst_data_t *dst, src0_data_t x, src1_data_t y, const params_t &params) {
    float acc = compute_alg<float>(x, y, params.alg);
    float scaled_dst = params.do_sum ? params.sum_scale * dst[0] : 0.f;
    acc += scaled_dst;
    if (params.eltwise_ker) acc = params.eltwise_ker->compute_scalar(acc);
    dst[0] = (dst_data_t)acc;
}

template <data_type_t src0_type, data_type_t src1_type, data_type_t dst_type>
void ref_binary_t<src0_type, src1_type, dst_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    const auto src0 = CTX_IN_MEM(const src0_data_t *, DNNL_ARG_SRC_0);
    const auto src1 = CTX_IN_MEM(const src1_data_t *, DNNL_ARG_SRC_1);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src0_d(pd()->src_md(0));
    const memory_desc_wrapper src1_d(pd()->src_md(1));

    const auto alg = pd()->desc()->alg_kind;

    // 0:src0 1:src1
    constexpr int nargs = 2;
    scales_t scales[nargs];
    int args[nargs] = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};

    if (nstl::is_integral<src0_data_t>::value)
        scales[0] = pd()->attr()->scales_.get(args[0]);

    if (nstl::is_integral<src0_data_t>::value)
        scales[1] = pd()->attr()->scales_.get(args[1]);

    bool do_scale_src0 = !scales[0].has_default_values();
    bool do_scale_src1 = !scales[1].has_default_values();

    const dims_t &dims_bcast = pd()->broadcast_dims();
    const dims_t &dims_A = src0_d.dims();
    const int ndims = pd()->ndims();
    const auto nelems_A = src0_d.nelems();
    const bool is_tensor_op = pd()->is_tensor_op();

    const auto &po = pd()->attr()->post_ops_;
    const bool do_sum = po.contain(primitive_kind::sum, 0)
            && po.entry_[0].sum.scale != 0.f;
    const float sum_scale = do_sum ? po.entry_[0].sum.scale : 0.f;

    params_t params {alg, do_sum, sum_scale, eltwise_ker_, scales,
            do_scale_src0, do_scale_src1};

    auto map_idx_B = [&](dim_t off) {
        dims_t dims;
        for (int d = ndims - 1; d >= 0; --d) {
            dims[d] = off % dims_A[d];
            off /= dims_A[d];
        }
        assert(off == 0);

        for (int d = 0; d < ndims; ++d) {
            dims[d] *= (!dims_bcast[d]);
        }

        return src1_d.off_v(dims);
    };

    parallel_nd(nelems_A, [&](dim_t i) {
        auto off_A = src0_d.off_l(i);
        auto off_B = is_tensor_op ? src1_d.off_l(i) : map_idx_B(i);
        perform_op(&dst[off_A], src0[off_A], src1[off_B], params);
    });
}

using namespace data_type;

template struct ref_binary_t<f32>;
template struct ref_binary_t<bf16>;
template struct ref_binary_t<s8, u8, s8>;
template struct ref_binary_t<s8, s8, s8>;
template struct ref_binary_t<u8, s8, u8>;
template struct ref_binary_t<u8, u8, u8>;

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
