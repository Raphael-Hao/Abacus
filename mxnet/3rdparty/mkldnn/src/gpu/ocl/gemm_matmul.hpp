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

#ifndef GPU_OCL_GEMM_MATMUL_HPP
#define GPU_OCL_GEMM_MATMUL_HPP

#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

namespace {
status_t create_gemm_pd(primitive_desc_t **gemm_pd, engine_t *engine,
        transpose_t transa, transpose_t transb, dim_t batch, dim_t m, dim_t n,
        dim_t k, dim_t stride_a, dim_t stride_b, dim_t stride_c, dim_t lda,
        dim_t ldb, dim_t ldc, dim_t bias_mask, data_type_t a_dt,
        data_type_t b_dt, data_type_t c_dt, data_type_t acc_dt,
        data_type_t bias_dt, const primitive_attr_t *attr) {
    gemm_desc_t gemm_desc;
    gemm_desc.primitive_kind = primitive_kind::gemm;
    gemm_desc.transa = transa;
    gemm_desc.transb = transb;
    gemm_desc.batch = batch;
    gemm_desc.m = m;
    gemm_desc.n = n;
    gemm_desc.k = k;
    gemm_desc.lda = stride_a;
    gemm_desc.ldb = stride_b;
    gemm_desc.ldc = stride_c;
    gemm_desc.lda = lda;
    gemm_desc.ldb = ldb;
    gemm_desc.ldc = ldc;
    gemm_desc.a_type = a_dt;
    gemm_desc.b_type = b_dt;
    gemm_desc.c_type = c_dt;
    gemm_desc.acc_type = acc_dt;
    gemm_desc.bias_type = bias_dt;
    gemm_desc.bias_mask = bias_mask;

    return dnnl_primitive_desc_create(
            gemm_pd, (op_desc_t *)&gemm_desc, attr, engine, nullptr);
}
} // namespace

struct gemm_matmul_t : public primitive_impl_t {
    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ocl:gemm:any", gemm_matmul_t);

        status_t init() {
            using namespace data_type;

            bool ok = set_default_formats();
            if (!ok) return status::unimplemented;

            const bool is_batched = batched();
            const dim_t MB = batch();
            const dim_t M = src_md()->dims[is_batched + 0];
            const dim_t N = dst_md()->dims[is_batched + 1];
            const dim_t K = src_md()->dims[is_batched + 1];

            const auto &dst_bd = dst_md()->format_desc.blocking;
            const auto &src_strides
                    = &src_md()->format_desc.blocking.strides[0];
            const auto &weights_strides
                    = &weights_md()->format_desc.blocking.strides[0];
            const auto &dst_strides
                    = &dst_md()->format_desc.blocking.strides[0];

            const auto bia_d = weights_md(1);
            dim_t bias_mask = 0;
            if (memory_desc_wrapper(bia_d).has_runtime_dims()) {
                bias_mask = DNNL_RUNTIME_DIM_VAL;
            } else {
                if (is_batched) bias_mask |= (bia_d->dims[0] > 1) ? 1 << 0 : 0;
                for (int d = is_batched; d < bia_d->ndims; ++d) {
                    bias_mask |= (bia_d->dims[d] > 1) ? 1 << (bia_d->ndims - d)
                                                      : 0;
                }
            }

            const transpose_t transA = src_strides[is_batched + 1] == 1
                            && src_md()->dims[is_batched + 0] > 1
                    ? transpose::notrans
                    : transpose::trans;
            const transpose_t transB = weights_strides[is_batched + 1] == 1
                            && weights_md()->dims[is_batched + 0] > 1
                    ? transpose::notrans
                    : transpose::trans;

            const dim_t lda = src_strides[is_batched
                    + (transA == transpose::notrans ? 0 : 1)];
            const dim_t ldb = weights_strides[is_batched
                    + (transB == transpose::notrans ? 0 : 1)];
            const dim_t ldc = dst_bd.strides[is_batched + 0];

            const dim_t stride_a = src_strides[0];
            const dim_t stride_b = weights_strides[0];
            const dim_t stride_c = dst_strides[0];

            auto prepare_gemm_attributes
                    = [](const primitive_attr_t &attr,
                              primitive_attr_t &gemm_attr) {
                          if (!attr.output_scales_.has_default_values()) {
                              gemm_attr.output_scales_ = attr.output_scales_;
                          }

                          auto map_gemm_zp = [&attr, &gemm_attr](
                                                     int arg, int gemm_arg) {
                              if (!attr.zero_points_.has_default_values(arg)) {
                                  dim_t count = 0;
                                  int mask = 0;
                                  const int *zero_points = nullptr;
                                  attr.zero_points_.get(
                                          arg, &count, &mask, &zero_points);
                                  gemm_attr.zero_points_.set(
                                          gemm_arg, count, mask, zero_points);
                              }
                          };

                          if (!attr.zero_points_.has_default_values()) {
                              map_gemm_zp(DNNL_ARG_SRC, DNNL_ARG_B);
                              map_gemm_zp(DNNL_ARG_WEIGHTS, DNNL_ARG_A);
                              map_gemm_zp(DNNL_ARG_DST, DNNL_ARG_C);
                          }

                          if (!attr.post_ops_.has_default_values()) {
                              gemm_attr.post_ops_ = attr.post_ops_;
                          }
                      };

            primitive_attr_t gemm_attr;
            prepare_gemm_attributes(*attr(), gemm_attr);

            const auto src_dt = desc()->src_desc.data_type;
            const auto weights_dt = desc()->weights_desc.data_type;
            const auto bias_dt = desc()->bias_desc.data_type;
            const auto dst_dt = desc()->dst_desc.data_type;
            const auto acc_dt = desc()->accum_data_type;

            bool gemm_ok = status::success
                    == create_gemm_pd(&gemm_pd_, engine(), transB, transA, MB,
                            N, M, K, stride_b, stride_a, stride_c, ldb, lda,
                            ldc, bias_mask, weights_dt, src_dt, dst_dt, acc_dt,
                            bias_dt, &gemm_attr);
            if (!gemm_ok) return status::unimplemented;

            return status::success;
        }

        primitive_desc_t *gemm_pd_ = nullptr;
    };

    virtual status_t init() override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(&gemm_);
        if (gemm_status != status::success) return gemm_status;
        return status::success;
    }

    gemm_matmul_t(const pd_t *apd) : primitive_impl_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

private:
    primitive_t *gemm_ = nullptr;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
