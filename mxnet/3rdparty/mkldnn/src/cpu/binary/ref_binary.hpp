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

#ifndef REF_BINARY_HPP
#define REF_BINARY_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_isa_traits.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_binary_pd.hpp"

#include "ref_eltwise.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t src0_type, data_type_t src1_type = src0_type,
        data_type_t dst_type = src0_type>
struct ref_binary_t : public primitive_impl_t {
    struct pd_t : public cpu_binary_pd_t {
        using cpu_binary_pd_t::cpu_binary_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_binary_t);

        status_t init() {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            bool ok = src0_type == src_md(0)->data_type
                    && src1_type == src_md(1)->data_type
                    && dst_type == dst_md()->data_type
                    && IMPLICATION(utils::everyone_is(bf16, src0_type,
                                           src1_type, dst_type),
                            mayiuse(avx512_core))
                    && set_default_params() == status::success
                    && IMPLICATION(utils::one_of(src0_type, f32, bf16),
                            attr()->has_default_values(sm::post_ops))
                    && IMPLICATION(utils::one_of(src0_type, s8, u8),
                            attr()->has_default_values(
                                    sm::post_ops | sm::scales))
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask())
                    && attr_post_ops_ok();
            if (!ok) return status::unimplemented;

            return status::success;
        }

    private:
        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }
    };

    ref_binary_t(const pd_t *apd) : primitive_impl_t(apd) {
        int e_idx = pd()->attr()->post_ops_.find(primitive_kind::eltwise);
        if (e_idx != -1)
            eltwise_ker_.reset(new ref_eltwise_scalar_fwd_t(
                    pd()->attr()->post_ops_.entry_[e_idx].eltwise));
    }

    ~ref_binary_t() {}

    using src0_data_t = typename prec_traits<src0_type>::type;
    using src1_data_t = typename prec_traits<src1_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_ref(ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    void execute_ref(const exec_ctx_t &ctx) const;

    std::unique_ptr<ref_eltwise_scalar_fwd_t> eltwise_ker_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
