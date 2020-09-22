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

#ifndef MATMUL_PD_HPP
#define MATMUL_PD_HPP

#include <assert.h>

#include "dnnl.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

struct matmul_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::matmul;

    typedef matmul_pd_t base_class;
    typedef matmul_pd_t hint_class;

    matmul_pd_t(engine_t *engine, const matmul_desc_t *adesc,
            const primitive_attr_t *attr, const matmul_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc) {}

    const matmul_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override {
        return reinterpret_cast<const op_desc_t *>(this->desc());
    }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
            case query::matmul_d:
                *(const matmul_desc_t **)result = desc();
                break;
            default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    virtual arg_usage_t arg_usage(int arg) const override {
        const bool input = utils::one_of(
                arg, DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_BIAS);
        if (input) return arg_usage_t::input;

        if (arg == DNNL_ARG_DST) return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *arg_md(int arg) const override {
        switch (arg) {
            case DNNL_ARG_SRC: return src_md(0);
            case DNNL_ARG_WEIGHTS: return weights_md(0);
            case DNNL_ARG_BIAS: return weights_md(1);
            case DNNL_ARG_DST: return dst_md(0);
            default: return primitive_desc_t::arg_md(arg);
        }
    }

    virtual const memory_desc_t *src_md(int index = 0) const override {
        return index == 0 ? &src_md_ : &glob_zero_md;
    }

    virtual const memory_desc_t *weights_md(int index = 0) const override {
        return utils::pick(index, &weights_md_, &bias_md_, &glob_zero_md);
    }

    virtual const memory_desc_t *dst_md(int index = 0) const override {
        return index == 0 ? &dst_md_ : &glob_zero_md;
    }

    virtual int n_inputs() const override { return 2 + with_bias(); }
    virtual int n_outputs() const override { return 1; }

    bool has_zero_dim_memory() const {
        return memory_desc_wrapper(dst_md(0)).has_zero_dim();
    }

    int ndims() const { return dst_md_.ndims; }

    bool with_bias() const { return bias_md_.ndims != 0; }
    bool batched() const { return ndims() == 3; }

    dim_t batch() const { return batched() ? dst_md_.dims[0] : 1; }
    dim_t M() const { return dst_md_.dims[batched() + 0]; }
    dim_t N() const { return dst_md_.dims[batched() + 1]; }
    dim_t K() const { return src_md_.dims[batched() + 1]; }

    bool is_bias_1xN() const {
        if (!with_bias()) return false;

        const auto &bia_md = *weights_md(1);
        return bia_md.dims[0] == 1
                && IMPLICATION(batched(), bia_md.dims[1] == 1)
                && bia_md.dims[batched() + 1] == dst_md()->dims[batched() + 1];
    }

protected:
    matmul_desc_t desc_;

    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;

    // temporary solution to deal with format `any`
    bool set_default_formats() {
        for (auto md : {&src_md_, &weights_md_, &bias_md_, &dst_md_}) {
            memory_desc_wrapper mdw(md);
            if (mdw.format_any()) {
                if (mdw.has_runtime_dims_or_strides()) return false;
                status_t status = memory_desc_init_by_strides(*md, nullptr);
                if (status != status::success) return false;
            }
        }

        return true;
    }
};

} // namespace impl
} // namespace dnnl

#endif
