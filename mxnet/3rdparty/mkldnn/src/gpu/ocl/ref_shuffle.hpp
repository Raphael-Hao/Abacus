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

#ifndef GPU_OCL_REF_SHUFFLE_HPP
#define GPU_OCL_REF_SHUFFLE_HPP

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_shuffle_pd.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_shuffle_t : public primitive_impl_t {
    struct pd_t : public gpu_shuffle_pd_t {
        using gpu_shuffle_pd_t::gpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_shuffle_t);

        status_t init() {
            using namespace format_tag;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            bool ok = true
                    && utils::one_of(
                            (int)types::data_type_size(data_md()->data_type), 1,
                            2, 4)
                    && attr()->has_default_values()
                    && IMPLICATION(
                            desc()->data_desc.data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && IMPLICATION(!is_fwd(), set_default_formats_common());
            if (!ok) return status::unimplemented;

            return init_conf();
        }

        status_t init_conf();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        shuffle_conf_t conf;
        offsets_t off;
    };

    ref_shuffle_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        compute_engine->create_kernel(&kernel_, "ref_shuffle", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    ~ref_shuffle_t() {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_<format_tag::any>(ctx);
    }

private:
    template <format_tag_t tag>
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
