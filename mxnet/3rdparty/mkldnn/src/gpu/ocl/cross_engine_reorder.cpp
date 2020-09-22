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

#include "gpu/ocl/cross_engine_reorder.hpp"

#include "common/utils.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t cross_engine_reorder_t::pd_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad) const {
    using namespace memory_tracking::names;
    const memory_desc_wrapper wspace_md(
            desc()->src_engine_kind == reorder_engine_kind_ ? dst_md()
                                                            : src_md());
    scratchpad.book(
            memory_tracking::names::key_reorder_cross_space, wspace_md.size());
    return status::success;
}

status_t cross_engine_reorder_t::pd_t::init() {
    bool args_ok = src_engine() != dst_engine()
            && utils::one_of(engine_kind::gpu, src_engine()->kind(),
                    dst_engine()->kind());

    if (!args_ok) return status::unimplemented;

    memory_desc_wrapper src_mdw(src_md());
    memory_desc_wrapper dst_mdw(dst_md());

    bool with_sum_ab = alpha() != 1.0 || beta() != 0.0;
    do_reorder_ = with_sum_ab || src_mdw != dst_mdw;

    engine_t *reorder_engine = src_engine()->kind() == engine_kind::gpu
            ? src_engine()
            : dst_engine();

    auto r_impls = reorder_engine->get_reorder_implementation_list(
            src_md(), dst_md());
    const primitive_attr_t r_attr(*attr());
    for (auto r = r_impls; *r; ++r) {
        reorder_pd_t *r_pd = nullptr;
        if ((*r)(&r_pd, reorder_engine, &r_attr, reorder_engine, src_md(),
                    reorder_engine, dst_md())
                == status::success) {
            reorder_.reset(r_pd);
            break;
        }
    }

    if (!reorder_) return status::unimplemented;
    if (do_reorder_) {
        auto scratchpad = scratchpad_registry().registrar();
        init_scratchpad(scratchpad);
    }

    return status::success;
}

status_t cross_engine_reorder_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);

    std::unique_ptr<memory_t> wspace;
    if (pd()->do_reorder_) {
        auto src_engine_kind = pd()->desc()->src_engine_kind;
        auto reorder_engine_kind = pd()->reorder_engine_kind_;
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                key_reorder_cross_space);
        auto wspace_md = src_engine_kind == reorder_engine_kind
                ? pd()->dst_md()
                : pd()->src_md();
        auto wspace_ptr = scratchpad->data_handle();
        wspace.reset(new memory_t(ctx.stream()->engine(), wspace_md,
                memory_flags_t::use_runtime_ptr, wspace_ptr));
    }

    auto exec_reorder = [&](const memory_t *src_mem, const memory_t *dst_mem) {
        exec_args_t r_args;
        r_args[DNNL_ARG_SRC]
                = memory_arg_t {const_cast<memory_t *>(src_mem), true};
        r_args[DNNL_ARG_DST]
                = memory_arg_t {const_cast<memory_t *>(dst_mem), false};

        exec_ctx_t r_ctx(ctx.stream(), std::move(r_args));
        return reorder_->execute(r_ctx);
    };

    status_t status = status::success;
    if (pd()->desc()->src_engine_kind == engine_kind::gpu) {
        // GPU -> CPU or GPU -> GPU
        if (pd()->do_reorder_) {
            status = exec_reorder(ctx.input(DNNL_ARG_FROM), wspace.get());
        }
        if (status == status::success) {
            memory_desc_wrapper dst_mdw(pd()->dst_md());
            status = compute_stream->copy(
                    pd()->do_reorder_ ? *wspace->memory_storage() : src, dst,
                    dst_mdw.size());
        }
    } else {
        // CPU -> GPU
        memory_desc_wrapper src_mdw(pd()->src_md());
        status = compute_stream->copy(src,
                pd()->do_reorder_ ? *wspace->memory_storage() : dst,
                src_mdw.size());
        if (status == status::success && pd()->do_reorder_) {
            status = exec_reorder(wspace.get(), ctx.output(DNNL_ARG_TO));
        }
    }
    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
