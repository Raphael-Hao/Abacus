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

#include "gpu/ocl/ref_convolution.hpp"

#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

static status_t init_conf_common(
        conv_conf_t &conf, offsets_t &off, const convolution_pd_t *pd) {
    const convolution_desc_t &cd = *pd->desc();
    const memory_desc_t &src_md = *pd->invariant_src_md();
    const memory_desc_t &weights_md = *pd->invariant_wei_md();
    const memory_desc_t &dst_md = *pd->invariant_dst_md();
    const memory_desc_t &bias_md = *pd->invariant_bia_md();
    const primitive_attr_t &attr = *pd->attr();

    set_default_conf(conf, cd, src_md, weights_md, dst_md, bias_md, attr);

    set_offsets(src_md, off.src_off);
    set_offsets(weights_md, off.wei_off);
    set_offsets(dst_md, off.dst_off);

    int oc_idx = (int)conf.with_groups;
    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(pd->engine());
    switch (cd.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference: {
            conf.dispatch = compute_engine->create_dispatch(&dst_md);
            conf.dispatch.define_dim("MB", 0, conf.mb);
            conf.dispatch.define_dim("G", 1, conf.ngroups);
            conf.dispatch.define_dim("OC", 1, conf.oc);
            conf.dispatch.define_dim(
                    "OD", nstl::max(2, conf.ndims - 3), conf.od);
            conf.dispatch.define_dim(
                    "OH", nstl::max(2, conf.ndims - 2), conf.oh);
            conf.dispatch.define_dim(
                    "OW", nstl::max(2, conf.ndims - 1), conf.ow);
            conf.dispatch.generate();
            break;
        }
        case prop_kind::backward_data: {
            conf.dispatch = compute_engine->create_dispatch(&src_md);
            conf.dispatch.define_dim_with_nesting_level(
                    "IC", conf.ndims, conf.ic);
            conf.dispatch.define_dim("MB", conf.mb);
            conf.dispatch.define_dim("G", conf.ngroups);
            conf.dispatch.define_dim(
                    "ID", nstl::max(2, conf.ndims - 3), conf.id);
            conf.dispatch.define_dim(
                    "IH", nstl::max(2, conf.ndims - 2), conf.ih);
            conf.dispatch.define_dim(
                    "IW", nstl::max(2, conf.ndims - 1), conf.iw);
            conf.dispatch.generate();
            break;
        }
        case prop_kind::backward_weights: {
            conf.dispatch = compute_engine->create_dispatch(&weights_md);
            conf.dispatch.define_dim("G", 0, conf.ngroups);
            conf.dispatch.define_dim("OC", oc_idx, conf.oc);
            conf.dispatch.define_dim("IC", oc_idx + 1, conf.ic);
            conf.dispatch.define_dim(
                    "KD", oc_idx + nstl::max(2, conf.ndims - 3), conf.kd);
            conf.dispatch.define_dim(
                    "KH", oc_idx + nstl::max(2, conf.ndims - 2), conf.kh);
            conf.dispatch.define_dim(
                    "KW", oc_idx + nstl::max(2, conf.ndims - 1), conf.kw);
            conf.dispatch.generate();
            break;
        }
        default: break;
    }

    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const conv_conf_t &conf, const offsets_t &off) {
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("WITH_GROUPS", conf.with_groups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OD", conf.od);
    kernel_ctx.define_int("OH", conf.oh);
    kernel_ctx.define_int("OW", conf.ow);
    kernel_ctx.define_int("KD", conf.kd);
    kernel_ctx.define_int("KH", conf.kh);
    kernel_ctx.define_int("KW", conf.kw);
    kernel_ctx.define_int("SD", conf.stride_d);
    kernel_ctx.define_int("SH", conf.stride_h);
    kernel_ctx.define_int("SW", conf.stride_w);
    kernel_ctx.define_int("PD", conf.f_pad);
    kernel_ctx.define_int("PH", conf.t_pad);
    kernel_ctx.define_int("PW", conf.l_pad);
    kernel_ctx.define_int("PD_R", conf.back_pad);
    kernel_ctx.define_int("PH_R", conf.b_pad);
    kernel_ctx.define_int("PW_R", conf.r_pad);
    kernel_ctx.define_int("DD", conf.dilate_d);
    kernel_ctx.define_int("DH", conf.dilate_h);
    kernel_ctx.define_int("DW", conf.dilate_w);
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("IS_FWD",
            utils::one_of(conf.prop_kind, prop_kind::forward_inference,
                    prop_kind::forward_training));
    kernel_ctx.define_int(
            "IS_BWD_D", conf.prop_kind == prop_kind::backward_data);
    kernel_ctx.define_int(
            "IS_BWD_W", conf.prop_kind == prop_kind::backward_weights);

    def_offsets(off.src_off, kernel_ctx, "SRC", conf.ndims);
    def_offsets(off.wei_off, kernel_ctx, "WEI", conf.ndims + conf.with_groups);
    def_offsets(off.bias_off, kernel_ctx, "BIA", 1);
    def_offsets(off.dst_off, kernel_ctx, "DST", conf.ndims);

    def_dispatch(kernel_ctx, conf.dispatch);

    switch (conf.prop_kind) {
        case prop_kind::forward_training:
        case prop_kind::forward_inference:
            kernel_ctx.set_data_type(conf.dst_data_type);
            break;
        case prop_kind::backward_data:
            kernel_ctx.set_data_type(conf.src_data_type);
            break;
        case prop_kind::backward_weights:
            kernel_ctx.set_data_type(conf.weights_data_type);
            break;
        default: break;
    }

    def_data_type(kernel_ctx, conf.src_data_type, "SRC");
    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");
    def_data_type(kernel_ctx, conf.bias_data_type, "BIA");
    def_data_type(kernel_ctx, conf.dst_data_type, "DST");
    def_data_type(kernel_ctx, conf.acc_data_type, "ACC");

    if (conf.with_eltwise || conf.with_post_sum_eltwise) {
        def_postops(kernel_ctx, conf.eltwise.alg);
    }
    kernel_ctx.define_int("WITH_ELTWISE", conf.with_eltwise);
    kernel_ctx.define_int("WITH_SUM", conf.with_sum);
    kernel_ctx.define_int("WITH_POST_SUM_ELTWISE", conf.with_post_sum_eltwise);

    kernel_ctx.define_int("SCALES_COMMON", conf.with_common_scales);
    kernel_ctx.define_int("SCALES_PER_OC", conf.with_per_oc_scales);

    return status::success;
}

status_t ref_convolution_fwd_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t ref_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {

    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    auto eltwise_alpha = pd()->eltwise_alpha();
    auto eltwise_beta = pd()->eltwise_beta();
    auto eltwise_scale = pd()->eltwise_scale();
    auto sum_scale = pd()->sum_scale();

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    arg_list.set(4, eltwise_alpha);
    arg_list.set(5, eltwise_beta);
    arg_list.set(6, eltwise_scale);
    arg_list.set(7, sum_scale);
    if (utils::one_of(
                pd()->src_md()->data_type, data_type::u8, data_type::s8)) {
        if (pd()->with_common_scales()) {
            float scales = pd()->attr()->output_scales_.scales_[0];
            arg_list.set(8, scales);
        }
        if (pd()->with_per_oc_scales()) {
            arg_list.set(8, *scales_mem_->memory_storage());
        }
    }

    auto nd_range = pd()->conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);
    return status;
}

status_t ref_convolution_bwd_data_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t ref_convolution_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {

    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, bias);

    auto nd_range = pd()->conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t ref_convolution_bwd_weights_t::pd_t::init_conf() {
    return init_conf_common(conf, off, this);
}

status_t ref_convolution_bwd_weights_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf, off);
}

status_t ref_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {

    compute::compute_stream_t *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_weights);
    arg_list.set(2, diff_bias);
    arg_list.set(3, diff_dst);

    auto nd_range = pd()->conf.dispatch.nd_range();
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
