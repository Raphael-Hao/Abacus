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

#include "gpu/ocl/gen9_convolution.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"

using namespace dnnl::impl::memory_tracking::names;

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using math::saturate;

status_t gen9_convolution_fwd_t::pd_t::init_conf() {
    using namespace dnnl::impl::format_tag;
    using namespace data_type;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *src_md(), *weights_md(), *dst_md(),
            *weights_md(1), *attr());

    const bool is_nhwc
            = src_mdw.matches_one_of_tag(nwc, nhwc, ndhwc) != format_tag::undef
            || dst_mdw.matches_one_of_tag(nwc, nhwc, ndhwc)
                    != format_tag::undef;

    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);
    conf.is_depthwise = is_depthwise;

    if (is_nhwc && (is_depthwise || is_1stconv)) return status::unimplemented;

    if (is_1stconv || conf.with_groups) {
        conf.ic = conf.ic_without_padding;
        if (is_1stconv && conf.oc_without_padding % 16 != 0)
            conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
        else
            conf.oc = conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }

    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise) conf.ngroups = utils::rnd_up(conf.ngroups, 16);

    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);

    const bool is_16oc = conf.oc % 16 == 0;
    const bool is_16ic = conf.ic % 16 == 0;
    const bool use_16mb_unroll = !is_nhwc
            && !(conf.mb == 1 || conf.mb % 16 != 0) && !is_1stconv
            && ((is_16ic && is_16oc) || is_dw_16g)
            && IMPLICATION(src_mdw.data_type() == f16, conf.mb % 32 == 0)
            && IMPLICATION(src_mdw.data_type() == f16 && conf.is_depthwise,
                    conf.ngroups % 16 == 0);

    const bool is_32oc
            = IMPLICATION(src_mdw.data_type() == f16, conf.oc % 32 == 0);

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;

    if (is_nhwc && src_mdw.data_type() == f32)
        conf.ver = ver_nhwc;
    else if (use_16mb_unroll)
        conf.ver = ver_16mb16c;
    else if ((is_16oc && is_16ic) || is_dw_16g)
        conf.ver = ver_8ow16c;
    else if (is_1stconv && is_16oc)
        conf.ver = ver_1stconv;
    else
        return status::unimplemented;

    status_t status = status::success;
    conf.ocb = 1;

    switch (conf.ver) {
        case ver_nhwc:
            switch (src_mdw.data_type()) {
                case f32: {
                    conf.mb_block = 1;
                    conf.oc_block = 16;
                    conf.ic_block = 16;

                    int max_ow_block
                            = (conf.kw > 1 && conf.stride_w > 1) ? 18 : 24;

                    if (conf.oc <= 64 && conf.ic <= 64) max_ow_block = 8;

                    conf.ow_block = utils::max_div(conf.ow, max_ow_block);

                    if (conf.ow_block <= 8) {
                        int max_tail = 0;
                        for (int j = 8; j < max_ow_block; j++) {
                            if (conf.ow % j > max_tail) {
                                max_tail = conf.ow % j;
                                conf.ow_block = j;
                            }
                        }
                    }
                    if (conf.ow_block <= 8) conf.ow_block = 8;
                    if (conf.ow <= 8 || conf.oc <= 32) conf.ow_block = 8;

                    conf.oh_block = 1;
                    conf.sub_group_size = 16;
                    conf.lws_d[0] = 16;
                    conf.lws_d[1] = 1;
                    conf.lws_d[2] = 1;

                    int max_oc_block
                            = (conf.ic * conf.kh * conf.kw > 2048) ? 12 : 16;
                    conf.ocb = conf.oc_block
                            * utils::max_div(
                                    conf.oc / conf.oc_block, max_oc_block);

                    conf.gws_d[0] = conf.ocb;
                    conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                            * utils::div_up(conf.ow, conf.ow_block) * conf.od;
                    conf.gws_d[2]
                            = conf.mb * (conf.oc / conf.ocb) * conf.ngroups;
                    break;
                }
                default: return status::unimplemented; break;
            }
            break;
        case ver_16mb16c:
            conf.mb_block = 16;
            if (src_mdw.data_type() == f16 && conf.mb % 32 != 0) {
                conf.mb_block = 1;
                conf.oc_block = 16;
                conf.ic_block = 16;
                conf.ow_block = 8;
                conf.oh_block = 1;
                conf.sub_group_size = 16;
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.ngroups * conf.oc;
                conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                        * utils::div_up(conf.ow, conf.ow_block) * conf.od;
                conf.gws_d[2] = conf.mb;
            } else {
                conf.oc_block = 16;
                conf.ic_block = 16;
                conf.sub_group_size = 16;
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.oc * conf.ngroups;
                conf.gws_d[1] = conf.oh * conf.ow * conf.od;
                conf.gws_d[2]
                        = (src_mdw.data_type() == f16 && !conf.is_depthwise)
                        ? conf.mb / (conf.mb_block * 2)
                        : conf.mb / (conf.mb_block * 1);
            }

#ifdef DEBUG_PRINT
            printf("LWS = %ld\n",
                    conf.lws_d[0] * conf.lws_d[2] * conf.lws_d[1]);
            fflush(0);
            printf("LWS GWS: (%ld %ld %ld) (%ld %ld %ld)\n", conf.lws_d[0],
                    conf.lws_d[1], conf.lws_d[2], conf.gws_d[0], conf.gws_d[1],
                    conf.gws_d[2]);
#endif

            break;
        case ver_1stconv:
            if (src_mdw.data_type() == f16) {
                //use single blocked kernel when mb % 32 != 0
                conf.mb_block = conf.mb % 32 == 0 ? 16 : 1;
                conf.oc_block = 16;
                conf.ic_block = 16;
                conf.ow_block = 8;
                while (conf.ow_block > 1) {
                    if (conf.stride_w * conf.ow_block
                                    + conf.kw * (1 + conf.dilate_w)
                            > 32)
                        conf.ow_block--;
                    else
                        break;
                };
                conf.oh_block = 1;
                conf.sub_group_size = 16;
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0]
                        = (is_32oc ? (conf.oc / 2) : conf.oc) * conf.ngroups;
                conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                        * utils::div_up(conf.ow, conf.ow_block) * conf.od;
                conf.gws_d[2] = conf.mb % 2 == 0 ? conf.mb / 2
                                                 : conf.mb; // unroll mb by 2
                break;
            } else if (!conf.is_depthwise) {
                conf.mb_block = (conf.mb % 16 == 0) ? 16 : 1;
                conf.oc_block = 16;
                conf.ic_block = 1;
                conf.ow_block = 8;
                while (conf.ow_block > 1) {
                    if (conf.stride_w * conf.ow_block
                                    + conf.kw * (1 + conf.dilate_w)
                            > 32)
                        conf.ow_block--;
                    else
                        break;
                };
                conf.oh_block = 1;
                conf.sub_group_size = 16;
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.ocb = (conf.oc % 32 == 0) ? 32 : 16;
                conf.gws_d[0] = 16;
                conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                        * utils::div_up(conf.ow, conf.ow_block) * conf.od;
                conf.gws_d[2] = conf.mb * (conf.oc / conf.ocb) * conf.ngroups;

                break;
            }
        case ver_8ow16c:
            switch (src_mdw.data_type()) {
                case f32:
                    conf.mb_block = 1;
                    conf.oc_block = 16;
                    conf.ic_block = 16;
                    if (conf.is_depthwise) {
                        conf.ow_block = utils::max_div(conf.ow, 8);
                    } else {
                        conf.ow_block
                                = nstl::max(8, utils::max_div(conf.ow, 16));
                    }
                    conf.oh_block = 1;
                    conf.sub_group_size = 16;
                    conf.lws_d[0] = 16;
                    conf.lws_d[1] = 1;
                    conf.lws_d[2] = 1;
                    if (conf.is_depthwise) {
                        conf.ocb = conf.ngroups;
                    } else {
                        conf.ocb = 128;
                        while (conf.ocb > 16) {
                            if (conf.oc % conf.ocb == 0)
                                break;
                            else
                                conf.ocb /= 2;
                        }
                    }
                    conf.gws_d[0] = conf.ocb;
                    conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                            * utils::div_up(conf.ow, conf.ow_block) * conf.od;
                    if (conf.is_depthwise) {
                        conf.gws_d[2] = conf.mb * (conf.ngroups / conf.ocb);
                    } else {
                        conf.gws_d[2]
                                = conf.mb * (conf.oc / conf.ocb) * conf.ngroups;
                    }
                    break;
                case f16:
                    conf.mb_block = 1;
                    conf.oc_block = 16;
                    conf.ic_block = 16;
                    if (conf.is_depthwise)
                        conf.ow_block = utils::max_div(conf.ow, 8);
                    else
                        conf.ow_block
                                = nstl::max(8, utils::max_div(conf.ow, 16));
                    conf.oh_block = 1;
                    conf.sub_group_size = 16;
                    conf.lws_d[0] = 16;
                    conf.lws_d[1] = 1;
                    conf.lws_d[2] = 1;
                    conf.ocb = 128;
                    if (conf.is_depthwise) {
                        conf.ocb = conf.ngroups;
                    } else {
                        while (conf.ocb > 16) {
                            if (conf.oc % conf.ocb == 0)
                                break;
                            else
                                conf.ocb /= 2;
                        }
                    }
                    conf.gws_d[0] = conf.ocb;
                    conf.gws_d[1] = utils::div_up(conf.oh, conf.oh_block)
                            * utils::div_up(conf.ow, conf.ow_block) * conf.od;
                    if (conf.is_depthwise) {
                        conf.gws_d[2] = conf.mb * (conf.ngroups / conf.ocb);
                    } else {
                        conf.gws_d[2]
                                = conf.mb * (conf.oc / conf.ocb) * conf.ngroups;
                    }
                    break;
                default: return status::unimplemented;
            }
            break;
        default: status = status::unimplemented;
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_nhwc:
            src_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            dst_tag = utils::pick(conf.ndims - 3, nwc, nhwc, ndhwc);
            wei_tag = conf.with_groups ? utils::pick(conf.ndims - 3, gOIw16i16o,
                              gOIhw16i16o, gOIdhw16i16o)
                                       : utils::pick(conf.ndims - 3, OIw16i16o,
                                               OIhw16i16o, OIdhw16i16o);
            break;
        case ver_1stconv:
            src_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);
            dst_tag = conf.mb_block % 16 == 0
                    ? utils::pick(
                            conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c)
                    : utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.with_groups
                    ? utils::pick(conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(conf.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            break;
        case ver_16mb16c:
            if (utils::one_of(src_mdw.data_type(), f16)) {
                if (conf.mb % 32 == 0) {
                    src_tag = utils::pick(
                            conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                    dst_tag = utils::pick(
                            conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                    wei_tag = conf.is_depthwise
                            ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g,
                                    Goidhw16g)
                            : (conf.with_groups ? utils::pick(conf.ndims - 3,
                                       gOIw8i16o2i, gOIhw8i16o2i, gOIdhw8i16o2i)
                                                : utils::pick(conf.ndims - 3,
                                                        OIw8i16o2i, OIhw8i16o2i,
                                                        OIdhw8i16o2i));
                } else {
                    src_tag = utils::pick(
                            conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
                    dst_tag = utils::pick(
                            conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
                    wei_tag = conf.with_groups
                            ? utils::pick(conf.ndims - 3, gIOw16i16o,
                                    gIOhw16i16o, gIOdhw16i16o)
                            : utils::pick(conf.ndims - 3, IOw16i16o, IOhw16i16o,
                                    IOdhw16i16o);
                }
            } else {
                src_tag = utils::pick(
                        conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                dst_tag = utils::pick(
                        conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
                wei_tag = conf.is_depthwise
                        ? utils::pick(
                                conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                        : (conf.with_groups ? utils::pick(conf.ndims - 3,
                                   gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                            : utils::pick(conf.ndims - 3,
                                                    IOw16i16o, IOhw16i16o,
                                                    IOdhw16i16o));
            }
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
                                        : utils::pick(conf.ndims - 3, OIw16i16o,
                                                OIhw16i16o, OIdhw16i16o));
            break;
        default: status = status::unimplemented;
    }
    if (status != status::success) return status;

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    conf.is_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    return status;
}

status_t gen9_convolution_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
    kernel_ctx.define_int("G", conf.ngroups);
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
    kernel_ctx.define_int("OW_PADDED", utils::rnd_up(conf.ow, 4));
    kernel_ctx.define_int("OC_PADDED", conf.oc);
    kernel_ctx.define_int("OCB", conf.ocb);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OH_BLOCK", conf.oh_block);
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("OW_LAST", utils::rnd_dn(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OWB", utils::div_up(conf.ow, conf.ow_block));
    kernel_ctx.define_int("OHB", utils::div_up(conf.oh, conf.oh_block));
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int(
            "WITH_ELTWISE", conf.with_eltwise || conf.with_post_sum_eltwise);
    if (conf.with_eltwise || conf.with_post_sum_eltwise)
        def_postops(kernel_ctx, conf.eltwise.alg);
    kernel_ctx.define_int("WITH_SUM", conf.with_sum);
    kernel_ctx.define_int("SUM_SCALE", conf.sum_scale == 1.0);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("IC_WO_PADDING", conf.ic_without_padding);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("OC_GROUP", conf.lws_d[0] / 8);
    kernel_ctx.define_int("MB_GROUP", 1);
    kernel_ctx.define_int("SP_GROUP", conf.lws_d[1]);
    if (conf.kw == 1)
        kernel_ctx.define_int("SRC_SP_GROUP", conf.lws_d[1] + conf.kw - 1);
    else
        kernel_ctx.define_int(
                "SRC_SP_GROUP", conf.stride_w * (conf.lws_d[1] - 1) + conf.kw);

    const int use_fast_path = 1 && conf.scale_idx_mult == 0 && conf.ngroups == 1
            && !conf.with_bias;
    kernel_ctx.define_int("USE_FAST_PATH", use_fast_path);
    kernel_ctx.define_int("SCALE_IDX_MULT", conf.scale_idx_mult);

    kernel_ctx.set_data_type(conf.src_data_type);

    switch (conf.ver) {
        case ver_16mb16c: kernel_ctx.define_int("VER_16MB16C", 1); break;
        case ver_1stconv:
        case ver_8ow16c: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    if (conf.is_nchw)
        kernel_ctx.define_int("NCHW", 1);
    else if (conf.is_nhwc)
        kernel_ctx.define_int("NHWC", 1);

    kernel_ctx.print_options();
    return status::success;
}

status_t gen9_convolution_bwd_data_t::pd_t::init_conf() {
    using namespace dnnl::impl::format_tag;
    using namespace data_type;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(diff_src_md());
    const memory_desc_wrapper weights_mdw(weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bias_mdw(weights_md(1));

    set_default_conf(conf, cd, *diff_src_md(), *weights_md(), *diff_dst_md(),
            *weights_md(1), *attr());

    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);
    conf.is_depthwise = is_depthwise;

    if (is_1stconv || conf.with_groups) {
        conf.ic = conf.ic_without_padding;
        conf.oc = conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }

    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise) conf.ngroups = utils::rnd_up(conf.ngroups, 16);
    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);

    const bool is_16ic = conf.ic % 16 == 0;
    const bool is_16oc = conf.oc % 16 == 0;
    const bool use_16mb_unroll = !(conf.mb == 1 || conf.mb % 16 != 0)
            && !is_1stconv && ((is_16ic && is_16oc) || is_dw_16g);

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.icb = 1;
    if (use_16mb_unroll)
        conf.ver = ver_16mb16c;
    else if (conf.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g))
        conf.ver = ver_8ow16c;
    else
        return status::unimplemented;

    status_t status = status::success;

    switch (conf.ver) {
        case ver_16mb16c:
            conf.mb_block = 16;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.od_block = 1;
            conf.ih_block = 1;
            conf.iw_block = 1;
            conf.sub_group_size = 16;
            if (conf.is_depthwise) {
                conf.icb = conf.ngroups;
                conf.lws_d[0] = 1;
                conf.lws_d[1] = 16;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.ih * conf.iw * conf.id;
                conf.gws_d[1] = conf.ic * conf.ngroups;
                conf.gws_d[2] = conf.mb / 16;
            } else {
                conf.icb = 64;
                while (conf.icb > 16) {
                    if (conf.ic % conf.icb == 0) break;
                    conf.icb /= 2;
                }
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.icb;
                conf.gws_d[1] = conf.ih * conf.iw * conf.id;
                conf.gws_d[2]
                        = conf.mb / 16 * (conf.ic / conf.icb) * conf.ngroups;
            }
            break;
        case ver_8ow16c:
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.od_block = 1;
            conf.ih_block = 1;
            conf.iw_block = nstl::max(8, utils::max_div(conf.iw, 16));
            conf.sub_group_size = 16;
            if (conf.is_depthwise) {
                conf.icb = conf.ngroups;
                conf.lws_d[0] = 1;
                conf.lws_d[1] = 16;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.ih * utils::div_up(conf.iw, conf.iw_block)
                        * conf.id;
                conf.gws_d[1] = conf.ic * conf.ngroups;
                conf.gws_d[2] = conf.mb;
            } else {
                conf.icb = 64;
                while (conf.icb > 16) {
                    if (conf.ic % conf.icb == 0) break;
                    conf.icb /= 2;
                }
                conf.lws_d[0] = 16;
                conf.lws_d[1] = 1;
                conf.lws_d[2] = 1;
                conf.gws_d[0] = conf.icb;
                conf.gws_d[1] = conf.ih * utils::div_up(conf.iw, conf.iw_block)
                        * conf.id;
                conf.gws_d[2] = conf.mb * (conf.ic / conf.icb) * conf.ngroups;
            }
            break;
        default: status = status::unimplemented;
    }

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                                        : utils::pick(conf.ndims - 3, OIw16o16i,
                                                OIhw16o16i, OIdhw16o16i));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gOIw16o16i, gOIhw16o16i, gOIdhw16o16i)
                                        : utils::pick(conf.ndims - 3, OIw16o16i,
                                                OIhw16o16i, OIdhw16o16i));
            break;
        default: status = status::unimplemented;
    }
    if (status != status::success) return status;

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    conf.is_nchw = utils::one_of(src_tag, ncw, nchw, ncdhw);
    conf.is_nhwc = utils::one_of(src_tag, nwc, nhwc, ndhwc);

    return status::success;
}

status_t gen9_convolution_bwd_data_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
    kernel_ctx.define_int("BWD_DATA", 1);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ICB", conf.icb);
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
    kernel_ctx.define_int("OC_PADDED", conf.oc);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("IH_BLOCK", conf.ih_block);
    kernel_ctx.define_int("IW_BLOCK", conf.iw_block);
    kernel_ctx.define_int("IWB", utils::div_up(conf.iw, conf.iw_block));
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.set_data_type(conf.src_data_type);

    switch (conf.ver) {
        case ver_16mb16c: kernel_ctx.define_int("VER_16MB16C", 1); break;
        case ver_8ow16c: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    return status::success;
}

static void bwd_w_compute_block_sizes(
        conv_conf_t &conf, const convolution_pd_t *pd) {
    if (conf.is_depthwise) {
        conf.odb = 1;
        conf.ohb = 1;
        conf.owb = utils::rnd_up(conf.ow, conf.ow_block);
        conf.ocb = 1;
        conf.icb = 1;
        conf.osp_chunk = utils::div_up(conf.od, conf.odb)
                * utils::div_up(conf.oh, conf.ohb)
                * utils::div_up(conf.ow, conf.owb);

        conf.mb_chunk = conf.mb / conf.mb_block;
        conf.nchunk = conf.osp_chunk * conf.mb_chunk;
        return;
    }
    auto *dev_info = utils::downcast<compute::compute_engine_t *>(pd->engine())
                             ->device_info();
    int hw_threads = dev_info->hw_threads();
    size_t llc_bytes = dev_info->llc_cache_size();

    auto next_candidate = [](int size, int block) {
        if (size == block) return block;
        // If size is big enough, then do not care about the remainder.
        if (block * 16 < size) return block + 1;
        // Otherwise search for the next divisor.
        block++;
        while (size % block != 0)
            block++;
        return block;
    };

    int mb_nb = 1;
    conf.odb = 1;
    conf.ohb = 1;
    conf.owb = 1;

    int ic_nb_max = (conf.ver == ver_1stconv)
            ? 1
            : nstl::min(conf.ic / conf.ic_block, 16);
    int oc_nb_max = nstl::min(conf.oc / conf.oc_block, 16);
    int ic_nb = (conf.ver == ver_1stconv)
            ? 1
            : utils::max_div(conf.ic / conf.ic_block, ic_nb_max);
    int oc_nb = utils::max_div(conf.oc / conf.oc_block, oc_nb_max);

    auto get_nthr = [&]() {
        int nthr = (conf.mb / conf.mb_block / mb_nb)
                * utils::div_up(conf.od, conf.odb)
                * utils::div_up(conf.oh, conf.ohb)
                * utils::div_up(conf.ow, conf.owb) * conf.kh * conf.kw * conf.kd
                * (conf.oc / conf.oc_block)
                * (conf.ver == ver_1stconv ? 1 : conf.ic / conf.ic_block)
                * conf.ngroups;
        return nthr;
    };

    auto get_src_dst_size = [&]() {
        int iwb = conf.ndims < 3 ? 1 : conf.owb + 2 * (conf.kw - 1);
        int ihb = conf.ndims < 4 ? 1 : conf.ohb + 2 * (conf.kh - 1);
        int idb = conf.ndims < 5 ? 1 : conf.odb + 2 * (conf.kd - 1);

        size_t ispb = iwb * ihb * idb;
        size_t ospb = conf.owb * conf.ohb * conf.odb;
        size_t src_size = sizeof(float) * conf.mb_block
                * (conf.ver == ver_1stconv ? conf.ic : ic_nb * conf.ic_block)
                * ispb;
        size_t dst_size = sizeof(float) * conf.mb_block
                * (oc_nb * conf.oc_block) * ospb;

        int nthr_per_spb
                = conf.kh * conf.kw * conf.kd * ic_nb * oc_nb * conf.ngroups;
        size_t sz = (size_t)(src_size + dst_size);
        if (nthr_per_spb < hw_threads) sz = sz * hw_threads / nthr_per_spb;
        return sz;
    };

    auto try_next = [&](int &v, int next) {
        if (next <= v) return false;
        int v_old = v;
        v = next;
        // Heuristics:
        // - src and dst size accessed in the inner loops should fit LLC
        // - Require at least (3 * hw_threads) to spawn to have enough
        //   parallelism
        if (get_src_dst_size() > llc_bytes || get_nthr() < 3 * hw_threads) {
            v = v_old;
            return false;
        }
        return true;
    };

    if (conf.ver == ver_8ow16c || conf.ver == ver_1stconv)
        conf.owb = conf.ow_block;

    // Increase spatial tile size as much as possible.
    for (int i = 0; i < 128; i++) {
        int owb_next;
        if (conf.ver == ver_8ow16c || conf.ver == ver_1stconv) {
            int ow_padded = utils::rnd_up(conf.ow, conf.ow_block);
            owb_next = conf.ow_block
                    * next_candidate(ow_padded / conf.ow_block,
                            conf.owb / conf.ow_block);
        } else {
            owb_next = next_candidate(conf.ow, conf.owb);
        }
        try_next(conf.owb, owb_next);

        int ohb_next = next_candidate(conf.oh, conf.ohb);
        try_next(conf.ohb, ohb_next);

        int odb_next = next_candidate(conf.od, conf.odb);
        try_next(conf.odb, odb_next);
    }

    conf.icb = (conf.ver == ver_1stconv) ? conf.ic : ic_nb * conf.ic_block;
    conf.ocb = oc_nb * conf.oc_block;

    conf.osp_chunk = utils::div_up(conf.od, conf.odb)
            * utils::div_up(conf.oh, conf.ohb)
            * utils::div_up(conf.ow, conf.owb);

    conf.mb_chunk = utils::div_up(conf.mb / conf.mb_block, mb_nb);

    conf.nchunk = conf.mb_chunk * conf.osp_chunk;
}

status_t gen9_convolution_bwd_weights_t::pd_t::init_conf() {
    using namespace dnnl::impl::format_tag;
    using namespace data_type;

    const convolution_desc_t &cd = *desc();
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper weights_mdw(diff_weights_md());
    const memory_desc_wrapper dst_mdw(diff_dst_md());
    const memory_desc_wrapper bias_mdw(diff_weights_md(1));

    set_default_conf(conf, cd, *src_md(), *diff_weights_md(), *diff_dst_md(),
            *diff_weights_md(1), *attr());

    const bool is_1stconv = conf.ic_without_padding == 3;
    const bool is_depthwise = conf.with_groups && (conf.ic_without_padding == 1)
            && (conf.oc_without_padding == 1);
    conf.is_depthwise = is_depthwise;

    if (is_1stconv || conf.with_groups) {
        conf.ic = conf.ic_without_padding;
        conf.oc = conf.oc_without_padding;
    } else {
        conf.ic = utils::rnd_up(conf.ic_without_padding, 16);
        conf.oc = utils::rnd_up(conf.oc_without_padding, 16);
    }

    conf.ngroups_without_padding = conf.ngroups;
    if (is_depthwise) conf.ngroups = utils::rnd_up(conf.ngroups, 16);
    const bool is_dw_16g = (conf.is_depthwise && conf.ngroups % 16 == 0);

    const bool is_16ic = conf.ic % 16 == 0;
    const bool is_16oc = conf.oc % 16 == 0;
    const bool use_16mb_unroll = !(conf.mb == 1 || conf.mb % 16 != 0)
            && !is_1stconv && ((is_16ic && is_16oc) || is_dw_16g);

    conf.mb_block = 1;
    conf.oc_block = 1;
    conf.ic_block = 1;
    conf.od_block = 1;
    conf.oh_block = 1;
    conf.ow_block = 1;
    conf.osp_chunk = 1;
    conf.mb_chunk = 1;
    if (use_16mb_unroll)
        conf.ver = ver_16mb16c;
    else if (conf.mb % 16 != 0 && ((is_16oc && is_16ic) || is_dw_16g))
        conf.ver = ver_8ow16c;
    else if (is_1stconv && is_16oc)
        conf.ver = ver_1stconv;
    else
        return status::unimplemented;

    status_t status = status::success;

    switch (conf.ver) {
        case ver_1stconv:
        case ver_8ow16c:
            conf.mb_block = 1;
            conf.oc_block = 16;
            conf.ic_block = conf.ver == ver_8ow16c ? 16 : 1;
            conf.ow_block = 8;
            bwd_w_compute_block_sizes(conf, this);
            break;
        case ver_16mb16c:
            conf.mb_block = 16;
            conf.oc_block = 16;
            conf.ic_block = 16;
            conf.ow_block = 1;
            bwd_w_compute_block_sizes(conf, this);
            break;
        default: status = status::unimplemented;
    }

    conf.sub_group_size = 16;
    conf.lws_d[0] = 16;
    conf.lws_d[1] = 1;
    conf.lws_d[2] = 1;

    if (conf.is_depthwise) {
        conf.gws_d[0] = conf.ngroups;
    } else {
        conf.gws_d[0] = conf.ver == ver_1stconv
                ? conf.ocb * conf.ngroups
                : conf.ocb * (conf.icb / 16) * conf.ngroups;
    }
    conf.gws_d[1] = conf.kh * conf.kw * conf.kd;
    conf.gws_d[2] = conf.nchunk * (conf.ic / conf.icb) * (conf.oc / conf.ocb);

    format_tag_t src_tag, dst_tag, wei_tag;

    switch (conf.ver) {
        case ver_1stconv:
            assert(!conf.is_depthwise);
            src_tag = utils::pick(conf.ndims - 3, ncw, nchw, ncdhw);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.with_groups
                    ? utils::pick(conf.ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                    : utils::pick(conf.ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
            break;
        case ver_16mb16c:
            src_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            dst_tag = utils::pick(
                    conf.ndims - 3, NCw16n16c, NChw16n16c, NCdhw16n16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        case ver_8ow16c:
            src_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            dst_tag = utils::pick(conf.ndims - 3, nCw16c, nChw16c, nCdhw16c);
            wei_tag = conf.is_depthwise
                    ? utils::pick(conf.ndims - 3, Goiw16g, Goihw16g, Goidhw16g)
                    : (conf.with_groups ? utils::pick(conf.ndims - 3,
                               gIOw16i16o, gIOhw16i16o, gIOdhw16i16o)
                                        : utils::pick(conf.ndims - 3, IOw16i16o,
                                                IOhw16i16o, IOdhw16i16o));
            break;
        default: status = status::unimplemented;
    }
    if (status != status::success) return status;

    if (src_mdw.format_kind() == format_kind::any) {
        conf.src_tag = src_tag;
    } else {
        conf.src_tag = src_mdw.matches_one_of_tag(src_tag);
    }
    if (conf.src_tag != src_tag) return status::unimplemented;

    if (weights_mdw.format_kind() == format_kind::any) {
        conf.wei_tag = wei_tag;
    } else {
        conf.wei_tag = weights_mdw.matches_one_of_tag(wei_tag);
    }
    if (conf.wei_tag != wei_tag) return status::unimplemented;

    if (dst_mdw.format_kind() == format_kind::any) {
        conf.dst_tag = dst_tag;
    } else {
        conf.dst_tag = dst_mdw.matches_one_of_tag(dst_tag);
    }
    if (conf.dst_tag != dst_tag) return status::unimplemented;

    return status::success;
}

status_t gen9_convolution_bwd_weights_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    kernel_ctx.define_int("IS_DW", conf.is_depthwise);
    kernel_ctx.define_int("BWD_WEIGHTS", 1);
    kernel_ctx.define_int("G", conf.ngroups);
    kernel_ctx.define_int("MB", conf.mb);
    kernel_ctx.define_int("IC", conf.ic);
    kernel_ctx.define_int("ICB", conf.icb);
    kernel_ctx.define_int("ID", conf.id);
    kernel_ctx.define_int("IH", conf.ih);
    kernel_ctx.define_int("IW", conf.iw);
    kernel_ctx.define_int("OC", conf.oc);
    kernel_ctx.define_int("OCB", conf.ocb);
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
    kernel_ctx.define_int("OC_PADDED", conf.oc);
    kernel_ctx.define_int("OC_WO_PADDING", conf.oc_without_padding);
    kernel_ctx.define_int("G_WO_PADDING", conf.ngroups_without_padding);

    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);
    kernel_ctx.define_int("ODB", conf.odb);
    kernel_ctx.define_int("OHB", conf.ohb);
    kernel_ctx.define_int("OWB", conf.owb);

    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);
    kernel_ctx.define_int("MB_BLOCK", conf.mb_block);
    kernel_ctx.define_int("OC_BLOCK", conf.oc_block);
    kernel_ctx.define_int("IC_BLOCK", conf.ic_block);
    kernel_ctx.define_int("NCHUNK", conf.nchunk);
    kernel_ctx.define_int("OSP_CHUNK", conf.osp_chunk);
    kernel_ctx.define_int("MB_CHUNK", conf.mb_chunk);
    kernel_ctx.define_int(
            "MB_CHUNK_SIZE", utils::div_up(conf.mb, conf.mb_chunk));
    kernel_ctx.define_int("OW_BLOCK", conf.ow_block);

    kernel_ctx.define_int("LWS_0", conf.lws_d[0]);
    kernel_ctx.define_int("LWS_1", conf.lws_d[1]);
    kernel_ctx.define_int("LWS_2", conf.lws_d[2]);

    kernel_ctx.add_option("-cl-std=CL2.0");

    switch (conf.ver) {
        case ver_16mb16c: kernel_ctx.define_int("VER_16MB16C", 1); break;
        case ver_1stconv:
        case ver_8ow16c: kernel_ctx.define_int("VER_8OW16C", 1); break;
        default: break;
    }

    return status::success;
}

status_t gen9_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, weights);
    arg_list.set(2, bias);
    arg_list.set(3, dst);
    arg_list.set(4, conf.eltwise.alpha);
    arg_list.set(5, conf.eltwise.beta);
    arg_list.set(6, conf.eltwise.scale);
    arg_list.set(7, conf.sum_scale);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t gen9_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &weights = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS);
    auto &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    auto &bias = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    const auto &conf = pd()->conf;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, diff_src);
    arg_list.set(1, weights);
    arg_list.set(2, diff_dst);
    arg_list.set(3, bias);

    auto nd_range = compute::nd_range_t(conf.gws_d, conf.lws_d);
    status_t status = compute_stream->parallel_for(nd_range, kernel_, arg_list);

    return status;
}

status_t gen9_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &diff_weights = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS);
    auto &diff_bias = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const auto &conf = pd()->conf;

    const float zero = 0;
    memory_desc_wrapper wei_mdw(pd()->diff_weights_md());
    CHECK(compute_stream->fill(
            diff_weights, &zero, sizeof(zero), wei_mdw.size()));
    if (conf.with_bias) {
        memory_desc_wrapper bia_mdw(pd()->diff_weights_md(1));
        CHECK(compute_stream->fill(
                diff_bias, &zero, sizeof(zero), bia_mdw.size()));
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, diff_weights);
    arg_list.set(2, diff_bias);
    arg_list.set(3, diff_dst);

    status_t status = compute_stream->parallel_for(
            compute::nd_range_t(conf.gws_d, conf.lws_d), kernel_, arg_list);
    if (status != status::success) return status;

    return status::success;
}

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
