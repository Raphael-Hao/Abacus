/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "memory_tracking.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"

#include "jit_avx512_core_x8s8s32x_1x1_conv_kernel.hpp"
#include "jit_avx512_core_x8s8s32x_convolution.hpp"
#include "jit_uni_1x1_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t
    : public primitive_impl_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_()
            , jcp_dw_()
            , dw_conv_pd_(nullptr) {}

        pd_t(const pd_t &other) : cpu_convolution_fwd_pd_t(other) {
            copy(other);
        }

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            cpu_convolution_fwd_pd_t::operator=(other);
            clear();
            copy(other);

            return *this;
        }

        ~pd_t() { clear(); }

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_int8_1x1:",
                                    ((jcp_.ver == ver_vnni) ? avx512_core_vnni
                                                            : avx512_core),
                                    ""),
                jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t);

        status_t init() {
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, data_type::s8,
                            data_type::undef, dst_type, data_type::s32)
                    && IMPLICATION(with_bias(),
                            utils::one_of(desc()->bias_desc.data_type,
                                    data_type::f32, data_type::s32,
                                    data_type::s8, data_type::u8))
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::oscale
                            | primitive_attr_t::skip_mask_t::post_ops)
                    && !has_zero_dim_memory()
                    && set_default_formats_common(
                            dat_tag(), format_tag::any, dat_tag());

            if (!ok) return status::unimplemented;
            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md());

            status_t status
                    = jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_conf(jcp_,
                            *desc(), src_md_, weights_md_, dst_md_, bias_md_,
                            *attr(), dnnl_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            if (jcp_.with_dw_conv) {
                status = depthwise_po_init();
                if (status != status::success) return status;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_, *attr());

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        virtual const memory_desc_t *dst_md(int index = 0) const override {
            return jcp_.with_dw_conv ? dw_conv_pd_->dst_md(index) : &dst_md_;
        }

        virtual const memory_desc_t *arg_md(int index = 0) const override {
            if (jcp_.with_dw_conv) {
                switch (index) {
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS:
                        return dw_conv_pd_->weights_md(0);
                    case DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS:
                        return dw_conv_pd_->weights_md(1);
                    default: break;
                }
            }
            return convolution_fwd_pd_t::arg_md(index);
        }

        virtual arg_usage_t arg_usage(int arg) const override {

            if (utils::one_of(arg, DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
                        DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS))
                return arg_usage_t::input;

            return convolution_fwd_pd_t::arg_usage(arg);
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;
        jit_conv_conf_t jcp_dw_;
        cpu_convolution_fwd_pd_t *dw_conv_pd_ = nullptr;

    protected:
        format_tag_t dat_tag() const {
            return utils::pick(src_md_.ndims - 3, format_tag::nwc,
                    format_tag::nhwc, format_tag::ndhwc);
        }

        void clear() {
            if (dw_conv_pd_) {
                delete dw_conv_pd_;
                dw_conv_pd_ = nullptr;
            }
            return;
        }

        void copy(const pd_t &other) {
            jcp_ = other.jcp_;
            rtus_ = other.rtus_;
            jcp_dw_ = other.jcp_dw_;
            if (other.dw_conv_pd_)
                dw_conv_pd_ = static_cast<decltype(dw_conv_pd_)>(
                        other.dw_conv_pd_->clone());
            return;
        }

        status_t depthwise_po_init() {
            using namespace memory_tracking;
            auto &jcp_1x1 = jcp_;
            auto &jcp_dw = jcp_dw_;
            auto &attr_1x1 = attr_;
            const auto &src_md = dst_md_;
            const memory_desc_wrapper src_d(src_md);
            const auto nthr = dnnl_get_max_threads();
            auto l2_cache = get_cache_size(2, true) * nthr;

            // Note: A robust fusion implementation would be to check if both
            // 1x1 conv and dw conv that are considered here for fusion are
            // optimal independently. This would require creating a new
            // primitive_desc through primitive_iterator & check if they match.
            // Due to concern that these creations and/or checks could be heavy,
            // for 1x1: Check that no better ISA is available.
            // for dw: Always fuse with same ISA.
            // Caveat: May be a better dw conv exists.

            // TODO: Add a check if better ISA exists following above note.
            bool ok = true && is_fwd()
                    && (attr_1x1.post_ops_.find(primitive_kind::sum) == -1)
                    // TODO: Below may be further tuned.
                    && (l2_cache < src_d.size())
                    // load_grp_count check can be redundant due to l2 check
                    // above. Adding it explicitly as the current driver doesn't
                    // work if this condition fails.
                    && (jcp_1x1.load_grp_count < 2);
            if (!ok) return status::unimplemented;

            int dw_po_index
                    = attr_1x1.post_ops_.find(primitive_kind::convolution);

            convolution_desc_t cd_dw;
            primitive_attr_t attr_dw;
            CHECK(get_depthwise_conv_desc(
                    cd_dw, src_md, attr_1x1, attr_dw, dw_po_index));

            auto dw_dst_dt = cd_dw.dst_desc.data_type;

            // Check if the fusable_pd matches with the primitive_desc returned
            // by dnnl_primitive_desc_create(). If it doesn't match, then there
            // probably exists a more optimized version of depthwise convolution
            // than the fusable_pd. In this case, fallback to reference fusion.
#define CASE(sdt, ddt) \
    case ddt: { \
        using dw_pd_t \
                = jit_avx512_core_x8s8s32x_convolution_fwd_t<sdt, ddt>::pd_t; \
        auto fusable_pd = dw_pd_t(engine(), &cd_dw, &attr_dw, nullptr); \
        CHECK(fusable_pd.init()); \
        dw_conv_pd_ = fusable_pd.clone(); \
        jcp_dw = static_cast<dw_pd_t *>(dw_conv_pd_)->jcp_; \
        break; \
    }
            if (jcp_1x1.dst_dt == data_type::u8) {
                switch (dw_dst_dt) {
                    CASE(data_type::u8, data_type::u8);
                    CASE(data_type::u8, data_type::s8);
                    CASE(data_type::u8, data_type::s32);
                    CASE(data_type::u8, data_type::f32);
                    default: return status::unimplemented;
                }
            } else if (jcp_1x1.dst_dt == data_type::s8) {
                switch (dw_dst_dt) {
                    CASE(data_type::s8, data_type::u8);
                    CASE(data_type::s8, data_type::s8);
                    CASE(data_type::s8, data_type::s32);
                    CASE(data_type::s8, data_type::f32);
                    default: return status::unimplemented;
                }
            } else
                return status::unimplemented;
#undef CASE

            ok = true
                    && (dnnl_memory_desc_equal(&src_md, dw_conv_pd_->src_md(0)))
                    && (jcp_1x1.oc_without_padding % jcp_1x1.oc_block == 0)
                    && IMPLICATION(
                            jcp_dw.ow_block, jcp_dw.ow_block == jcp_dw.ow);
            if (!ok) return status::unimplemented;

            assert(dw_conv_pd_->dst_md(0)->format_kind != format_kind::any);
            assert(dw_conv_pd_->weights_md(0)->format_kind != format_kind::any);
            assert(IMPLICATION(
                    dw_conv_pd_->weights_md(1)->data_type != data_type::undef,
                    dw_conv_pd_->weights_md(1)->format_kind
                            != format_kind::any));

            jcp_dw.is_fused_conv = true;
            // TODO: Support/experiment arbitary oc_work in dw conv.
            // Until then we keep ch_work perfectly divisible.
            while (jcp_1x1.nb_load % jcp_1x1.nb_load_blocking != 0)
                --jcp_1x1.nb_load_blocking;
            jcp_1x1.nb_load_blocking_max = jcp_1x1.nb_load_blocking;

            while (jcp_1x1.nb_load_blocking % jcp_dw.nb_ch_blocking != 0)
                --jcp_dw.nb_ch_blocking;

            jcp_dw.dw_conv_buffer_oc
                    = jcp_1x1.nb_load_blocking * jcp_1x1.oc_block;
            jcp_1x1.bcast_loop_output_step = jcp_1x1.ur
                    * (jcp_1x1.nb_load_blocking * jcp_1x1.oc_block)
                    * jcp_1x1.typesize_out;

            registrar_t scratchpad(scratchpad_registry_);
            registrar_t dw_scratchpad(scratchpad, names::prefix_fusion);

            size_t dw_conv_buffer_size_ = (size_t)nthr * jcp_dw.kh * jcp_dw.iw
                    * jcp_dw.dw_conv_buffer_oc
                    * types::data_type_size(dw_conv_pd_->src_md()->data_type);
            assert(dw_conv_buffer_size_);
            dw_scratchpad.book(memory_tracking::names::key_fusion_inout_buffer,
                    dw_conv_buffer_size_);

            dw_conv_kernel_t::init_scratchpad(
                    dw_scratchpad, jcp_dw, *(dw_conv_pd_->attr()));
            return status::success;
        }
    };
    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_impl_t(apd), kernel_(nullptr), rtus_driver_(nullptr) {
        kernel_ = new jit_avx512_core_x8s8s32x_1x1_conv_kernel(
                pd()->jcp_, *pd()->attr());

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new dw_conv_kernel_t(
                    pd()->jcp_dw_, *(pd()->dw_conv_pd_->attr()));
        }

        init_rtus_driver<avx512_common>(this);
    }

    ~jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t() {
        delete kernel_;
        if (kernel_dw_) { delete kernel_dw_; }
        delete rtus_driver_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    // Note: In case of fused depthwise convolution, the final output datatype
    // after fusion may not be dst_data_t.
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights, const char *bias,
            const wei_data_t *weights_dw, const char *bias_dw, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    jit_avx512_core_x8s8s32x_1x1_conv_kernel *kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
    using dw_conv_kernel_t = jit_avx512_core_x8s8s32x_fwd_kernel;
    dw_conv_kernel_t *kernel_dw_ = nullptr;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
