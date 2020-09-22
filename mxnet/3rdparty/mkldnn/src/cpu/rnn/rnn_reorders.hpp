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

#ifndef CPU_RNN_REORDERS_HPP
#define CPU_RNN_REORDERS_HPP

#include <assert.h>

#include "bfloat16.hpp"
#include "cpu_reorder_pd.hpp"
#include "dnnl_thread.hpp"
#include "gemm/gemm_pack.hpp"
#include "simple_q10n.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <data_type_t type_i, data_type_t type_o>
struct rnn_data_reorder_t : public primitive_impl_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_data_reorder", rnn_data_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            using namespace status;
            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = id.data_type() == type_i && od.data_type() == type_o
                    && utils::one_of(id.ndims(), 3, 4)
                    && attr->has_default_values(
                            primitive_attr_t::skip_mask_t::rnn_data_qparams
                            | primitive_attr_t::skip_mask_t::
                                    rnn_weights_qparams)
                    && IMPLICATION(id.ndims() == 3,
                            id.matches_tag(format_tag::tnc)
                                    && od.matches_tag(format_tag::tnc))
                    && IMPLICATION(id.ndims() == 4,
                            id.matches_tag(format_tag::ldnc)
                                    && od.matches_tag(format_tag::ldnc));
            if (!args_ok) return invalid_arguments;

            auto _pd = new pd_t(
                    engine, attr, src_engine, src_md, dst_engine, dst_md);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) {
                delete _pd;
                return unimplemented;
            }
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

    rnn_data_reorder_t(const pd_t *apd) : primitive_impl_t(apd) {}

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto input = CTX_IN_MEM(const in_data_t *, DNNL_ARG_FROM);
        auto output = CTX_OUT_MEM(out_data_t *, DNNL_ARG_TO);
        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        const size_t nelems = input_d.nelems();
        const float scale = pd()->attr()->rnn_data_qparams_.scale_;
        const float shift = pd()->attr()->rnn_data_qparams_.shift_;

        parallel_nd(nelems, [&](size_t i) {
            float in = (float)input[input_d.off_l(i)] * scale + shift;
            output[output_d.off_l(i)] = qz_a1b0<float, out_data_t>()(in);
        });

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <data_type_t type_i>
struct rnn_weights_reorder_s8_t : public primitive_impl_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_weights_reorder_s8", rnn_weights_reorder_s8_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            using namespace status;
            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true;
#define PD_CHECK_ARG(x) args_ok = args_ok && (x)
            // Fast checks
            PD_CHECK_ARG(id.data_type() == type_i);
            PD_CHECK_ARG(od.data_type() == data_type::s8);
            PD_CHECK_ARG(od.format_kind() == format_kind::rnn_packed);
            PD_CHECK_ARG(od.rnn_packed_desc().format == dnnl_ldigo_p);
            PD_CHECK_ARG(od.rnn_packed_desc().n_parts == 1);
            PD_CHECK_ARG(attr->has_default_values(
                    primitive_attr_t::skip_mask_t::rnn_data_qparams
                    | primitive_attr_t::skip_mask_t::rnn_weights_qparams));
            if (!args_ok) return invalid_arguments;

            // Slower checks
            PD_CHECK_ARG(id.is_dense());
            if (!args_ok) return invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(
                    format_tag::ldigo, format_tag::ldgoi);
            if (itag == format_tag::undef) return invalid_arguments;

            const int mask = attr->rnn_weights_qparams_.mask_;
            if (!utils::one_of(mask, 0, 24)) return unimplemented;

            auto _pd = new pd_t(
                    engine, attr, src_engine, src_md, dst_engine, dst_md);
            if (_pd == nullptr) return out_of_memory;
            _pd->itag_ = itag;
            if (_pd->init() != success) {
                delete _pd;
                return unimplemented;
            }
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
#undef PD_CHECK_ARG
        }

        status_t init() {
            status_t status = cpu_reorder_pd_t::init();
            if (status != status::success) return status;

            init_scratchpad();

            return status::success;
        }

        format_tag_t itag_ = dnnl_format_tag_undef;
        size_t thr_scratch_comp_sz_ = 0;

    private:
        void init_scratchpad() {
            const memory_desc_wrapper id(src_md());
            const size_t nelems = id.nelems();
            const auto &dims = id.dims();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            size_t quantization_size = sizeof(int8_t) * nelems;
            // we do not use GO directly, as this can cause false sharing
            // (2 threads writing to the same cache line)
            thr_scratch_comp_sz_ = utils::rnd_up(dims[3] * dims[4], 16);
            size_t reduction_size;
            reduction_size = itag_ == format_tag::ldigo ? dnnl_get_max_threads()
                            * sizeof(int32_t) * thr_scratch_comp_sz_
                                                        : 0;

            scratchpad.book(
                    key_reorder_rnn_weights_quantization, quantization_size);
            scratchpad.book(key_reorder_rnn_weights_reduction, reduction_size);
        }
    };

    rnn_weights_reorder_s8_t(const pd_t *apd) : primitive_impl_t(apd) {}

private:
    typedef typename prec_traits<type_i>::type in_data_t;

    void quantize_goi(int8_t *scratch_quantized,
            const memory_desc_wrapper &src_d, const float *src) const {
        const auto &dims = src_d.dims();
        // TODO: trivial strides assumes here.
        //       Use proper strides where appropriate
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        const float *scales = pd()->attr()->rnn_weights_qparams_.scales_;
        const int mask = pd()->attr()->rnn_weights_qparams_.mask_;

        parallel_nd(L * D, G * O, [&](int ld, int go) {
            const float s = scales[(mask == 0) ? 0 : go];
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < I; i++) {
                scratch_quantized[ld * I * G * O + i * G * O + go]
                        = qz_b0<in_data_t, int8_t>()(
                                src[ld * G * O * I + go * I + i], s);
            }
        });
    }

    void quantize_igo(int8_t *scratch_quantized,
            const memory_desc_wrapper &src_d, const float *src) const {
        const auto &dims = src_d.dims();
        // TODO: trivial strides assumes here.
        //       Use proper strides where appropriate
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        const float *scales = pd()->attr()->rnn_weights_qparams_.scales_;
        const int mask = pd()->attr()->rnn_weights_qparams_.mask_;

        int nthr = dnnl_get_max_threads();
        parallel(nthr, [&](const int ithr, const int nthr) {
            int start, end;
            balance211(L * D * I, nthr, ithr, start, end);
            for (int ldi = start; ldi < end; ldi++) {
                for (int go = 0; go < G * O; go++) {
                    const float s = scales[(mask == 0) ? 0 : go];
                    scratch_quantized[ldi * G * O + go]
                            = qz_b0<in_data_t, int8_t>()(
                                    src[ldi * G * O + go], s);
                }
            }
        });
    }

    void compensate_goi(float *compensation, const memory_desc_wrapper &src_d,
            int8_t *scratch_quantized) const {
        const auto &dims = src_d.dims();
        // TODO: trivial strides assumes here.
        //       Use proper strides where appropriate
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        parallel_nd(L * D, G * O, [&](int ld, int go) {
            int32_t compensation_s32 = 0;
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < I; i++) {
                compensation_s32
                        += scratch_quantized[ld * I * G * O + i * G * O + go];
            }
            compensation[ld * G * O + go]
                    = math::saturate<float>(compensation_s32);
        });
    }

    void compensate_igo(float *compensation, const memory_desc_wrapper &src_d,
            int8_t *scratch_quantized, int32_t *scratch_compensation) const {
        const auto &dims = src_d.dims();
        // TODO: trivial strides assumed here.
        //       Use proper strides where appropriate
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        // We parallelize on LD and GO
        // TODO: maybe restrict parallelism as we might have large
        // parallelisation overhead if dimensions are small
        int nthr = dnnl_get_max_threads();
        int LD_nthr = nstl::min(L * D, nthr);
        int GO_nthr = nstl::min(G * O, nthr / LD_nthr);
        parallel(nthr, [&](const int ithr, const int nthr) {
            int LD_ithr = -1, LD_s = -1, LD_e = -1;
            int GO_ithr = -1, GO_s = -1, GO_e = -1;
            if (ithr < LD_nthr * GO_nthr) {
                LD_ithr = ithr % LD_nthr;
                GO_ithr = ithr / LD_nthr;
                balance211(L * D, LD_nthr, LD_ithr, LD_s, LD_e);
                balance211(G * O, GO_nthr, GO_ithr, GO_s, GO_e);
            }
            int32_t *compensation_s32
                    = scratch_compensation + ithr * pd()->thr_scratch_comp_sz_;
            for (int ld = LD_s; ld < LD_e; ld++) {
                if (I == 1) {
                    PRAGMA_OMP_SIMD()
                    for (int go = GO_s; go < GO_e; go++)
                        compensation[ld * G * O + go] = math::saturate<float>(
                                scratch_quantized[go + I * (ld)]);
                } else {
                    // We split the loop on I in three to avoid conditionals or zeroing compensation
                    int i = 0;
                    PRAGMA_OMP_SIMD()
                    for (int go = GO_s; go < GO_e; go++)
                        compensation_s32[go] = scratch_quantized[go
                                + G * O * (i + I * (ld))];
                    // 1 <= i < I-1
                    for (i = 1; i < I - 1; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int go = GO_s; go < GO_e; go++)
                            compensation_s32[go] += scratch_quantized[go
                                    + G * O * (i + I * (ld))];
                    }
                    // i = I-1
                    PRAGMA_OMP_SIMD()
                    for (int go = GO_s; go < GO_e; go++)
                        compensation[ld * G * O + go]
                                = math::saturate<float>(compensation_s32[go]
                                        + scratch_quantized[go
                                                + G * O * (i + I * (ld))]);
                }
            }
        });
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto src = CTX_IN_MEM(const in_data_t *, DNNL_ARG_FROM);
        auto dst = CTX_OUT_MEM(char *, DNNL_ARG_TO);
        const memory_desc_wrapper &src_d = pd()->src_md();
        const memory_desc_wrapper &dst_d = pd()->dst_md();
        if (src_d.has_zero_dim()) {
            assert(dst_d.has_zero_dim());
            return status::success;
        }

        const auto &dims = src_d.dims();
        // TODO: trivial strides assumes here.
        //       Use proper strides where appropriate
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        /* Quantize src & compute compensation */
        auto scratch_quantized
                = (int8_t * __restrict) ctx.get_scratchpad_grantor()
                          .template get<void>(memory_tracking::names::
                                          key_reorder_rnn_weights_quantization);
        auto scratch_compensation
                = (int32_t * __restrict) ctx.get_scratchpad_grantor()
                          .template get<void>(memory_tracking::names::
                                          key_reorder_rnn_weights_reduction);
        float *comp = reinterpret_cast<float *>(
                dst + dst_d.rnn_packed_desc().offset_compensation);

        /* Step 1: we quantize if we need to */
        if (type_i == data_type::f32) {
            switch (pd()->itag_) {
                case format_tag::ldigo:
                    quantize_igo(scratch_quantized, src_d, (float *)src);
                    break;
                case format_tag::ldgoi:
                    quantize_goi(scratch_quantized, src_d, (float *)src);
                    break;
                default: assert(!"Unsupported reorder");
            }
        } else
            scratch_quantized = (int8_t * __restrict) src;

        /* Step 2: we pre-compute the compensation */
        switch (pd()->itag_) {
            case format_tag::ldigo:
                compensate_igo(
                        comp, src_d, scratch_quantized, scratch_compensation);
                break;
            case format_tag::ldgoi:
                compensate_goi(comp, src_d, scratch_quantized);
                break;
            default: assert(!"Unsupported reorder");
        }

        /* Step 3: we pack the matrix */
        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return o + O * (g + G * (i + I * (d + D * l)));
        };
        int n_parts = dst_d.rnn_packed_desc().n_parts;
        const size_t *size_packed_cell = dst_d.rnn_packed_desc().part_pack_size;
        const int *parts = dst_d.rnn_packed_desc().parts;
        const int n = dst_d.rnn_packed_desc().n;
        const int ldb = dst_d.rnn_packed_desc().ldb;
        char *to_pack = dst;

        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = parts[p] * O;
                    int k_p = I;
                    int lda = G * O;
                    gemm_s8u8s32_pack("A", "N", "N", &m_p, &n, &k_p, &lda, &ldb,
                            &scratch_quantized[off_igo(l, d, 0, g, 0)],
                            to_pack);
                    to_pack += size_packed_cell[p];
                }
            }
        }
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

template <data_type_t type_i, data_type_t type_o>
struct rnn_weights_reorder_t : public primitive_impl_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_weights_reorder", rnn_weights_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md) {
            using namespace status;

            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true
                    && IMPLICATION(type_o == data_type::bf16
                                    || type_i == data_type::bf16,
                            mayiuse(avx512_core))
                    && id.data_type() == type_i && od.data_type() == type_o
                    && od.format_kind() == format_kind::rnn_packed
                    && utils::one_of(od.rnn_packed_desc().format, dnnl_ldigo_p,
                            dnnl_ldgoi_p)
                    && attr->has_default_values();
            if (!args_ok) return invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(
                    format_tag::ldigo, format_tag::ldgoi);
            if (itag == format_tag::undef) return invalid_arguments;

            auto _pd = new pd_t(
                    engine, attr, src_engine, src_md, dst_engine, dst_md);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) {
                delete _pd;
                return unimplemented;
            }
            _pd->itag_ = itag;
            _pd->init_scratchpad_md();
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        format_tag_t itag_;

        status_t init() {
            status_t status = cpu_reorder_pd_t::init();
            if (status != status::success) return status;

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            const memory_desc_wrapper id(src_md());
            const memory_desc_wrapper od(dst_md());
            const rnn_packed_desc_t &rnn_pdata = od.rnn_packed_desc();

            format_tag_t itag = id.matches_one_of_tag(
                    format_tag::ldigo, format_tag::ldgoi);
            bool layout_cross_case
                    = (itag == format_tag::ldigo
                              && rnn_pdata.format == rnn_packed_format::ldgoi_p)
                    || (itag == format_tag::ldgoi
                            && rnn_pdata.format == rnn_packed_format::ldigo_p),
                    dt_cross_case
                    = type_i == data_type::f32 && type_o == data_type::bf16;
            size_t sz = id.nelems() * sizeof(out_data_t);

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_reorder_rnn_weights_transposition,
                    layout_cross_case ? sz : 0);
            scratchpad.book(
                    key_reorder_rnn_weights_bf16_cvt, dt_cross_case ? sz : 0);
        }
    };

    rnn_weights_reorder_t(const pd_t *apd) : primitive_impl_t(apd) {}

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto input = CTX_IN_MEM(const in_data_t *, DNNL_ARG_FROM);
        auto output = CTX_OUT_MEM(out_data_t *, DNNL_ARG_TO);
        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        if (input_d.has_zero_dim()) {
            assert(output_d.has_zero_dim());
            return status::success;
        }

        const auto &dims = input_d.dims();
        const rnn_packed_desc_t &rnn_pdata = output_d.rnn_packed_desc();
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        /* Pack */
        const bool from_igo = pd()->itag_ == format_tag::ldigo;
        const bool to_igo = rnn_pdata.format == dnnl_ldigo_p;
        int n_parts = rnn_pdata.n_parts;
        const size_t *size_packed_cell = rnn_pdata.part_pack_size;
        const int *parts = rnn_pdata.parts;
        const int n = rnn_pdata.n;

        /* Convert fp32 input to bf16 */
        out_data_t *input_cvt = (out_data_t *)input;
        if (type_i == data_type::f32 && type_o == data_type::bf16) {
            input_cvt
                    = (out_data_t *)ctx.get_scratchpad_grantor()
                              .template get<void>(memory_tracking::names::
                                              key_reorder_rnn_weights_bf16_cvt);
            parallel_nd(L * D, [&](int ld) {
                cvt_float_to_bfloat16((bfloat16_t *)input_cvt + ld * G * O * I,
                        (float *)input + ld * G * O * I, G * O * I);
            });
        }

        /* Transpose weights prior to packing to ensure that packed GEMM
         * algorithm will be dispatched */
        out_data_t *input_tr = input_cvt;
        if (from_igo != to_igo) {
            input_tr
                    = (out_data_t *)ctx.get_scratchpad_grantor().template get<void>(
                            memory_tracking::names::
                                    key_reorder_rnn_weights_transposition);
            const int M = to_igo ? G * O : I;
            const int N = to_igo ? I : G * O;
            parallel_nd(L * D, N, [&](int ld, int i) {
                for (int j = 0; j < M; j++) {
                    input_tr[ld * M * N + i * M + j]
                            = input_cvt[ld * M * N + j * N + i];
                }
            });
        }

        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return l * D * I * G * O + d * I * G * O + i * G * O + g * O + o;
        };
        auto off_goi = [&](int l, int d, int i, int g, int o) {
            return l * D * G * O * I + d * G * O * I + g * O * I + o * I + i;
        };
        const int lda = to_igo ? G * O : I;
        const int ldb = rnn_pdata.ldb;
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = to_igo ? parts[p] * O : I;
                    int k_p = to_igo ? I : parts[p] * O;
                    dnnl_status_t st;
                    if (type_o == data_type::bf16) {
                        st = gemm_bf16bf16f32_pack("A", "N", "N", &m_p, &n,
                                &k_p, &lda, &ldb,
                                (bfloat16_t *)&input_tr[to_igo
                                                ? off_igo(l, d, 0, g, 0)
                                                : off_goi(l, d, 0, g, 0)],
                                (bfloat16_t *)output);
                    } else {
                        st = sgemm_pack("A", "N", "N", &m_p, &n, &k_p, &lda,
                                &ldb,
                                (float *)&input_tr[to_igo
                                                ? off_igo(l, d, 0, g, 0)
                                                : off_goi(l, d, 0, g, 0)],
                                (float *)output);
                    }
                    assert(st == dnnl_success);
                    MAYBE_UNUSED(st);
                    output += size_packed_cell[p] / sizeof(out_data_t);
                }
            }
        }
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
