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

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_uni_eltwise_injector.hpp"
#include "jit_uni_softmax.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {

typedef float data_t;

using namespace Xbyak;

template <cpu_isa_t isa>
struct jit_softmax_base_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const data_t *src, *dst;
        size_t spat_offt_count;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_t)

    // cpu specific part
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    const AddressFrame &vmmword
            = (isa == sse41) ? xword : (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    const softmax_pd_t *pd_;

    void (*ker)(const call_params_t *);
    void operator()(const call_params_t *p) { (*ker)(p); }
    jit_uni_eltwise_injector_f32<isa> *exp_injector_, *log_injector_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_exp_injector_table = rax;
    Reg64 reg_log_injector_table = rbx;
    Reg64 reg_src = r8;
    Reg64 reg_dst = r9;
    Reg64 reg_spat_offt = r10;
    Reg64 reg_spat_offt_count = r11;
    Reg64 reg_reverse_spat_offt = r12;
    Reg64 reg_tmp = r13;

    Opmask injector_mask = Opmask(1);

    Vmm vtmp; // assigned at placed where used
    Vmm tail_vmask = Vmm(0);
    Xmm xneg_flt_max = Xmm(12);
    Vmm vneg_flt_max = Vmm(isa == avx512_common ? 28 : 12);
    Xmm xone = Xmm(13);
    Vmm vone = Vmm(isa == avx512_common ? 29 : 13);
    Vmm vsum = Vmm(isa == avx512_common ? 30 : 14);
    Vmm vmax = Vmm(isa == avx512_common ? 31 : 15);

    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();

    size_t simd_w_ = vlen / sizeof(data_t);
    size_t unroll_regs_ = 4;

    size_t axis_simd_full_;
    size_t axis_simd_tail_;
    size_t n_loops_;
    size_t loop_tail_;
    size_t axis_stride_;

    void compute_predefined_variables() {
        axis_simd_full_ = pd_->axis_size() / simd_w_;
        axis_simd_tail_ = pd_->axis_size() % simd_w_;
        n_loops_ = axis_simd_full_ / unroll_regs_;
        loop_tail_ = axis_simd_full_ - n_loops_ * unroll_regs_;
        axis_stride_ = compute_axis_stride();
    }

    size_t compute_axis_stride() {
        const memory_desc_wrapper data_d(pd_->src_md());
        const auto &bd = data_d.blocking_desc();

        if (bd.inner_nblks) return sizeof(data_t) * bd.strides[pd_->axis()];
        return vlen;
    }

    void load_common_params() {
        mov(reg_tmp, float2int(1.0f));
        uni_vmovq(xone, reg_tmp);
        uni_vbroadcastss(vone, xone);
        mov(reg_tmp, float2int(-FLT_MAX));
        uni_vmovq(xneg_flt_max, reg_tmp);
        uni_vbroadcastss(vneg_flt_max, xneg_flt_max);

#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_spat_offt_count, ptr[reg_param + PARAM_OFF(spat_offt_count)]);
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
#undef PARAM_OFF
    }

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + reg_spat_offt + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_spat_offt + offt];
    }

    enum class op_t : unsigned { max, sum };

    void perform_op(Vmm v, Vmm vtmp, op_t op) {
        if (op == op_t::max)
            uni_vmaxps(v, v, vtmp);
        else if (op == op_t::sum)
            uni_vaddps(v, v, vtmp);
    }

    template <typename body_t>
    void axis_loop(body_t body) {
        Label main_loop, tail_loop, tail_axis;

        // reverse_spat_offt to dispatch between labels
        mov(reg_reverse_spat_offt, reg_spat_offt_count);
        xor_(reg_spat_offt, reg_spat_offt); // spat_offt to get addr of src/dst
        L(main_loop);
        {
            if (n_loops_) {
                cmp(reg_reverse_spat_offt, unroll_regs_ * axis_stride_);
                jl(tail_loop, T_NEAR);

                body(unroll_regs_, false);
                sub(reg_reverse_spat_offt, unroll_regs_ * axis_stride_);
                add(reg_spat_offt, unroll_regs_ * axis_stride_);
                jmp(main_loop);
            }
        }

        L(tail_loop);
        {
            if (loop_tail_) {
                body(loop_tail_, false);
                add(reg_spat_offt, loop_tail_ * axis_stride_);
            }
        }

        L(tail_axis);
        {
            if (axis_simd_tail_) { body(1, true); }
        }
    }

    virtual void prepare_tail_mask() = 0;
    virtual void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) = 0;
    virtual void accumulate_vmax() = 0;
    virtual void accumulate_vsum() = 0;
    virtual void compute_dst() = 0;

    void forward() {
        accumulate_vmax();
        accumulate_vsum();
        compute_dst();
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void get_code() {
        exp_injector_ = new jit_uni_eltwise_injector_f32<isa>(this,
                alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, true,
                reg_exp_injector_table, injector_mask);
        if (is_logsoftmax_) {
            log_injector_ = new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, true,
                    reg_log_injector_table, injector_mask);
        }

        compute_predefined_variables();
        preamble();
        exp_injector_->load_table_addr();
        if (is_logsoftmax_) log_injector_->load_table_addr();
        if (axis_simd_tail_) prepare_tail_mask();
        load_common_params();
        forward();
        postamble();
        exp_injector_->prepare_table();
        if (is_logsoftmax_) log_injector_->prepare_table();

        ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t *>(getCode()));
    }

    jit_softmax_base_t(const softmax_pd_t *pd) : pd_(pd) {}
};

template <cpu_isa_t isa>
struct jit_softmax_t;

template <>
struct jit_softmax_t<avx512_common> : public jit_softmax_base_t<avx512_common> {
    Opmask tail_opmask = Opmask(2);

    void prepare_tail_mask() override {
        const int mask_f32 = (1 << axis_simd_tail_) - 1;
        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask_f32);
        kmovw(tail_opmask, regw_tmp);
    }

    void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) override {
        vshuff32x4(vtmp, v, v, 0x4E); // 256-bit shuffle
        perform_op(v, vtmp, op);
        vshuff32x4(vtmp, v, v, 0xB1); // 128/256-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0x4E); // 64/128-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0xB1); // 32/64-bit shuffle
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                if (!tail)
                    uni_vmaxps(vmax, vmax, src_ptr(axis_stride_ * i));
                else {
                    uni_vmaxps(vmax | tail_opmask, vmax,
                            src_ptr(axis_stride_ * i));
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg_tmp_src, src_ptr(axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps(vsum, vsum, vreg_tmp_src);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                } else {
                    uni_vmovups_tail(vreg_tmp_src, tail_opmask,
                            src_ptr(axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups_tail(dst_ptr(axis_stride_ * i), tail_opmask,
                                vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps(vsum | tail_opmask, vsum, vreg_tmp_src);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups_tail(dst_ptr(axis_stride_ * i), tail_opmask,
                                vreg_tmp_src);
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    if (is_softmax_)
                        uni_vmulps(
                                vreg_tmp_src, vsum, dst_ptr(axis_stride_ * i));
                    if (is_logsoftmax_) {
                        uni_vmovups(vreg_tmp_src, dst_ptr(axis_stride_ * i));
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                    }
                    uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                } else {
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src | tail_opmask, vsum,
                                dst_ptr(axis_stride_ * i));
                    if (is_logsoftmax_) {
                        uni_vmovups_tail(vreg_tmp_src, tail_opmask,
                                dst_ptr(axis_stride_ * i));
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                    }
                    uni_vmovups_tail(dst_ptr(axis_stride_ * i), tail_opmask,
                            vreg_tmp_src);
                }
            }
        });
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {
        get_code();
    }

    virtual ~jit_softmax_t() {
        delete exp_injector_;
        if (is_logsoftmax_) delete log_injector_;
    }
};

template <>
struct jit_softmax_t<avx2> : public jit_softmax_base_t<avx2> {
    Vmm tail_vmask = Vmm(0);

    void prepare_tail_mask() override {
        static const uint32_t mask_f32[14]
                = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0};
        mov(reg_tmp, reinterpret_cast<size_t>(&mask_f32[7 - axis_simd_tail_]));
        vmovups(tail_vmask, ptr[reg_tmp]);
    }

    void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) override {
        vperm2f128(vtmp, v, v, 0x1); // 128/256-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0x4E); // 64/128-bit shuffle
        perform_op(v, vtmp, op);
        vshufps(vtmp, v, v, 0xB1); // 32/64-bit shuffle
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                if (!tail)
                    uni_vmaxps(vmax, vmax, src_ptr(axis_stride_ * i));
                else {
                    vtmp = Vmm(i + 1);
                    uni_vmovups_tail(
                            vtmp, tail_vmask, src_ptr(axis_stride_ * i));
                    uni_vblendvps(vtmp, vneg_flt_max, vtmp, tail_vmask);
                    uni_vmaxps(vmax, vmax, vtmp);
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg_tmp_src, src_ptr(axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps(vsum, vsum, vreg_tmp_src);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                } else {
                    uni_vmovups_tail(vreg_tmp_src, tail_vmask,
                            src_ptr(axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups_tail(dst_ptr(axis_stride_ * i), tail_vmask,
                                vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    vtmp = Vmm(vreg_tmp_src.getIdx() + 1);
                    uni_vpxor(vtmp, vtmp, vtmp);
                    uni_vblendvps(vtmp, vtmp, vreg_tmp_src, tail_vmask);
                    uni_vaddps(vsum, vsum, vtmp);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups_tail(dst_ptr(axis_stride_ * i), tail_vmask,
                                vreg_tmp_src);
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    if (is_softmax_)
                        uni_vmulps(
                                vreg_tmp_src, vsum, dst_ptr(axis_stride_ * i));
                    if (is_logsoftmax_) {
                        uni_vmovups(vreg_tmp_src, dst_ptr(axis_stride_ * i));
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                    }
                    uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                } else {
                    uni_vmovups_tail(vreg_tmp_src, tail_vmask,
                            dst_ptr(axis_stride_ * i));
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                    if (is_logsoftmax_)
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                    uni_vmovups_tail(dst_ptr(axis_stride_ * i), tail_vmask,
                            vreg_tmp_src);
                }
            }
        });
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {
        get_code();
    }

    virtual ~jit_softmax_t() {
        delete exp_injector_;
        if (is_logsoftmax_) delete log_injector_;
    }
};

template <>
struct jit_softmax_t<sse41> : public jit_softmax_base_t<sse41> {
    Vmm tail_vmask = Vmm(0);

    void prepare_tail_mask() override {
        static const uint32_t mask_f32[4] = {0xffffffff, 0, 0, 0};
        mov(reg_tmp, reinterpret_cast<size_t>(mask_f32));
        movups(tail_vmask, ptr[reg_tmp]);
    }

    void get_horizontal_op(const Vmm &v, const Vmm &vtmp, op_t op) override {
        uni_vmovups(vtmp, v);
        shufps(vtmp, vtmp, 0x4E); // 64/128-bit shuffle
        perform_op(v, vtmp, op);
        uni_vmovups(vtmp, v);
        shufps(vtmp, vtmp, 0xB1); // 32/64-bit shuffle
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    // SIGSEGV on unaligned addr if do maxps directly on memory
                    uni_vmovups(vreg_tmp_src, src_ptr(axis_stride_ * i));
                    uni_vmaxps(vmax, vmax, vreg_tmp_src);
                } else {
                    vtmp = Vmm(vreg_tmp_src.getIdx()
                            + 1); // next after vreg_tmp_src

                    for (size_t j = 0; j < axis_simd_tail_; j++) {
                        uni_vmovups(vreg_tmp_src, vneg_flt_max);
                        uni_vmovss(vtmp,
                                src_ptr(axis_stride_ * i + sizeof(data_t) * j));
                        uni_vblendvps(
                                vreg_tmp_src, vreg_tmp_src, vtmp, tail_vmask);
                        uni_vmaxps(vmax, vmax, vreg_tmp_src);
                    }
                }
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg_tmp_src, src_ptr(axis_stride_ * i));
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) // store before applying exp
                        uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps(vsum, vsum, vreg_tmp_src);
                    if (is_softmax_) // store after applying exp
                        uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                } else {
                    vtmp = Vmm(vreg_tmp_src.getIdx() + 1);
                    for (size_t j = 0; j < axis_simd_tail_; j++) {
                        uni_vmovss(vreg_tmp_src,
                                src_ptr(axis_stride_ * i + sizeof(data_t) * j));
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                        if (is_logsoftmax_) // store before applying exp
                            uni_vmovss(dst_ptr(axis_stride_ * i
                                               + sizeof(data_t) * j),
                                    vreg_tmp_src);
                        exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                        uni_vpxor(vtmp, vtmp, vtmp);
                        uni_vblendvps(vtmp, vtmp, vreg_tmp_src, tail_vmask);
                        uni_vaddps(vsum, vsum, vtmp);
                        if (is_softmax_) // store after applying exp
                            uni_vmovss(dst_ptr(axis_stride_ * i
                                               + sizeof(data_t) * j),
                                    vreg_tmp_src);
                    }
                }
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (!tail) {
                    uni_vmovups(vreg_tmp_src, dst_ptr(axis_stride_ * i));
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                    if (is_logsoftmax_)
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                    uni_vmovups(dst_ptr(axis_stride_ * i), vreg_tmp_src);
                } else {
                    for (size_t j = 0; j < axis_simd_tail_; j++) {
                        uni_vmovss(vreg_tmp_src,
                                dst_ptr(axis_stride_ * i + sizeof(data_t) * j));
                        if (is_softmax_)
                            uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                        if (is_logsoftmax_)
                            uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);
                        uni_vmovss(
                                dst_ptr(axis_stride_ * i + sizeof(data_t) * j),
                                vreg_tmp_src);
                    }
                }
            }
        });
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {
        get_code();
    }

    virtual ~jit_softmax_t() {
        delete exp_injector_;
        if (is_logsoftmax_) delete log_injector_;
    }
};

} // namespace

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *apd)
    : primitive_impl_t(apd) {
    softmax_driver_ = new softmax_impl::driver_t<isa>(pd());
}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto &bd = data_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto outer_stride = data_d.padded_dims()[axis] * inner_size;
    const auto outer_size = data_d.nelems(true) / outer_stride;

    parallel_nd(outer_size, inner_size, [&](dim_t ou, dim_t in) {
        dim_t offset = ou * outer_stride + in * inner_stride;
        const data_t *src_ptr = src + offset;
        data_t *dst_ptr = dst + offset;
        softmax_driver_->exec(src_ptr, dst_ptr, outer_stride);
    });

    return status::success;
}

namespace softmax_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {

    driver_t(const softmax_pd_t *pd) : pd_(pd), ker_(pd_) {}
    ~driver_t() {}

    void exec(const data_t *src, data_t *dst, const dim_t outer_stride) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.spat_offt_count = outer_stride * sizeof(data_t);
        p.src = src;
        p.dst = dst;
        ker_(&p);
    }

private:
    const softmax_pd_t *pd_;

    jit_softmax_t<isa> ker_;
};

} // namespace softmax_impl

/* struct instantiation */
template struct jit_uni_softmax_fwd_t<sse41>;
template struct jit_uni_softmax_fwd_t<avx2>;
template struct jit_uni_softmax_fwd_t<avx512_common>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
