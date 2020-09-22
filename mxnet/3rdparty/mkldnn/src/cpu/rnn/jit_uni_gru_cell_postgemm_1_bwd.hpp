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

#ifndef CPU_JIT_UNI_GRU_CELL_POSTGEMM_1_BWD_HPP
#define CPU_JIT_UNI_GRU_CELL_POSTGEMM_1_BWD_HPP

#include "jit_uni_rnn_common_postgemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

template <cpu_isa_t isa, impl::data_type_t src_data_t,
        impl::data_type_t scratch_data_t>
struct jit_uni_gru_cell_postgemm_part1_bwd : public jit_uni_rnn_postgemm {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_gru_cell_postgemm_part1_bwd)

    jit_uni_gru_cell_postgemm_part1_bwd(
            const rnn_utils::rnn_conf_t &rnn, const rnn_pd_t *pd)
        : jit_uni_rnn_postgemm(rnn, pd) {}

    ~jit_uni_gru_cell_postgemm_part1_bwd() {}

    void init(data_type_t sdt) override {
        jit_uni_rnn_postgemm::init(src_data_t);
        generate();
        kernel_ = (kernel_t)this->getCode();
    }

protected:
    // register size in bytes
    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    size_t vlen = cpu_isa_traits<isa>::vlen;
    size_t vlen_scratch
            = vlen / (sizeof(float) / types::data_type_size(scratch_data_t));
    size_t hstate_dt_size = sizeof(float);
    size_t gate_dt_size = types::data_type_size(scratch_data_t);
    size_t scratch_dt_size = types::data_type_size(scratch_data_t);

    void generate() {
        using namespace Xbyak;

        // Labels declaration
        Label vector_loop_start_label, vector_loop_inc_regs,
                vector_loop_end_label;
        Label rem_loop_start_label, rem_loop_inc_regs, rem_loop_end_label;
        Label table_label;

        // Register map
        Reg64 table_reg(rbx); // used to load ones before the loop
        Reg64 loop_cnt(rbx); // loop counter, can be aliased with table_reg

        // We skip vmm0 as it can be used by the injector for masks on sse4.1
        int dG0_idx = 1, dG2_idx = 3, G0_idx = 4, G2_idx = 6, h_idx = 7,
            dHt_idx = 8, one_idx = 9, tmp1_idx = 10, tmp2_idx = 11;
        Vmm one_vmm(one_idx);
        Xmm one_xmm(one_idx);

        // constant table map
        Address one_addr = ptr[table_reg];

        // We start code generations here
        preamble();

        // extract addresses passed as parameter
        auto addr_ws_gates_reg = abi_param1;
        auto addr_scratch_gates_reg = abi_param2;
        auto addr_diff_states_t_lp1_reg = abi_param3;
        auto addr_diff_states_tp1_l_reg = abi_param4;
#ifdef _WIN32
        auto addr_diff_states_t_l_reg = r10;
        auto addr_states_tm1_l_reg = r11;
        auto base_args = get_stack_params_address();
        mov(addr_diff_states_t_l_reg, ptr[base_args]);
        mov(addr_states_tm1_l_reg, ptr[base_args + 8]);
#else
        auto addr_diff_states_t_l_reg = abi_param5;
        auto addr_states_tm1_l_reg = abi_param6;
#endif

        // helper lambda to address the gates and biases
        auto sg_addr = [&](int i) {
            return ptr[addr_scratch_gates_reg + i * rnn_.dic * scratch_dt_size];
        };
        auto wg_addr = [&](int i) {
            return ptr[addr_ws_gates_reg + i * rnn_.dic * gate_dt_size];
        };

        // initialize registers with addresses and constants
        mov(table_reg, table_label);
        init_regs(vlen);
        uni_vmovups(one_vmm, one_addr);

        mov(loop_cnt, rnn_.dic * scratch_dt_size);
        cmp(loop_cnt, vlen_scratch);
        jl(vector_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        L(vector_loop_start_label);
        {
            Vmm dG0(dG0_idx), dG2(dG2_idx), G0(G0_idx), G2(G2_idx),
                    dHt(dHt_idx), tmp1(tmp1_idx), tmp2(tmp2_idx), h(h_idx);

            to_float<src_data_t>(G0, wg_addr(0), vlen);
            to_float<src_data_t>(G2, wg_addr(2), vlen);

            // compute dHt
            uni_vmovups(dHt, ptr[addr_diff_states_tp1_l_reg]);
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovups(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddps(dHt, dHt, tmp1);

            // compute dG0
            to_float<src_data_t>(h, ptr[addr_states_tm1_l_reg], vlen);
            uni_vmovups(dG0, G0);
            uni_vmovups(tmp1, G0);
            uni_vfnmadd231ps(dG0, tmp1, tmp1); // (G0 - G0^2)
            uni_vsubps(h, h, G2); // (h - G2)
            uni_vmulps(dG0, dG0, h);
            uni_vmulps(dG0, dG0, dHt); // (h - G2) * (G0 - G0^2) * dHt

            // compute dG2
            uni_vmovups(tmp1, one_vmm);
            uni_vsubps(tmp1, tmp1, G0); // (1 - G0)
            uni_vmovups(dG2, one_vmm);
            uni_vmovups(tmp2, G2);
            uni_vfnmadd231ps(dG2, tmp2, tmp2); // (1 - G2^2)
            uni_vmulps(dG2, dG2, tmp1);
            uni_vmulps(dG2, dG2, dHt); //(1 - G0) * (1 - G2^2) * dHt

            // compute diff_state_t_l
            uni_vmulps(dHt, dHt, G0);
            uni_vmovups(ptr[addr_diff_states_t_l_reg], dHt);

            // downconvert and write data
            to_src<scratch_data_t>(sg_addr(0), dG0, vlen);
            to_src<scratch_data_t>(sg_addr(2), dG2, vlen);

            // increment address pointers
            add(addr_ws_gates_reg, vlen_scratch);
            add(addr_scratch_gates_reg, vlen_scratch);
            add(addr_diff_states_t_lp1_reg, vlen);
            add(addr_diff_states_tp1_l_reg, vlen);
            add(addr_diff_states_t_l_reg, vlen);
            add(addr_states_tm1_l_reg, vlen_scratch);
            inc_regs(vlen);

            // increment loop counter
            sub(loop_cnt, vlen_scratch);
            cmp(loop_cnt, vlen_scratch);
            jge(vector_loop_start_label);
        }
        L(vector_loop_end_label);

        cmp(loop_cnt, 0);
        je(rem_loop_end_label, Xbyak::CodeGenerator::T_NEAR);

        // Same code as above, we just use movuss for accessing inputs
        // TODO: smarter handling of tails with Zmm -> Ymm -> Xmm -> scalar
        L(rem_loop_start_label);
        {
            Xmm dG0(dG0_idx), dG2(dG2_idx), G0(G0_idx), G2(G2_idx),
                    dHt(dHt_idx), tmp1(tmp1_idx), tmp2(tmp2_idx), h(h_idx);

            to_float<src_data_t>(G0, wg_addr(0), hstate_dt_size);
            to_float<src_data_t>(G2, wg_addr(2), hstate_dt_size);

            // compute dHt
            uni_vmovss(dHt, ptr[addr_diff_states_tp1_l_reg]);
            // assumption: the diff_states_t_lp1 address is already offset by rnn.n_states
            uni_vmovss(tmp1, ptr[addr_diff_states_t_lp1_reg]);
            uni_vaddss(dHt, dHt, tmp1);

            // compute dG0
            to_float<src_data_t>(h, ptr[addr_states_tm1_l_reg], hstate_dt_size);
            uni_vmovss(dG0, G0);
            uni_vmovss(tmp1, G0);
            uni_vfnmadd231ps(dG0, tmp1, tmp1); // (G0 - G0^2)
            uni_vsubss(h, h, G2); // (h - G2)
            uni_vmulss(dG0, dG0, h);
            uni_vmulss(dG0, dG0, dHt); // (h - G2) * (G0 - G0^2) * dHt

            // compute dG2
            uni_vmovss(tmp1, one_xmm);
            uni_vsubss(tmp1, tmp1, G0); // (1 - G0)
            uni_vmovss(dG2, one_xmm);
            uni_vmovss(tmp2, G2);
            uni_vfnmadd231ps(dG2, tmp2, tmp2); // (1 - G2^2)
            uni_vmulss(dG2, dG2, tmp1);
            uni_vmulss(dG2, dG2, dHt); //(1 - G0) * (1 - G2^2) * dHt

            // compute diff_state_t_l
            uni_vmulss(dHt, dHt, G0);
            uni_vmovss(ptr[addr_diff_states_t_l_reg], dHt);

            // downconvert and write data
            to_src<scratch_data_t>(sg_addr(0), dG0, hstate_dt_size);
            to_src<scratch_data_t>(sg_addr(2), dG2, hstate_dt_size);

            // increment address pointers
            add(addr_ws_gates_reg, scratch_dt_size);
            add(addr_scratch_gates_reg, scratch_dt_size);
            add(addr_diff_states_t_lp1_reg, hstate_dt_size);
            add(addr_diff_states_tp1_l_reg, hstate_dt_size);
            add(addr_diff_states_t_l_reg, hstate_dt_size);
            add(addr_states_tm1_l_reg, scratch_dt_size);
            inc_regs(hstate_dt_size);

            // increment loop counter
            sub(loop_cnt, scratch_dt_size);
            jnz(rem_loop_start_label);
        }
        L(rem_loop_end_label);

        postamble();

        init_table(vlen);
        L(table_label);
        {
            for (size_t i = 0; i < vlen / sizeof(float); i++)
                dd(float2int(1.0f));
        }
    }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
