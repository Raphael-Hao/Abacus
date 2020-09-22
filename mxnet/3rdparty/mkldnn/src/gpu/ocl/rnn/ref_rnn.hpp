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

#ifndef GPU_OCL_RNN_REF_RNN_HPP
#define GPU_OCL_RNN_REF_RNN_HPP

#include <assert.h>
#include <stdio.h>

#include "common/c_types_map.hpp"
#include "common/primitive_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_rnn_pd.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/rnn/rnn_utils.hpp"
#include "gpu/primitive_conf.hpp"

// not implemented
#define USE_MKL_PACKED_GEMM 0

// TODO just to debug
#define WS_NAN_FILLING 0

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

enum gemm_kind_t {
    gemm_iter_fwd,
    gemm_layer_fwd,
    gemm_iter_bwd,
    gemm_layer_bwd,
    gemm_diff_wei_iter,
    gemm_diff_wei_layer
};

template <prop_kind_t aprop>
struct _ref_rnn_common_t : public primitive_impl_t {

    using class_name = _ref_rnn_common_t<aprop>;

    typedef elemwise_sig((class_name::*elemwise_f));
    typedef cell_execution_sig((class_name::*cell_execution_f));
    typedef grid_execution_sig((class_name::*grid_execution_f));

    typedef gemm_sig((class_name::*gemm_t));
    typedef packing_sig((class_name::*packing_t));
    typedef free_packed_sig((class_name::*free_packed_t));

    using base_pd_t =
            typename utils::conditional<false || aprop == prop_kind::forward,
                    gpu_rnn_fwd_pd_t, gpu_rnn_bwd_pd_t>::type;

    struct pd_t : public base_pd_t {

        using base_pd_t::base_pd_t;

        pd_t(const pd_t &other) : base_pd_t(other) { copy_from(other); }

        pd_t &operator=(const pd_t &other) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(other);
            base_pd_t::operator=(other);
            clear();
            copy_from(other);
            return *this;
        }

        ~pd_t() { clear(); }

        DECLARE_COMMON_PD_T("ref:any", class_name);

        status_t init();

        status_t set_default_params();

        rnn_conf_t conf;
        rnn_offsets_t off;
        rnn_utils::conf_t rnn_conf;
        data_type_t acc_data_t;
        data_type_t src_type;
        data_type_t weights_type;

        primitive_desc_t *gemm_iter_fwd_pd_ = nullptr;
        primitive_desc_t *gemm_layer_fwd_pd_ = nullptr;
        primitive_desc_t *gemm_iter_bwd_pd_ = nullptr;
        primitive_desc_t *gemm_layer_bwd_pd_ = nullptr;
        primitive_desc_t *gemm_diff_wei_layer_pd_ = nullptr;
        primitive_desc_t *gemm_diff_wei_iter_pd_ = nullptr;

    private:
        void init_scratchpad(size_t scratchpad_sz) {
            using namespace memory_tracking::names;
            auto scratchpad = this->scratchpad_registry().registrar();
            scratchpad.book(key_rnn_space, scratchpad_sz, 4096);
            scratchpad.book(key_rnn_gates, rnn_conf.scratch_gates_size, 4096);
        }

        void copy_from(const pd_t &other) {
            conf = other.conf;
            off = other.off;
            rnn_conf = other.rnn_conf;
            acc_data_t = other.acc_data_t;
            src_type = other.src_type;
            weights_type = other.weights_type;
            gemm_layer_fwd_pd_ = other.gemm_layer_fwd_pd_
                    ? other.gemm_layer_fwd_pd_->clone()
                    : nullptr;
            gemm_iter_fwd_pd_ = other.gemm_iter_fwd_pd_
                    ? other.gemm_iter_fwd_pd_->clone()
                    : nullptr;
            gemm_layer_bwd_pd_ = other.gemm_layer_bwd_pd_
                    ? other.gemm_layer_bwd_pd_->clone()
                    : nullptr;
            gemm_iter_bwd_pd_ = other.gemm_iter_bwd_pd_
                    ? other.gemm_iter_bwd_pd_->clone()
                    : nullptr;
            gemm_diff_wei_layer_pd_ = other.gemm_diff_wei_layer_pd_
                    ? other.gemm_diff_wei_layer_pd_->clone()
                    : nullptr;
            gemm_diff_wei_iter_pd_ = other.gemm_diff_wei_iter_pd_
                    ? other.gemm_diff_wei_iter_pd_->clone()
                    : nullptr;
        }

        void clear() {
            delete gemm_layer_fwd_pd_;
            delete gemm_iter_fwd_pd_;
            delete gemm_layer_bwd_pd_;
            delete gemm_iter_bwd_pd_;
            delete gemm_diff_wei_layer_pd_;
            delete gemm_diff_wei_iter_pd_;
        }

    }; // struct pd_t : public base_pd_t

    status_t init() override;

    _ref_rnn_common_t(const pd_t *apd) : primitive_impl_t(apd) {
        using namespace rnn_utils;
        /// @todo set max_feature_size assuming that we limit the number of
        /// iterations and layer to one if slc != dic and sic != dic
        /// respectively
        ///
        auto set_pack_funcs = [](bool packed_gemm, gemm_t &g, bool pack_w,
                                      packing_t &p, free_packed_t &f) {
            g = packed_gemm ? &class_name::packed_gemm
                            : &class_name::gemm_primitive;
            p = pack_w ? &class_name::pack_weights
                       : &class_name::no_pack_weights;
            f = pack_w ? &class_name::free_packed_weights
                       : &class_name::free_no_packed_weights;
        };

        set_pack_funcs(false, gemm_iter_func, false, weights_state_pack_func,
                weights_state_free_packed_func);

        set_pack_funcs(false, gemm_layer_func, false, weights_input_pack_func,
                weights_input_free_packed_func);

        switch (pd()->cell_kind()) {
            case alg_kind::vanilla_lstm:
                cell_func = &class_name::cell_execution;
                elemwise_func = pd()->src_type == data_type::u8
                                && pd()->weights_type == data_type::s8
                        ? &class_name::lstm_elemwise_u8s8
                        : &class_name::lstm_elemwise;
                break;
            case alg_kind::vanilla_rnn: // @todo switch on cell kind
                cell_func = &class_name::cell_execution;
                elemwise_func = &class_name::rnn_elemwise;
                break;
            case alg_kind::vanilla_gru:
                cell_func = &class_name::cell_execution_gru;
                break;
            case alg_kind::lbr_gru:
                cell_func = &class_name::cell_execution_gru_lbr;
                elemwise_func = &class_name::gru_lbr_elemwise;
                break;
            default: break;
        }

        /// @todo put a heuristic to choose between linear execution and
        /// wavefront
        grid_computation = &class_name::linear_execution;

        size_t scratchpad_size, workspace_size;
        rnn_utils::set_offsets(pd()->rnn_conf, ws_gates_offset_,
                ws_states_offset_, ws_c_states_offset_, ws_diff_states_offset_,
                ws_grid_comp_offset_, ws_cell_comp_offset_, ws_bias_offset_,
                scratch_gates_offset_, scratchpad_size, workspace_size);

        int max_nparts = (pd()->cell_kind() == alg_kind::vanilla_gru) ? 2 : 1;
        int offset_wei_sz = pd()->L() * pd()->D() * max_nparts;

        offset_wei_input_
                = (size_t *)malloc(sizeof(size_t) * offset_wei_sz, 64);
        offset_wei_state_
                = (size_t *)malloc(sizeof(size_t) * offset_wei_sz, 64);
    }

    ~_ref_rnn_common_t() {
        free(offset_wei_input_);
        free(offset_wei_state_);

        delete gemm_iter_fwd_;
        delete gemm_layer_fwd_;
        delete gemm_iter_bwd_;
        delete gemm_layer_bwd_;
        delete gemm_diff_wei_layer_;
        delete gemm_diff_wei_iter_;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_(ctx);
    }

private:
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    grid_execution_sig(linear_execution);
    // grid_execution_sig(wavefront_execution);
    cell_execution_sig(cell_execution);
    cell_execution_sig(cell_execution_gru);
    cell_execution_sig(cell_execution_gru_lbr);
    elemwise_sig(rnn_elemwise);
    elemwise_sig(lstm_elemwise);
    elemwise_sig(lstm_elemwise_u8s8);
    elemwise_sig(gru_lbr_elemwise);
    gemm_sig(gemm_primitive);
    gemm_sig(packed_gemm);
    packing_sig(pack_weights);
    packing_sig(no_pack_weights);
    free_packed_sig(free_packed_weights);
    free_packed_sig(free_no_packed_weights);

    float (*activation_func)(float dd, float s, float alpha, float cliping);
    void bias_prepare(compute::compute_stream_t *compute_stream, int n_layer,
            int n_dir, int n_bias, int n_gates, int dic,
            const memory_storage_t &ws, const memory_storage_t &scales,
            const memory_storage_t &wei_layer, const memory_storage_t &wei_iter,
            const memory_storage_t &bias) const;
    void copy_init_layer(compute::compute_stream_t *compute_stream, bool lr,
            bool rl, int n_iter, int batch, int slc, const memory_storage_t &ws,
            const memory_storage_t &input,
            const memory_storage_t &diff_dst_layer) const;
    void copy_init_iter(compute::compute_stream_t *compute_stream, int n_layer,
            int n_dir, int batch, int sic, int dic, const memory_storage_t &ws,
            const memory_storage_t &firstit_states,
            const memory_storage_t &firstit_c_states,
            const memory_storage_t &diff_dst_iter,
            const memory_storage_t &diff_dst_iter_c, const float shift,
            const float scale, const bool quantize) const;
    void copy_res_layer(compute::compute_stream_t *compute_stream, bool lr,
            bool rl, int n_iter, int batch, int slc, int dlc,
            const memory_storage_t &dst_last_layer,
            const memory_storage_t &diff_src_layer, const memory_storage_t &ws,
            const float shift, const float scale, const bool dequantize) const;
    void copy_res_iter(compute::compute_stream_t *compute_stream, int n_layer,
            int n_dir, int batch, int sic, int dic,
            const memory_storage_t &dst_last_iter,
            const memory_storage_t &dst_last_iter_c,
            const memory_storage_t &diff_src_iter,
            const memory_storage_t &diff_src_iter_c, const memory_storage_t &ws,
            const float shift, const float scale, const bool dequantize) const;
    void gates_reduction(const exec_ctx_t &ctx, int dir, int lay, int iter,
            int n_gates, int dic, int batch, const memory_storage_t &gates,
            const memory_storage_t &diff_bias) const;
    void ws_set(compute::compute_stream_t *compute_stream,
            const memory_storage_t &workspace, const cl_ulong ws_offset,
            const int ws_part, const float val, const size_t size) const;
#if DEBUGPRINT
    void ws_print(compute::compute_stream_t *s,
            const memory_storage_t &workspace) const;
    compute::kernel_t ws_print_kernel_;
#endif

    compute::kernel_t bias_prepare_kernel_;
    compute::kernel_t copy_init_layer_kernel_;
    compute::kernel_t copy_init_iter_kernel_;
    compute::kernel_t copy_res_layer_kernel_;
    compute::kernel_t copy_res_iter_kernel_;

    compute::kernel_t ws_set_kernel_;
    compute::kernel_t elemwise_fwd_kernel_;
    compute::kernel_t elemwise_bwd_kernel_;
    compute::kernel_t gates_reduction_kernel_;

    // GEMM primitives.
    primitive_t *gemm_layer_fwd_ = nullptr;
    primitive_t *gemm_iter_fwd_ = nullptr;
    primitive_t *gemm_layer_bwd_ = nullptr;
    primitive_t *gemm_iter_bwd_ = nullptr;
    primitive_t *gemm_diff_wei_layer_ = nullptr;
    primitive_t *gemm_diff_wei_iter_ = nullptr;

    std::unique_ptr<memory_storage_t> scales_buf_;
    std::unique_ptr<memory_storage_t> tm_scales_buf_;

    cl_ulong ws_gates_offset_;
    cl_ulong ws_states_offset_;
    cl_ulong ws_c_states_offset_;
    cl_ulong ws_diff_states_offset_;
    cl_ulong ws_grid_comp_offset_;
    cl_ulong ws_cell_comp_offset_;
    cl_ulong ws_bias_offset_;
    cl_ulong scratch_gates_offset_;

    size_t *offset_wei_input_;
    size_t *offset_wei_state_;

    grid_execution_f grid_computation;
    cell_execution_f cell_func;

    packing_t weights_input_pack_func;
    packing_t weights_state_pack_func;

    gemm_t gemm_iter_func;
    gemm_t gemm_layer_func;
    elemwise_f elemwise_func;

    free_packed_t weights_input_free_packed_func;
    free_packed_t weights_state_free_packed_func;
};
using ref_rnn_fwd_t = _ref_rnn_common_t<prop_kind::forward>;
using ref_rnn_bwd_t = _ref_rnn_common_t<prop_kind::backward>;
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
