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

// General architecture
//
// for diff states, we have n_states + 1 as we have n_states diff
// to propagate to the previous iteration and 1 states to propagate
// to the previous layer
// index 0 is dh for cell(t-1, l) to consume
// index 1 is dc for cell(t-1, l) to consume
// index 2 is dh for cell(t, l-1) to consume
// this indexing enables to have the same indexing for states in elemwise
// function
// only the cell execution function should be impacted

#include "gpu/ocl/rnn/ref_rnn.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gemm/gpu_gemm_utils.hpp"

#define DEBUGPRINT 0
#if DEBUGPRINT
#define DPRINT(fmt, ...) \
    printf(fmt, __VA_ARGS__); \
    fflush(0)
#define WS_PRINT(s, w) ws_print(s, w)
#else
#define DPRINT(fmt, ...)
#define WS_PRINT(s, w)
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::math;
using namespace prop_kind;
using namespace alg_kind;
using namespace rnn_utils;
using namespace dnnl::impl::memory_tracking::names;

#define AOC array_offset_calculator

static status_t init_conf(rnn_conf_t &conf, const rnn_pd_t *rnn_pd,
        const rnn_utils::conf_t &rnn, const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &bias_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d,
        const memory_desc_wrapper &diff_src_layer_d,
        const memory_desc_wrapper &diff_src_iter_d,
        const memory_desc_wrapper &diff_src_iter_c_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d,
        const memory_desc_wrapper &diff_bias_d,
        const memory_desc_wrapper &diff_dst_layer_d,
        const memory_desc_wrapper &diff_dst_iter_d,
        const memory_desc_wrapper &diff_dst_iter_c_d,
        const memory_desc_wrapper &ws_d, rnn_offsets_t &off) {

    using namespace rnn_utils;

    conf.src_dt = src_layer_d.data_type();
    conf.wei_dt = weights_layer_d.data_type();
    conf.acc_dt = rnn.acc_data_type;
    conf.aux_dt = rnn.aux_data_type;
    conf.diff_dt = rnn.diff_data_type;
    conf.input_dt = rnn.input_data_type;
    conf.output_dt = rnn.output_data_type;
    conf.dst_dt = rnn.dst_data_type;

    conf.is_fwd = rnn.is_fwd;
    conf.n_layer = rnn.n_layer;
    conf.n_dir = rnn.n_dir;
    conf.n_iter = rnn.n_iter;
    conf.n_iter_scratch_gates = rnn.n_iter_scratch_gates;
    conf.n_gates = rnn.n_gates;
    conf.n_bias = rnn.n_bias;
    conf.n_states = rnn.n_states;
    conf.n_weights_input = weights_layer_d.dims()[2];
    conf.n_weights_state = weights_iter_d.dims()[2];
    conf.batch = rnn.mb;
    conf.slc = rnn.slc;
    conf.sic = rnn.sic;
    conf.dic = rnn.dic;
    conf.dlc = rnn.dlc;
    conf.wic = nstl::max(conf.slc, nstl::max(conf.sic, conf.dic));

    conf.n_parts_weights_iter = rnn.n_parts_weights_iter;
    conf.n_parts_weights_layer = rnn.n_parts_weights_layer;

    conf.with_bias = rnn_pd->with_bias();
    conf.with_src_iter = rnn_pd->with_src_iter();
    conf.with_src_iter_c = rnn_pd->with_src_iter_c();
    conf.with_dst_iter = rnn_pd->with_dst_iter();
    conf.with_dst_iter_c = rnn_pd->with_dst_iter_c();
    conf.is_lbr = rnn.is_lbr;
    conf.copy_bias = rnn.copy_bias;
    conf.is_int8 = rnn.is_int8;
    conf.is_training = rnn.is_training;

    conf.states_ws_ld = rnn.states_ws_ld;
    conf.diff_states_ws_ld = rnn.diff_states_ws_ld;
    conf.gates_ws_ld = rnn.gates_ws_ld;
    conf.scratch_gates_ld = rnn.scratch_gates_ld;

    conf.src_layer_ndims = src_layer_d.ndims();
    conf.src_iter_ndims = src_iter_d.ndims();
    if (conf.with_src_iter_c) conf.src_iter_c_ndims = src_iter_c_d.ndims();
    conf.weights_layer_ndims = weights_layer_d.ndims();
    conf.weights_iter_ndims = weights_iter_d.ndims();
    conf.dst_layer_ndims = dst_layer_d.ndims();
    conf.dst_iter_ndims = dst_iter_d.ndims();
    if (conf.with_dst_iter_c) conf.dst_iter_c_ndims = dst_iter_c_d.ndims();
    conf.bias_ndims = bias_d.ndims();

    gpu::set_offsets(src_layer_d, off.src_layer_off);
    gpu::set_offsets(src_iter_d, off.src_iter_off);
    if (conf.with_src_iter_c)
        gpu::set_offsets(src_iter_c_d, off.src_iter_c_off);
    gpu::set_offsets(weights_layer_d, off.weights_layer_off);
    gpu::set_offsets(weights_iter_d, off.weights_iter_off);
    gpu::set_offsets(bias_d, off.bias_off);
    gpu::set_offsets(dst_layer_d, off.dst_layer_off);
    gpu::set_offsets(dst_iter_d, off.dst_iter_off);
    if (conf.with_dst_iter_c)
        gpu::set_offsets(dst_iter_c_d, off.dst_iter_c_off);

    if (!conf.is_fwd) {
        conf.diff_src_layer_ndims = diff_src_layer_d.ndims();
        conf.diff_src_iter_ndims = diff_src_iter_d.ndims();
        if (conf.with_src_iter_c)
            conf.diff_src_iter_c_ndims = diff_src_iter_c_d.ndims();
        conf.diff_weights_layer_ndims = diff_weights_layer_d.ndims();
        conf.diff_weights_iter_ndims = diff_weights_iter_d.ndims();
        conf.diff_dst_layer_ndims = diff_dst_layer_d.ndims();
        conf.diff_dst_iter_ndims = diff_dst_iter_d.ndims();
        if (conf.with_dst_iter_c)
            conf.diff_dst_iter_c_ndims = diff_dst_iter_c_d.ndims();
        conf.diff_bias_ndims = diff_bias_d.ndims();

        gpu::set_offsets(diff_src_layer_d, off.diff_src_layer_off);
        gpu::set_offsets(diff_src_iter_d, off.diff_src_iter_off);
        if (conf.with_src_iter_c)
            gpu::set_offsets(diff_src_iter_c_d, off.diff_src_iter_c_off);
        gpu::set_offsets(diff_weights_layer_d, off.diff_weights_layer_off);
        gpu::set_offsets(diff_weights_iter_d, off.diff_weights_iter_off);
        gpu::set_offsets(diff_bias_d, off.diff_bias_off);
        gpu::set_offsets(diff_dst_layer_d, off.diff_dst_layer_off);
        gpu::set_offsets(diff_dst_iter_d, off.diff_dst_iter_off);
        if (conf.with_dst_iter_c)
            gpu::set_offsets(diff_dst_iter_c_d, off.diff_dst_iter_c_off);
    }

    rnn_utils::set_offsets(rnn, conf.ws_gates_offset, conf.ws_states_offset,
            conf.ws_c_state_offset, conf.ws_diff_states_offset,
            conf.ws_grid_comp_offset, conf.ws_cell_comp_offset,
            conf.ws_bias_offset, conf.scratch_gates_offset,
            conf.scratchpad_size, conf.workspace_size);

    conf.cell_kind = rnn_pd->cell_kind();
    conf.activation_kind = rnn_pd->activation_kind();
    conf.direction_kind = rnn_pd->direction();

    conf.wei_qparam_mask = rnn_pd->attr()->rnn_weights_qparams_.mask_;
    conf.is_testmode = rnn.is_testmode;

    return status::success;
}

static status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx,
        const rnn_conf_t &conf, const rnn_offsets_t &off) {

    kernel_ctx.define_int("IS_FWD", conf.is_fwd);
    kernel_ctx.define_int("IS_TRAINING", conf.is_training);
    kernel_ctx.define_int("WITH_BIAS", conf.with_bias);
    kernel_ctx.define_int("WITH_SRC_ITER", conf.with_src_iter);
    kernel_ctx.define_int("WITH_SRC_ITER_C", conf.with_src_iter_c);
    kernel_ctx.define_int("WITH_DST_ITER", conf.with_dst_iter);
    kernel_ctx.define_int("WITH_DST_ITER_C", conf.with_dst_iter_c);
    kernel_ctx.define_int("IS_LBR", conf.is_lbr);

    kernel_ctx.define_int("VANILLA_RNN", alg_kind::vanilla_rnn);
    kernel_ctx.define_int("VANILLA_LSTM", alg_kind::vanilla_lstm);
    kernel_ctx.define_int("VANILLA_GRU", alg_kind::vanilla_gru);
    kernel_ctx.define_int("LBR_GRU", alg_kind::lbr_gru);
    kernel_ctx.define_int("CELL_KIND", conf.cell_kind);

    kernel_ctx.define_int("ELTWISE_RELU", alg_kind::eltwise_relu);
    kernel_ctx.define_int("ELTWISE_TANH", alg_kind::eltwise_tanh);
    kernel_ctx.define_int("ELTWISE_LOGISTIC", alg_kind::eltwise_logistic);
    kernel_ctx.define_int("ACTIVATION_KIND", conf.activation_kind);

    kernel_ctx.define_int("WS_GATES", rnn_utils::gates);
    kernel_ctx.define_int("WS_STATES", rnn_utils::states);
    kernel_ctx.define_int("WS_C_STATES", rnn_utils::c_states);
    kernel_ctx.define_int("WS_DIFF_STATES", rnn_utils::diff_states);
    kernel_ctx.define_int("WS_BIAS", rnn_utils::bias);

    kernel_ctx.define_int("L2R", dnnl_unidirectional_left2right);
    kernel_ctx.define_int("R2L", dnnl_unidirectional_right2left);
    kernel_ctx.define_int("CONCAT", dnnl_bidirectional_concat);
    kernel_ctx.define_int("SUM", dnnl_bidirectional_sum);
    kernel_ctx.define_int("UNIDEF", dnnl_unidirectional);
    kernel_ctx.define_int("DIRECTION_KIND", conf.direction_kind);

    kernel_ctx.define_int("BATCH", conf.batch);
    kernel_ctx.define_int("N_DIR", conf.n_dir);
    kernel_ctx.define_int("N_LAYER", conf.n_layer);
    kernel_ctx.define_int("N_ITER", conf.n_iter);
    kernel_ctx.define_int("N_ITER_SCRATCH_GATES", conf.n_iter_scratch_gates);
    kernel_ctx.define_int("N_GATES", conf.n_gates);
    kernel_ctx.define_int("N_BIAS", conf.n_bias);
    kernel_ctx.define_int("N_STATES", conf.n_states);

    kernel_ctx.define_int("SLC", conf.slc);
    kernel_ctx.define_int("SIC", conf.sic);
    kernel_ctx.define_int("DIC", conf.dic);
    kernel_ctx.define_int("WIC", conf.wic);

    kernel_ctx.define_int("N_PARTS_WEI_ST", conf.n_parts_weights_iter);
    kernel_ctx.define_int("N_PARTS_WEI_I", conf.n_parts_weights_layer);

    def_offsets(off.src_layer_off, kernel_ctx, "SRC_L", conf.src_layer_ndims);
    def_offsets(off.src_iter_off, kernel_ctx, "SRC_I", conf.src_iter_ndims);
    if (conf.with_src_iter_c)
        def_offsets(off.src_iter_c_off, kernel_ctx, "SRC_I_C",
                conf.src_iter_c_ndims);
    def_offsets(off.weights_layer_off, kernel_ctx, "WEI_L",
            conf.weights_layer_ndims);
    def_offsets(
            off.weights_iter_off, kernel_ctx, "WEI_I", conf.weights_iter_ndims);
    def_offsets(off.dst_layer_off, kernel_ctx, "DST_L", conf.dst_layer_ndims);
    def_offsets(off.dst_iter_off, kernel_ctx, "DST_I", conf.dst_iter_ndims);
    if (conf.with_dst_iter_c)
        def_offsets(off.dst_iter_c_off, kernel_ctx, "DST_I_C",
                conf.dst_iter_c_ndims);
    def_offsets(off.bias_off, kernel_ctx, "BIAS", conf.bias_ndims);

    if (!conf.is_fwd) {
        def_offsets(off.diff_src_layer_off, kernel_ctx, "DIFF_SRC_L",
                conf.diff_src_layer_ndims);
        def_offsets(off.diff_src_iter_off, kernel_ctx, "DIFF_SRC_I",
                conf.diff_src_iter_ndims);
        if (conf.with_src_iter_c)
            def_offsets(off.diff_src_iter_c_off, kernel_ctx, "DIFF_SRC_I_C",
                    conf.diff_src_iter_c_ndims);
        def_offsets(off.diff_weights_layer_off, kernel_ctx, "DIFF_WEI_L",
                conf.diff_weights_layer_ndims);
        def_offsets(off.diff_weights_iter_off, kernel_ctx, "DIFF_WEI_I",
                conf.diff_weights_iter_ndims);
        def_offsets(off.diff_dst_layer_off, kernel_ctx, "DIFF_DST_L",
                conf.diff_dst_layer_ndims);
        def_offsets(off.diff_dst_iter_off, kernel_ctx, "DIFF_DST_I",
                conf.diff_dst_iter_ndims);
        if (conf.with_dst_iter_c)
            def_offsets(off.diff_dst_iter_c_off, kernel_ctx, "DIFF_DST_I_C",
                    conf.diff_dst_iter_c_ndims);
        def_offsets(off.diff_bias_off, kernel_ctx, "DIFF_BIAS",
                conf.diff_bias_ndims);
    }

    kernel_ctx.define_int("WS_GATES_OFFSET", conf.ws_gates_offset);
    kernel_ctx.define_int("WS_STATES_OFFSET", conf.ws_states_offset);
    kernel_ctx.define_int("WS_C_STATE_OFFSET", conf.ws_c_state_offset);
    kernel_ctx.define_int("WS_DIFF_STATES_OFFSET", conf.ws_diff_states_offset);
    kernel_ctx.define_int("WS_GRID_COMP_OFFSET", conf.ws_grid_comp_offset);
    kernel_ctx.define_int("WS_CELL_COMP_OFFSET", conf.ws_cell_comp_offset);
    kernel_ctx.define_int("WS_BIAS_OFFSET", conf.ws_bias_offset);
    kernel_ctx.define_int("SCRATCH_GATES_OFFSET", conf.scratch_gates_offset);
    kernel_ctx.define_int("STATES_WS_LD", conf.states_ws_ld);
    kernel_ctx.define_int("DIFF_STATES_WS_LD", conf.diff_states_ws_ld);
    kernel_ctx.define_int("GATES_WS_LD", conf.gates_ws_ld);
    kernel_ctx.define_int("SCRATCH_GATES_LD", conf.scratch_gates_ld);

    if (conf.src_dt == data_type::f16) {
        kernel_ctx.set_data_type(data_type::f16);
    } else
        kernel_ctx.set_data_type(data_type::f32);

    def_data_type(kernel_ctx, conf.src_dt, "WS_STATE");
    def_data_type(kernel_ctx, conf.src_dt, "SRC");
    def_data_type(kernel_ctx, conf.wei_dt, "WEI");
    def_data_type(kernel_ctx, conf.acc_dt, "ACC");
    def_data_type(kernel_ctx, conf.aux_dt, "AUX");
    def_data_type(kernel_ctx, conf.dst_dt, "DST");
    def_data_type(kernel_ctx, conf.input_dt, "INPUT");
    def_data_type(kernel_ctx, conf.output_dt, "OUTPUT");
    def_data_type(kernel_ctx, conf.diff_dt, "DIFF");

    kernel_ctx.define_int("IS_INT8", conf.is_int8);
    kernel_ctx.define_int("COPY_BIAS", conf.copy_bias);
    kernel_ctx.define_int("WEI_QPARAM_MASK", conf.wei_qparam_mask);
    kernel_ctx.define_int("IS_TESTMODE", conf.is_testmode);

    kernel_ctx.define_int("DEBUGPRINT", DEBUGPRINT);

#if DEBUGPRINT
    kernel_ctx.print_options();
#endif
    return status::success;
}

template <prop_kind_t aprop>
inline status_t init_conf(rnn_conf_t &conf, const rnn_utils::conf_t &rnn,
        const rnn_pd_t *rnn_pd, rnn_offsets_t &off) {

    const memory_desc_wrapper fakedesc = rnn_pd->src_md(0);
    return init_conf(conf, rnn_pd, rnn, rnn_pd->src_md(0), rnn_pd->src_md(1),
            rnn_pd->src_md(2), rnn_pd->weights_md(0), rnn_pd->weights_md(1),
            rnn_pd->weights_md(2), rnn_pd->dst_md(0), rnn_pd->dst_md(1),
            rnn_pd->dst_md(2), fakedesc, fakedesc, fakedesc, fakedesc, fakedesc,
            fakedesc, fakedesc, fakedesc, fakedesc, rnn_pd->workspace_md(0),
            off);
}

template <>
inline status_t init_conf<prop_kind::backward>(rnn_conf_t &conf,
        const rnn_utils::conf_t &rnn, const rnn_pd_t *rnn_pd,
        rnn_offsets_t &off) {
    return init_conf(conf, rnn_pd, rnn, rnn_pd->src_md(0), rnn_pd->src_md(1),
            rnn_pd->src_md(2), rnn_pd->weights_md(0), rnn_pd->weights_md(1),
            rnn_pd->weights_md(2), rnn_pd->dst_md(0), rnn_pd->dst_md(1),
            rnn_pd->dst_md(2), rnn_pd->diff_src_md(0), rnn_pd->diff_src_md(1),
            rnn_pd->diff_src_md(2), rnn_pd->diff_weights_md(0),
            rnn_pd->diff_weights_md(1), rnn_pd->diff_weights_md(2),
            rnn_pd->diff_dst_md(0), rnn_pd->diff_dst_md(1),
            rnn_pd->diff_dst_md(2), rnn_pd->workspace_md(0), off);
}

template <>
status_t _ref_rnn_common_t<prop_kind::forward>::pd_t::set_default_params() {
    using namespace format_tag;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&src_iter_c_md_))
            && (src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&dst_iter_c_md_))
            && (dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

    return status::success;
}

template <>
status_t _ref_rnn_common_t<prop_kind::backward>::pd_t::set_default_params() {
    using namespace format_tag;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (weights_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(weights_layer_md_, ldgoi));
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    if (weights_iter_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(weights_iter_md_, ldgoi));

    if (diff_src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_src_layer_md_, tnc));
    if (diff_weights_layer_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_layer_md_, ldigo));
        CHECK(rnn_utils::set_good_strides(diff_weights_layer_md_, ldigo));
    }
    if (diff_weights_iter_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_iter_md_, ldigo));
        CHECK(rnn_utils::set_good_strides(diff_weights_iter_md_, ldigo));
    }
    if (diff_dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&src_iter_c_md_))
            && (src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&dst_iter_c_md_))
            && (dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

    if ((!types::is_zero_md(&diff_src_iter_md_))
            && (diff_src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_src_iter_c_md_))
            && (diff_src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&diff_bias_md_))
            && (diff_bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_bias_md_, ldgo));
    if ((!types::is_zero_md(&diff_dst_iter_md_))
            && (diff_dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_dst_iter_c_md_))
            && (diff_dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_c_md_, ldnc));

    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::pd_t::init() {
    using namespace prop_kind;
    using namespace utils;
    using namespace rnn_utils;
    using namespace format_tag;

    assert(this->engine()->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<const compute::compute_engine_t *>(
            this->engine());

    const alg_kind_t cell_kind = this->desc()->cell_kind;

    data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
    data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
    bool src_is_u8 = src_layer_dt == data_type::u8;
    bool src_is_f16 = src_layer_dt == data_type::f16;
    if (src_is_u8 && !src_is_f16)
        acc_data_t = data_type::s32;
    else if (!src_is_u8 && src_is_f16)
        acc_data_t = data_type::f16;
    else if (!src_is_u8 && !src_is_f16)
        acc_data_t = data_type::f32;
    src_type = src_layer_dt;
    weights_type = weights_layer_dt;

    bool ok = true
            && one_of(cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm)
            && !this->is_lstm_peephole()
            && IMPLICATION(aprop == prop_kind::forward,
                    one_of(this->desc()->prop_kind, forward_training,
                            forward_inference))
            && IMPLICATION(aprop == backward,
                    one_of(this->desc()->prop_kind, backward))
            && src_layer_dt == src_type
            && ((aprop == prop_kind::forward && src_layer_dt == data_type::u8
                        && weights_layer_dt == data_type::s8)
                    || (aprop == prop_kind::forward
                            && one_of(src_layer_dt, data_type::f16,
                                    data_type::f32, data_type::bf16)
                            && weights_layer_dt == src_layer_dt)
                    || (aprop == prop_kind::backward
                            && one_of(weights_layer_dt, data_type::f32,
                                    data_type::bf16)
                            && weights_layer_dt == src_layer_dt))
            && weights_iter_dt == weights_layer_dt
            && everyone_is(weights_type, weights_iter_dt, weights_layer_dt)
            && this->set_default_params() == status::success
            && this->with_bias()
            && IMPLICATION(
                    src_type == data_type::f16 || src_type == data_type::u8,
                    this->desc()->prop_kind == forward_inference)
            && compute_engine->mayiuse(compute::device_ext_t::intel_subgroups)
            && IMPLICATION(src_type == data_type::f16,
                    true
                            && compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)
                            && compute_engine->mayiuse(compute::device_ext_t::
                                            intel_subgroups_short));
    if (!ok) return status::unimplemented;

    init_rnn_conf(rnn_conf, *this->desc(), this->src_md(0), this->src_md(1),
            this->weights_md(0), this->weights_md(1), this->dst_md(0));
    init_test_mode(rnn_conf, *this->attr());

    // Check that only supported attr have been passed.
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8)
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams;
    ok = ok && this->attr()->has_default_values(attr_mask);

    // TODO: implement something like check layout consistency
    switch (aprop) {
        case (prop_kind::forward): break;
        case (prop_kind::backward):
            ok = ok && utils::one_of(this->desc()->prop_kind, backward);
            ok = ok
                    && memory_desc_matches_one_of_tag(
                            this->weights_layer_md_, ldgoi)
                    && memory_desc_matches_one_of_tag(
                            this->weights_iter_md_, ldgoi);
            break;
        default: ok = false;
    }
    if (!ok) return status::unimplemented;

    // Set weights descriptors to desired format
    memory_desc_t new_weights_layer_md = *this->weights_md(0);
    CHECK(set_expected_desc(rnn_conf, new_weights_layer_md, false));

    if (this->weights_layer_md_.format_kind == format_kind::any) {
        this->weights_layer_md_ = new_weights_layer_md;
    } else if (this->weights_layer_md_.format_kind == format_kind::rnn_packed) {
        if (dnnl::impl::operator!=(
                    this->weights_layer_md_, new_weights_layer_md))
            return status::unimplemented;
    }

    memory_desc_t new_weights_iter_md = *this->weights_md(1);
    CHECK(set_expected_desc(rnn_conf, new_weights_iter_md, true));
    if (this->weights_iter_md_.format_kind == format_kind::any) {
        this->weights_iter_md_ = new_weights_iter_md;
    } else if (this->weights_iter_md_.format_kind == format_kind::rnn_packed) {
        if (dnnl::impl::operator!=(this->weights_iter_md_, new_weights_iter_md))
            return status::unimplemented;
    }

    // Check dimensions consistency
    int ls_multiplier
            = (this->direction() == dnnl_bidirectional_concat) ? 2 : 1;

    ok = ok && (ls_multiplier * this->DIC() == this->DLC())
            && ((ls_multiplier * this->SLC()) == this->DLC()
                    || (this->L() == 1))
            && (this->SIC() == this->DIC() || (this->T() == 1));
    if (!ok) return status::unimplemented;

    set_rnn_conf(rnn_conf, *this->desc(), this->weights_md(0),
            this->weights_md(1), this->diff_weights_md(0),
            this->diff_weights_md(1));

    size_t scratchpad_sz {0}, ws_sz {0};
    get_scratchpad_and_workspace_sizes(rnn_conf, scratchpad_sz, ws_sz);

    // initialize the workspace_pd if needed
    if (rnn_conf.use_workspace) {
        dims_t ws_dims = {(dim_t)ws_sz};
        dnnl_memory_desc_init_by_tag(
                &this->ws_md_, 1, ws_dims, data_type::u8, x);
    }

    init_scratchpad(scratchpad_sz);

    rnn_conf.acc_data_type = acc_data_t;
    rnn_conf.acc_data_type_elsz = types::data_type_size(acc_data_t);
    status_t status = init_conf<aprop>(conf, rnn_conf, this, this->off);
    if (status != status::success) { return status; }

    auto create_gemm_pd = [&](primitive_desc_t **gemm_pd, int m, int n, int k,
                                  int lda, int ldb, int ldc, data_type_t a_dt,
                                  data_type_t b_dt, data_type_t c_dt,
                                  bool is_B_trans, float beta) -> status_t {
        gemm_desc_t gemm_desc;
        gemm_desc.primitive_kind = primitive_kind::gemm;
        gemm_desc.transa = transpose::notrans;
        gemm_desc.transb = is_B_trans ? transpose::trans : transpose::notrans;
        gemm_desc.batch = 1;
        gemm_desc.m = m;
        gemm_desc.n = n;
        gemm_desc.k = k;
        gemm_desc.lda = lda;
        gemm_desc.ldb = ldb;
        gemm_desc.ldc = ldc;
        gemm_desc.stride_a = lda;
        gemm_desc.stride_b = ldb;
        gemm_desc.stride_c = ldc;
        gemm_desc.a_type = a_dt;
        gemm_desc.b_type = b_dt;
        gemm_desc.c_type = c_dt;
        gemm_desc.acc_type = c_dt;

        primitive_attr_t dummy_attr;
        dummy_attr.post_ops_.append_sum(beta);

        return dnnl_primitive_desc_create(gemm_pd, (op_desc_t *)&gemm_desc,
                &dummy_attr, this->engine(), nullptr);
    };

    int batch = rnn_conf.mb;
    int n_gates = rnn_conf.n_gates;
    int slc = rnn_conf.slc;
    int sic = rnn_conf.sic;
    int dic = rnn_conf.dic;

    int layer_merged_size
            = rnn_conf.merge_gemm_layer ? batch * rnn_conf.n_iter : batch;
    int iter_merged_size
            = rnn_conf.merge_gemm_iter ? batch * rnn_conf.n_iter : batch;

    bool gemm_ok = true;

    switch (aprop) {
        case prop_kind::forward:
            gemm_ok = true
                    && utils::everyone_is(status::success,
                            create_gemm_pd(&gemm_layer_fwd_pd_, n_gates * dic,
                                    layer_merged_size, slc,
                                    rnn_conf.weights_layer_ld,
                                    rnn_conf.states_ws_ld, rnn_conf.gates_ws_ld,
                                    weights_type, src_type,
                                    rnn_conf.acc_data_type, false, 0.0),
                            create_gemm_pd(&gemm_iter_fwd_pd_, n_gates * dic,
                                    batch, sic, rnn_conf.weights_iter_ld,
                                    rnn_conf.states_ws_ld, rnn_conf.gates_ws_ld,
                                    weights_type, src_type,
                                    rnn_conf.acc_data_type, false, 1.0));
            break;
        case prop_kind::backward:
            gemm_ok = true
                    && utils::everyone_is(status::success,
                            create_gemm_pd(&gemm_iter_bwd_pd_, sic, batch,
                                    n_gates * dic, rnn_conf.weights_iter_ld,
                                    rnn_conf.scratch_gates_ld,
                                    rnn_conf.diff_states_ws_ld, weights_type,
                                    src_type, rnn_conf.acc_data_type, false,
                                    0.0f),
                            create_gemm_pd(&gemm_layer_bwd_pd_, slc,
                                    layer_merged_size, n_gates * dic,
                                    rnn_conf.weights_layer_ld,
                                    rnn_conf.scratch_gates_ld,
                                    rnn_conf.diff_states_ws_ld, weights_type,
                                    src_type, rnn_conf.acc_data_type, false,
                                    0.0f),
                            create_gemm_pd(&gemm_diff_wei_layer_pd_,
                                    n_gates * dic, slc, layer_merged_size,
                                    rnn_conf.scratch_gates_ld,
                                    rnn_conf.states_ws_ld,
                                    rnn_conf.diff_weights_layer_ld,
                                    weights_type, src_type,
                                    rnn_conf.acc_data_type, true, 1.0f),
                            create_gemm_pd(&gemm_diff_wei_iter_pd_,
                                    n_gates * dic, sic, iter_merged_size,
                                    rnn_conf.scratch_gates_ld,
                                    rnn_conf.states_ws_ld,
                                    rnn_conf.diff_weights_iter_ld, weights_type,
                                    src_type, rnn_conf.acc_data_type, true,
                                    1.0f));
            break;
        default: assert(!"unknown prop_kind"); return status::invalid_arguments;
    }

    if (!gemm_ok) return status::unimplemented;
    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::init() {
    auto *compute_engine
            = utils::downcast<compute::compute_engine_t *>(engine());
    compute::kernel_ctx_t kernel_ctx;

    status_t status = init_kernel_ctx(kernel_ctx, pd()->conf, pd()->off);
    CHECK(status);

    std::vector<const char *> kernel_names
            = { "ref_rnn_bias_prepare",
                  "ref_rnn_copy_init_layer",
                  "ref_rnn_copy_init_iter",
                  "ref_rnn_copy_res_layer",
                  "ref_rnn_copy_res_iter",
                  "ref_rnn_ws_set",
                  "ref_rnn_elemwise_fwd",
                  "ref_rnn_elemwise_bwd",
                  "ref_rnn_gates_reduction"
#if DEBUGPRINT
                  ,
                  "ref_rnn_ws_print"
#endif
              };

    std::vector<compute::kernel_t> kernels;
    status = compute_engine->create_kernels(&kernels, kernel_names, kernel_ctx);
    CHECK(status);

    bias_prepare_kernel_ = kernels[0];
    copy_init_layer_kernel_ = kernels[1];
    copy_init_iter_kernel_ = kernels[2];
    copy_res_layer_kernel_ = kernels[3];
    copy_res_iter_kernel_ = kernels[4];
    ws_set_kernel_ = kernels[5];
    elemwise_fwd_kernel_ = kernels[6];
    elemwise_bwd_kernel_ = kernels[7];
    gates_reduction_kernel_ = kernels[8];
#if DEBUGPRINT
    ws_print_kernel_ = kernels[9];
#endif

    if (pd()->rnn_conf.is_int8) {
        size_t size = pd()->rnn_conf.n_gates * pd()->rnn_conf.dic
                * sizeof(float); // G * O * sizeof(float);
        memory_storage_t *temp_buf_ptr;
        engine()->create_memory_storage(&temp_buf_ptr, size);
        scales_buf_.reset(temp_buf_ptr);
        if (!scales_buf_) return status::runtime_error;
    }

    // Prepare testmode scales defined by attributes. Doesn't introduce
    // primitive state, because it is a constant memory -- will not be
    // changed during execution.
    // TODO: add the testmode scales to ws
    if (pd()->rnn_conf.is_testmode && pd_->attr()->rnn_tparams_.scales_) {
        size_t size = pd()->rnn_conf.tm_ngates
                * sizeof(*pd_->attr()->rnn_tparams_.scales_);
        memory_storage_t *temp_buf_ptr;
        engine()->create_memory_storage(&temp_buf_ptr, size);
        tm_scales_buf_.reset(temp_buf_ptr);
        if (!tm_scales_buf_) return status::runtime_error;

        void *tmp_ptr = nullptr;
        status = tm_scales_buf_->map_data(&tmp_ptr);
        if (status != status::success) return status;
        utils::array_copy((float *)tmp_ptr, pd()->attr()->rnn_tparams_.scales_,
                pd()->attr()->rnn_tparams_.ngates_);
        status = tm_scales_buf_->unmap_data(tmp_ptr);
        if (status != status::success) return status;
    }

    bool gemm_ok = true;

    switch (aprop) {
        case prop_kind::forward:
            gemm_ok = true
                    && utils::everyone_is(status::success,
                            pd()->gemm_layer_fwd_pd_->create_primitive(
                                    &gemm_layer_fwd_),
                            pd()->gemm_iter_fwd_pd_->create_primitive(
                                    &gemm_iter_fwd_));
            break;
        case prop_kind::backward:
            gemm_ok = true
                    && utils::everyone_is(status::success,
                            pd()->gemm_layer_bwd_pd_->create_primitive(
                                    &gemm_layer_bwd_),
                            pd()->gemm_iter_bwd_pd_->create_primitive(
                                    &gemm_iter_bwd_),
                            pd()->gemm_diff_wei_layer_pd_->create_primitive(
                                    &gemm_diff_wei_layer_),
                            pd()->gemm_diff_wei_iter_pd_->create_primitive(
                                    &gemm_diff_wei_iter_));
            break;
        default: assert(!"unknown prop_kind"); return status::invalid_arguments;
    }

    if (!gemm_ok) return status::runtime_error;

    return status::success;
}

template <prop_kind_t aprop>
gemm_sig((_ref_rnn_common_t<aprop>::packed_gemm)) {
#if USE_MKL_PACKED_GEMM
// TBD
#else
    UNUSED(a);
    UNUSED(b);
    UNUSED(c);
    UNUSED(gemm_kind);
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop>
gemm_sig((_ref_rnn_common_t<aprop>::gemm_primitive)) {
    using namespace gemm_utils;

    // FIXME: This should be created once per execute() instead of creating
    // memory before each gemm call. Each cell type (+prop kind) might have
    // different number of GEMMs.

    void *scratchpad_ptr {nullptr}, *scratchpad_gates_ptr {nullptr};

    memory_t *workspace = (aprop == prop_kind::forward)
            ? ctx.output(DNNL_ARG_WORKSPACE)
            : ctx.input(DNNL_ARG_WORKSPACE);

    std::unique_ptr<memory_storage_t> scratchpad;
    if (pd()->rnn_conf.use_workspace) {
        workspace->memory_storage()->get_data_handle(&scratchpad_ptr);
    } else {
        scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                key_rnn_space);
        scratchpad->get_data_handle(&scratchpad_ptr);
    }

    auto scratchpad_gates
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_gates);
    scratchpad_gates->get_data_handle(&scratchpad_gates_ptr);

    void *weights_ptr {nullptr};
    memory_t *weights {nullptr};

    // These memory storages provide a mechanism to reuse existing memory
    // storage with an offset. These memory storages don't own attached memory
    memory_storage_t *gemm_A_ = nullptr;
    memory_storage_t *gemm_B_ = nullptr;
    memory_storage_t *gemm_C_ = nullptr;

    engine()->create_memory_storage(&gemm_A_, use_runtime_ptr, 0, nullptr);
    engine()->create_memory_storage(&gemm_B_, use_runtime_ptr, 0, nullptr);
    engine()->create_memory_storage(&gemm_C_, use_runtime_ptr, 0, nullptr);

    switch (gemm_kind) {
        case gemm_iter_fwd:
        case gemm_layer_fwd:
            weights = (gemm_kind == gemm_layer_fwd)
                    ? ctx.input(DNNL_ARG_WEIGHTS_LAYER)
                    : ctx.input(DNNL_ARG_WEIGHTS_ITER);
            weights->memory_storage()->get_data_handle(&weights_ptr);

            gemm_A_->set_data_handle(weights_ptr);
            gemm_B_->set_data_handle(scratchpad_ptr);
            gemm_C_->set_data_handle(scratchpad_gates_ptr);
            break;
        case gemm_iter_bwd:
        case gemm_layer_bwd:
            weights = (gemm_kind == gemm_layer_bwd)
                    ? ctx.input(DNNL_ARG_WEIGHTS_LAYER)
                    : ctx.input(DNNL_ARG_WEIGHTS_ITER);
            weights->memory_storage()->get_data_handle(&weights_ptr);

            gemm_A_->set_data_handle(weights_ptr);
            gemm_B_->set_data_handle(scratchpad_gates_ptr);
            gemm_C_->set_data_handle(scratchpad_ptr);
            break;
        case gemm_diff_wei_iter:
        case gemm_diff_wei_layer:
            weights = (gemm_kind == gemm_diff_wei_iter)
                    ? ctx.output(DNNL_ARG_DIFF_WEIGHTS_ITER)
                    : ctx.output(DNNL_ARG_DIFF_WEIGHTS_LAYER);
            weights->memory_storage()->get_data_handle(&weights_ptr);

            gemm_A_->set_data_handle(scratchpad_gates_ptr);
            gemm_B_->set_data_handle(scratchpad_ptr);
            gemm_C_->set_data_handle(weights_ptr);
            break;
        default: assert(!"unknown gemm_kind");
    }

    gemm_A_->set_offset(off_a);
    gemm_B_->set_offset(off_b);
    gemm_C_->set_offset(off_c);

    gemm_exec_args_t gemm_args;
    gemm_args.a = gemm_A_;
    gemm_args.b = gemm_B_;
    gemm_args.c = gemm_C_;

    auto gemm_ctx = gemm_exec_ctx_t(ctx.stream(), gemm_args);

    switch (gemm_kind) {
        case gemm_iter_fwd: gemm_impl(gemm_iter_fwd_)->execute(gemm_ctx); break;
        case gemm_layer_fwd:
            gemm_impl(gemm_layer_fwd_)->execute(gemm_ctx);
            break;
        case gemm_iter_bwd: gemm_impl(gemm_iter_bwd_)->execute(gemm_ctx); break;
        case gemm_layer_bwd:
            gemm_impl(gemm_layer_bwd_)->execute(gemm_ctx);
            break;
        case gemm_diff_wei_iter:
            gemm_impl(gemm_diff_wei_iter_)->execute(gemm_ctx);
            break;
        case gemm_diff_wei_layer:
            gemm_impl(gemm_diff_wei_layer_)->execute(gemm_ctx);
            break;
        default: assert(!"unknown gemm_kind");
    }

    delete gemm_A_;
    delete gemm_B_;
    delete gemm_C_;
}

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::gates_reduction(const exec_ctx_t &ctx, int dir,
        int lay, int iter, int n_gates, int dic, int batch,
        const memory_storage_t &scratch_gates,
        const memory_storage_t &diff_bias) const {
    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, dir);
    arg_list.set(1, lay);
    arg_list.set(2, iter);
    arg_list.set(3, diff_bias);
    arg_list.set(4, scratch_gates);

    auto nd_range = compute::nd_range_t({n_gates, dic});
    compute_stream->parallel_for(nd_range, gates_reduction_kernel_, arg_list);
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop>
grid_execution_sig((_ref_rnn_common_t<aprop>::linear_execution)) {

    data_type_t src_t = pd()->src_type;
    data_type_t wei_t = pd()->weights_type;

    // We run the grid of computation
    for (int dir = 0; dir < n_dir; dir++) {
        for (int j = 0; j < n_layer; j++) {
            int lay = (aprop == prop_kind::forward) ? j : n_layer - j - 1;

            if (aprop == prop_kind::forward
                    && pd()->rnn_conf.merge_gemm_layer) {
                cl_ulong wei_offset
                        = OFF3(lay, n_layer, dir, n_dir, 0,
                                  pd()->rnn_conf.weights_layer_nld
                                          * pd()->rnn_conf.weights_layer_ld)
                        * types::data_type_size(wei_t);

                cl_ulong offset_input = (cl_ulong)(ws_states_offset_
                        + OFF4(lay, n_layer + 1, dir, n_dir, 1, n_iter + 1, 0,
                                  batch * pd()->rnn_conf.states_ws_ld)
                                * types::data_type_size(src_t));

                gemm_primitive(ctx, w_input, wei_offset, workspace,
                        offset_input, scratch_gates, 0, gemm_layer_fwd);
            }

            for (int i = 0; i < n_iter; i++) {
                int iter = (aprop == prop_kind::forward) ? i : n_iter - i - 1;
                (this->*cell_func)(ctx, dir, lay, iter, dic, slc, sic, wic,
                        batch, n_layer, n_dir, n_iter, n_gates, n_states,
                        n_bias, offset_wei_input_, n_parts_weights_layer,
                        offset_wei_state_, n_parts_weights_iter, bias,
                        workspace, scratch_gates, w_input, w_state,
                        diff_weights_layer, diff_weights_iter, diff_bias,
                        scales, tm_scales);
            }

            if (aprop == prop_kind::backward
                    && pd()->rnn_conf.merge_gemm_layer) {
                AOC<size_t, 3> off_weights_i(
                        weights_input, n_layer, n_dir, n_parts_weights_layer);
                cl_ulong offset_w_input = (cl_ulong)(off_weights_i(lay, dir, 0))
                        * types::data_type_size(wei_t);
                cl_ulong offset_ws_states = ws_states_offset_
                        + OFF4(lay, n_layer + 1, dir, n_dir, 1, n_iter + 1, 0,
                                  batch * pd()->rnn_conf.states_ws_ld)
                                * types::data_type_size(src_t);
                cl_ulong offset_workspace_layer = ws_diff_states_offset_
                        + OFF5(lay, n_layer + 1, dir, n_dir, n_states,
                                  n_states + 1, 0, n_iter + 1, 0,
                                  pd()->rnn_conf.states_nld
                                          * pd()->rnn_conf.diff_states_ws_ld)
                                * sizeof(float);
                cl_ulong offset_weights_layer
                        = OFF3(lay, n_layer, dir, n_dir, 0,
                                  pd()->rnn_conf.diff_weights_layer_nld
                                          * pd()->rnn_conf
                                                    .diff_weights_layer_ld)
                        * sizeof(float);

                gemm_primitive(ctx, w_input, offset_w_input, scratch_gates, 0,
                        workspace, offset_workspace_layer, gemm_layer_bwd);
                gemm_primitive(ctx, scratch_gates, 0, workspace,
                        offset_ws_states, diff_weights_layer,
                        offset_weights_layer, gemm_diff_wei_layer);
            }

            if (aprop == prop_kind::backward
                    && pd()->rnn_conf.merge_gemm_iter) {
                cl_ulong offset_workspace_iter = ws_states_offset_
                        + OFF4(lay + 1, n_layer + 1, dir, n_dir, 0, n_iter + 1,
                                  0, batch * pd()->rnn_conf.states_ws_ld)
                                * types::data_type_size(src_t);
                cl_ulong offset_weights_iter
                        = OFF3(lay, n_layer, dir, n_dir, 0,
                                  pd()->rnn_conf.diff_weights_iter_nld
                                          * pd()->rnn_conf.diff_weights_iter_ld)
                        * sizeof(float);
                gemm_primitive(ctx, scratch_gates, 0, workspace,
                        offset_workspace_iter, diff_weights_iter,
                        offset_weights_iter, gemm_diff_wei_iter);
            }
        }
    }
}

//********* GRID computations strategy: utility functions **********//

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::bias_prepare(
        compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
        int n_bias, int n_gates, int dic, const memory_storage_t &ws,
        const memory_storage_t &scales, const memory_storage_t &wei_layer,
        const memory_storage_t &wei_iter, const memory_storage_t &bias) const {

    float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, ws);
    arg_list.set(1, scales);
    arg_list.set(2, wei_layer);
    arg_list.set(3, wei_iter);
    arg_list.set(4, bias);
    arg_list.set(5, data_shift);
    arg_list.set(6, data_scale);
    compute_stream->parallel_for(
            compute::nd_range_t({dic, n_bias, n_layer * n_dir}),
            bias_prepare_kernel_, arg_list);
}

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::copy_init_layer(
        compute::compute_stream_t *compute_stream, bool lr, bool rl, int n_iter,
        int batch, int slc, const memory_storage_t &ws,
        const memory_storage_t &input,
        const memory_storage_t &diff_dst_layer) const {

    if (aprop == prop_kind::forward) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, input);
        arg_list.set(2, (cl_int)lr);
        arg_list.set(3, (cl_int)rl);
        compute_stream->parallel_for(compute::nd_range_t({slc, batch, n_iter}),
                copy_init_layer_kernel_, arg_list);
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, diff_dst_layer);
        arg_list.set(2, (cl_int)0);
        arg_list.set(3, (cl_int)0);
        compute_stream->parallel_for(compute::nd_range_t({batch, n_iter}),
                copy_init_layer_kernel_, arg_list);
    }
}

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::copy_init_iter(
        compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
        int batch, int sic, int dic, const memory_storage_t &ws,
        const memory_storage_t &firstit_states,
        const memory_storage_t &firstit_c_states,
        const memory_storage_t &diff_dst_iter,
        const memory_storage_t &diff_dst_iter_c, const float shift,
        const float scale, const bool quantize) const {

    if (aprop == prop_kind::forward) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, firstit_states);
        arg_list.set(2, firstit_c_states);
        arg_list.set(3, shift);
        arg_list.set(4, scale);
        arg_list.set(5, (int)quantize);
        compute_stream->parallel_for(
                compute::nd_range_t({sic, batch, n_layer * n_dir}),
                copy_init_iter_kernel_, arg_list);
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, diff_dst_iter);
        arg_list.set(2, diff_dst_iter_c);
        compute_stream->parallel_for(
                compute::nd_range_t({dic, batch, n_layer * n_dir}),
                copy_init_iter_kernel_, arg_list);
    }
}

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::copy_res_layer(
        compute::compute_stream_t *compute_stream, bool lr, bool rl, int n_iter,
        int batch, int slc, int dic, const memory_storage_t &dst_last_layer,
        const memory_storage_t &diff_src_layer, const memory_storage_t &ws,
        const float shift, const float scale, const bool dequantize) const {

    if (aprop == prop_kind::forward) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, dst_last_layer);
        arg_list.set(2, (cl_int)lr);
        arg_list.set(3, (cl_int)rl);
        arg_list.set(4, shift);
        arg_list.set(5, scale);
        arg_list.set(6, (int)dequantize);
        compute_stream->parallel_for(compute::nd_range_t({dic, batch, n_iter}),
                copy_res_layer_kernel_, arg_list);
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, diff_src_layer);
        arg_list.set(2, (cl_int)lr);
        arg_list.set(3, (cl_int)rl);
        compute_stream->parallel_for(compute::nd_range_t({slc, batch, n_iter}),
                copy_res_layer_kernel_, arg_list);
    }
}

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::copy_res_iter(
        compute::compute_stream_t *compute_stream, int n_layer, int n_dir,
        int batch, int sic, int dic, const memory_storage_t &dst_last_iter,
        const memory_storage_t &dst_last_iter_c,
        const memory_storage_t &diff_src_iter,
        const memory_storage_t &diff_src_iter_c, const memory_storage_t &ws,
        const float shift, const float scale, const bool dequantize) const {

    if (aprop == prop_kind::forward) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, dst_last_iter);
        arg_list.set(2, dst_last_iter_c);
        arg_list.set(3, shift);
        arg_list.set(4, scale);
        arg_list.set(5, (int)dequantize);
        compute_stream->parallel_for(
                compute::nd_range_t({dic, batch, n_layer * n_dir}),
                copy_res_iter_kernel_, arg_list);
    } else {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, ws);
        arg_list.set(1, diff_src_iter);
        arg_list.set(2, diff_src_iter_c);
        compute_stream->parallel_for(
                compute::nd_range_t({sic, batch, n_layer * n_dir}),
                copy_res_iter_kernel_, arg_list);
    }
}

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::ws_set(compute::compute_stream_t *compute_stream,
        const memory_storage_t &workspace_, const cl_ulong ws_offset,
        const int ws_part, const float val, const size_t size) const {

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, workspace_);
    arg_list.set(1, ws_offset);
    arg_list.set(2, val);
    arg_list.set(3, ws_part);
    auto nd_range = compute::nd_range_t({size});
    compute_stream->parallel_for(nd_range, ws_set_kernel_, arg_list);
}

#if DEBUGPRINT
template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::ws_print(
        compute::compute_stream_t *compute_stream,
        const memory_storage_t &workspace_) const {

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, workspace_);
    auto nd_range = compute::nd_range_t({1});
    compute_stream->parallel_for(nd_range, ws_print_kernel_, arg_list);
}
#endif

template <prop_kind_t aprop>
packing_sig((_ref_rnn_common_t<aprop>::pack_weights)) {
#if USE_MKL_PACKED_GEMM
// TBD
#else
    UNUSED(n_layer);
    UNUSED(n_dir);
    UNUSED(n_weights);
    UNUSED(n_gates);
    UNUSED(n_parts);
    UNUSED(gates_per_part);
    UNUSED(batch);
    UNUSED(OC_size);
    UNUSED(IC_size);
    UNUSED(weights_);
    UNUSED(w_);
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop>
packing_sig((_ref_rnn_common_t<aprop>::no_pack_weights)) {
    AOC<size_t, 3> weights(weights_, n_layer, n_dir, n_parts);

    for (int i = 0; i < n_layer; i++) {
        for (int d = 0; d < n_dir; d++) {
            weights(i, d, 0) = OFF5(
                    i, n_layer, d, n_dir, 0, OC_size, 0, n_gates, 0, IC_size);
            for (int p = 1; p < n_parts; p++) {
                weights(i, d, p) = OFF5(i, n_layer, d, n_dir, 0, OC_size,
                        gates_per_part[p - 1], n_gates, 0, IC_size);
            }
        }
    }
}
template <prop_kind_t aprop>
free_packed_sig((_ref_rnn_common_t<aprop>::free_packed_weights)) {
#if USE_MKL_PACKED_GEMM
// TBD
#else
    UNUSED(n_layer);
    UNUSED(n_dir);
    UNUSED(n_parts);
    UNUSED(weights_);
    assert(!"packed gemm is disabled");
#endif
}
template <prop_kind_t aprop>
free_packed_sig((_ref_rnn_common_t<aprop>::free_no_packed_weights)) {
    UNUSED(n_layer);
    UNUSED(n_dir);
    UNUSED(n_parts);
    UNUSED(weights_);
}

//********************* Execution function *********************//

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::execute_(const exec_ctx_t &ctx) const {

    status_t status = status::success;

    auto *compute_stream
            = utils::downcast<compute::compute_stream_t *>(ctx.stream());

    auto rnn_pd = this->pd();
    const conf_t &rnn = this->pd()->rnn_conf;

    int n_layer = rnn.n_layer;

    int n_dir = rnn.n_dir;
    int n_iter = rnn.n_iter;
    int n_gates = rnn.n_gates;
    int n_bias = rnn.n_bias;
    int n_states = rnn.n_states;
    int n_weights_input = rnn_pd->SLC();
    int n_weights_state = rnn_pd->SIC();
    int batch = rnn.mb;
    int slc = rnn.slc;
    int sic = rnn.sic;
    int dic = rnn.dic;
    int dlc = rnn.dlc;
    int wic = nstl::max(slc, nstl::max(sic, dic));

    bool is_orig_gru = rnn_pd->cell_kind() == alg_kind::vanilla_gru;
    int n_parts_weights_iter = rnn.n_parts_weights_iter;
    int n_parts_weights_layer = rnn.n_parts_weights_layer;

    bool is_fwd = rnn.is_fwd;

    auto &input_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_LAYER);
    auto &states_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER);
    auto &c_states_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER_C);
    auto &w_input_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_LAYER);
    auto &w_state_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_ITER);
    auto &bias_native_ = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &dst_last_layer_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_LAYER)
                                          : CTX_IN_STORAGE(DNNL_ARG_DST_LAYER);
    auto &dst_last_iter_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_ITER)
                                         : CTX_IN_STORAGE(DNNL_ARG_DST_ITER);
    auto &dst_last_iter_c_native_ = is_fwd
            ? CTX_OUT_STORAGE(DNNL_ARG_DST_ITER_C)
            : CTX_IN_STORAGE(DNNL_ARG_DST_ITER_C);

    auto &diff_dst_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_LAYER);
    auto &diff_dst_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_ITER);
    auto &diff_dst_iter_c_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_ITER_C);

    auto scratchpad
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_space);
    auto &workspace_ = rnn.is_training ? is_fwd
                    ? CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE)
                    : CTX_IN_STORAGE(DNNL_ARG_WORKSPACE)
                                       : *scratchpad.get();

    auto scratchpad_gates
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_gates);
    auto &scratch_gates = *scratchpad_gates.get();

    auto &diff_src_layer_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_LAYER);
    auto &diff_src_iter_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_ITER);
    auto &diff_src_iter_c_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_ITER_C);

    auto &diff_weights_layer_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_LAYER);
    auto &diff_weights_iter_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_ITER);
    auto &diff_bias_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    auto prints = [=](void) {
        DPRINT("\n%s\n", "+++++++++++++++");
        DPRINT(" aprop = %d\n", (int)aprop);
        DPRINT("%s\n", "+++++++++++++++");
        DPRINT("  n_layer         = %d\n", n_layer);
        DPRINT("  n_dir           = %d\n", n_dir);
        DPRINT("  n_iter          = %d\n", n_iter);
        DPRINT("  n_gates         = %d\n", n_gates);
        DPRINT("  n_bias          = %d\n", n_bias);
        DPRINT("  n_states        = %d\n", n_states);
        DPRINT("  n_weights_input = %d\n", n_weights_input);
        DPRINT("  n_weights_state = %d\n", n_weights_state);
        DPRINT("  batch           = %d\n", batch);
        DPRINT("  slc             = %d\n", slc);
        DPRINT("  sic             = %d\n", sic);
        DPRINT("  dic             = %d\n", dic);
        DPRINT("  dlc             = %d\n", dlc);
        DPRINT("  wic             = %d\n", wic);
        DPRINT("%s\n", "+++++++++++++++");
        DPRINT("  is_fwd          = %s\n", is_fwd ? "yes" : "no");
        DPRINT("  is_orig_gru     = %s\n", is_orig_gru ? "yes" : "no");
        DPRINT("  use_workspace   = %s\n", rnn.use_workspace ? "yes" : "no");
        DPRINT("%s\n", "+++++++++++++++");
        DPRINT("  with_src_iter   = %s\n",
                rnn_pd->with_src_iter() ? "yes" : "no");
        DPRINT("  with_src_iter_c = %s\n",
                rnn_pd->with_src_iter_c() ? "yes" : "no");
        DPRINT("  with_bias       = %s\n", rnn_pd->with_bias() ? "yes" : "no");
        DPRINT("  with_dst_iter   = %s\n",
                rnn_pd->with_dst_iter() ? "yes" : "no");
        DPRINT("  with_dst_iter_c = %s\n",
                rnn_pd->with_dst_iter_c() ? "yes" : "no");
        DPRINT("%s\n", "+++++++++++++++");
    };

#if DEBUGPRINT
    prints();
#else
    UNUSED(dlc);
    UNUSED(is_orig_gru);
    UNUSED(prints);
#endif

#if WS_NAN_FILLING
    if (rnn.is_fwd) {
        DPRINT("DEBUG ws NaN filling: (offset, size) states: %ld %ld c_states: "
               "%ld %ld gates: %ld %ld\n",
                ws_states_offset_, rnn.ws_states_size, ws_c_states_offset_,
                rnn.ws_c_states_size, ws_gates_offset_, rnn.ws_gates_size);

        ws_set(compute_stream, workspace_, ws_states_offset_, rnn_utils::states,
                NAN, rnn.ws_states_size / rnn.ws_states_elsz);
        if (rnn_pd->with_src_iter_c()) {
            DPRINT("rnn.ws_c_states_elsz = %d\n", rnn.ws_c_states_elsz);
            ws_set(compute_stream, workspace_, ws_c_states_offset_,
                    rnn_utils::c_states, NAN,
                    rnn.ws_c_states_size / rnn.ws_c_states_elsz);
        }
        ws_set(compute_stream, workspace_, ws_gates_offset_, rnn_utils::gates,
                NAN, rnn.ws_gates_size / rnn.ws_gates_elsz);
        ws_set(compute_stream, workspace_, ws_bias_offset_, rnn_utils::bias,
                NAN, rnn.ws_bias_size / rnn.ws_bias_elsz);
    }
#endif

    // initialize diff_state to 0
    if (aprop == prop_kind::backward) {
        ws_set(compute_stream, workspace_, ws_diff_states_offset_,
                rnn_utils::diff_states, 0.0f, rnn.ws_diff_states_size);
    }

    DPRINT("\n%s(%d) WS before bias prepare\n\n", __FUNCTION__, __LINE__);
    WS_PRINT(compute_stream, workspace_);

    // TODO: implement without copies
    bool is_lr = !one_of(rnn.exec_dir, r2l, r2l);
    bool is_rl = !one_of(rnn.exec_dir, l2r, l2r);

    // copy bias to memory storage
    if (rnn.copy_bias && scales_buf_) {
        void *tmp_ptr = nullptr;
        status = scales_buf_->map_data(&tmp_ptr);
        if (status != status::success) return status;
        utils::array_copy((float *)tmp_ptr,
                pd()->attr()->rnn_weights_qparams_.scales_,
                rnn.n_gates * rnn.dic);
        status = scales_buf_->unmap_data(tmp_ptr);
        if (status != status::success) return status;
    }

    // XXX: this function is used for calculating offsets for buffers and not
    // used for packing weights
    (this->*weights_state_pack_func)(n_layer, n_dir, n_weights_state, n_gates,
            batch, dic, sic, offset_wei_state_, n_parts_weights_iter,
            rnn.parts_weights_iter, w_state_native_);
    (this->*weights_input_pack_func)(n_layer, n_dir, n_weights_input, n_gates,
            batch, dic, slc, offset_wei_input_, n_parts_weights_layer,
            rnn.parts_weights_layer, w_input_native_);

    // bias prepare if needed
    if (rnn.copy_bias) {
        bias_prepare(compute_stream, n_layer, n_dir, n_bias, n_gates, dic,
                workspace_, *scales_buf_, w_input_native_, w_state_native_,
                bias_native_);
    }
    DPRINT("\n%s(%d) WS before copy init\n\n", __FUNCTION__, __LINE__);
    WS_PRINT(compute_stream, workspace_);

    float shift = (pd()->attr()->rnn_data_qparams_.shift_);
    float scale = (pd()->attr()->rnn_data_qparams_.scale_);

    // we first need to copy the initial states and input into ws
    copy_init_layer(compute_stream, is_lr, is_rl, n_iter, batch, slc,
            workspace_, input_native_, diff_dst_layer_native_);
    const bool quantize = pd()->with_src_iter()
            && pd()->src_md(1)->data_type == data_type::f32 && rnn.is_int8;
    copy_init_iter(compute_stream, n_layer, n_dir, batch, sic, dic, workspace_,
            states_native_, c_states_native_, diff_dst_iter_native_,
            diff_dst_iter_c_native_, shift, scale, quantize);

    DPRINT("\n%s(%d) WS before grid\n\n", __FUNCTION__, __LINE__);
    WS_PRINT(compute_stream, workspace_);

    // run the execution on the grid
    (this->*grid_computation)(ctx, dic, slc, sic, wic, batch, n_layer, n_dir,
            n_iter, n_gates, n_states, n_bias, offset_wei_input_,
            n_parts_weights_layer, offset_wei_state_, n_parts_weights_iter,
            bias_native_, workspace_, scratch_gates, w_input_native_,
            w_state_native_, diff_weights_layer_native_,
            diff_weights_iter_native_, diff_bias_native_, scales_buf_.get(),
            tm_scales_buf_.get());

    DPRINT("\n%s(%d) WS before copy res\n\n", __FUNCTION__, __LINE__);
    WS_PRINT(compute_stream, workspace_);

    // Finally we copy the results to the result buffers

    const bool dequantize_l
            = pd()->dst_md(0)->data_type == data_type::f32 && rnn.is_int8;
    copy_res_layer(compute_stream, is_lr, is_rl, n_iter, batch, slc, dic,
            dst_last_layer_native_, diff_src_layer_native_, workspace_, shift,
            scale, dequantize_l);
    const bool dequantize_i = pd()->with_dst_iter()
            && pd()->dst_md(1)->data_type == data_type::f32 && rnn.is_int8;
    copy_res_iter(compute_stream, n_layer, n_dir, batch, sic, dic,
            dst_last_iter_native_, dst_last_iter_c_native_,
            diff_src_iter_native_, diff_src_iter_c_native_, workspace_, shift,
            scale, dequantize_i);

    // NOT USED YET
#if 0
    // We free the packed weights if they were packed internally
    // currently - not
    (this->*weights_state_free_packed_func)(n_layer, n_dir,
            n_parts_weights_iter, offset_wei_state_);
    (this->*weights_input_free_packed_func)(n_layer, n_dir,
            n_parts_weights_layer, offset_wei_input_);
#endif

    return status::success;
};

// Fix for MSVS warning C4661.
template <>
cell_execution_sig(ref_rnn_fwd_t::cell_execution);
template <>
cell_execution_sig(ref_rnn_bwd_t::cell_execution);
template <>
cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru);
template <>
cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru);
template <>
cell_execution_sig(ref_rnn_fwd_t::cell_execution_gru_lbr);
template <>
cell_execution_sig(ref_rnn_bwd_t::cell_execution_gru_lbr);
template <>
elemwise_sig(ref_rnn_fwd_t::rnn_elemwise);
template <>
elemwise_sig(ref_rnn_bwd_t::rnn_elemwise);
template <>
elemwise_sig(ref_rnn_fwd_t::lstm_elemwise);
template <>
elemwise_sig(ref_rnn_bwd_t::lstm_elemwise);
template <>
elemwise_sig(ref_rnn_fwd_t::lstm_elemwise_u8s8);
template <>
elemwise_sig(ref_rnn_bwd_t::lstm_elemwise_u8s8);
template <>
elemwise_sig(ref_rnn_fwd_t::gru_lbr_elemwise);
template <>
elemwise_sig(ref_rnn_bwd_t::gru_lbr_elemwise);

template struct _ref_rnn_common_t<prop_kind::forward>;
template struct _ref_rnn_common_t<prop_kind::backward>;

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
