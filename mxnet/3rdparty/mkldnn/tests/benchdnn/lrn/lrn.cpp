/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "lrn/lrn.hpp"

namespace lrn {

int compare(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r) {
    const auto nelems = mem_dt.nelems();

    r->errors = 0;
    r->total = nelems;
    const int summands = compute_n_summands(p);
    float trh = 1e-6 * summands;
    if (p->dt == dnnl_f16) trh = 1e-3 * summands;
    if (p->dt == dnnl_bf16) trh = 1e-2 * summands;

    for (int64_t i = 0; i < nelems; ++i) {
        const float dt = mem_dt.get_elem(i);
        const float fp0 = mem_fp.get_elem(i);
        const float fp = maybe_saturate(p->dt, fp0);

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
        const bool ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= trh;

        r->errors += !ok;

        const bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                || (verbose >= 50 && i < 30) || (verbose >= 99);
        if (dump) {
            int64_t mb = 0, ic = 0, d = 0, h = 0, w = 0;
            inv_data_off(p, i, mb, ic, d, h, w);

            print(0,
                    "[%4ld][" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                    "] "
                    "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                    (long)i, mb, ic, d, h, w, fp, fp0, dt, diff, rel_diff);
        }
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int fill_dat(const prb_t *p, data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp) {
    const auto nelems = mem_fp.nelems();
    const int range = 16;
    const int f_min = p->dt == dnnl_u8 ? 0 : -range / 2;

    dnnl::impl::parallel_nd(nelems, [&](int64_t i) {
        const int64_t gen = kind == SRC ? 1091 * i + 1637 : 1279 * i + 1009;
        const float value = f_min + gen % range;
        mem_fp.set_elem(i, value);
    });

    SAFE(mem_dt.reorder(mem_fp), WARN);

    return OK;
}

int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(p, SRC, mem_dt, mem_fp);
}

int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp) {
    return fill_dat(p, DST, mem_dt, mem_fp);
}

int init_pd(const prb_t *p, dir_t dir, dnnl_primitive_desc_t &lpd,
        const_dnnl_primitive_desc_t hint, res_t *r) {
    dnnl_lrn_desc_t ld;
    dnnl_memory_desc_t data_d;

    dnnl_dims_t data_dims_0d = {p->mb, p->ic};
    dnnl_dims_t data_dims_1d = {p->mb, p->ic, p->iw};
    dnnl_dims_t data_dims_2d = {p->mb, p->ic, p->ih, p->iw};
    dnnl_dims_t data_dims_3d = {p->mb, p->ic, p->id, p->ih, p->iw};

    dnnl_dim_t *data_dims = p->ndims == 5
            ? data_dims_3d
            : p->ndims == 4 ? data_dims_2d
                            : p->ndims == 3 ? data_dims_1d : data_dims_0d;

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&data_d, p->ndims, data_dims, p->dt,
                     convert_tag(p->tag, p->ndims)),
            WARN);

    dnnl_alg_kind_t alg = alg2alg_kind(p->alg);
    if (dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF ? dnnl_forward_inference
                                      : dnnl_forward_training;
        DNN_SAFE(dnnl_lrn_forward_desc_init(&ld, prop, alg, &data_d, p->ls,
                         p->alpha, p->beta, p->k),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, p->ndims, data_dims,
                         p->dt, dnnl_format_tag_any),
                WARN);
        DNN_SAFE(dnnl_lrn_backward_desc_init(&ld, alg, &diff_data_d, &data_d,
                         p->ls, p->alpha, p->beta, p->k),
                WARN);
    }

    auto dnnl_attr = create_dnnl_attr(attr_t());

    dnnl_status_t init_status = dnnl_primitive_desc_create(
            &lpd, &ld, dnnl_attr, engine_tgt, hint);

    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    // Return if pd is not the one being tested
    if ((dir & FLAG_FWD) != (p->dir & FLAG_FWD)) return OK;

    const char *impl_str = query_impl_info(lpd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: dnnl implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(lpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "dnnl implementation: %s\n", impl_str);
    }

    return OK;
}

int init_pd_fwd(const prb_t *p, dnnl_primitive_desc_t &lpd, res_t *r) {
    return init_pd(p, FLAG_FWD, lpd, nullptr, r);
}

int init_pd_bwd(const prb_t *p, dnnl_primitive_desc_t &lpd,
        const_dnnl_primitive_desc_t hint, res_t *r) {
    return init_pd(p, FLAG_BWD, lpd, hint, r);
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    dnnl_primitive_desc_t lpd;
    SAFE(init_pd_fwd(p, lpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    dnnl_primitive_t l;
    DNN_SAFE(dnnl_primitive_create(&l, lpd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(lpd), CRIT);

    const_dnnl_primitive_desc_t const_fpd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(l, &const_fpd), CRIT);

    if (dnn_mem_t::check_mem_size(const_fpd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(l));
        return r->state = SKIPPED, OK;
    }

    const auto q = [](const_dnnl_primitive_desc_t pd,
                           int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md = q(const_fpd, DNNL_ARG_SRC);
    const auto &ws_md = q(const_fpd, DNNL_ARG_WORKSPACE);
    const auto &scratchpad_md = q(const_fpd, DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = get_abx_tag(p->ndims);

    dnn_mem_t src_fp(data_md, fp, tag, engine_tgt);
    dnn_mem_t src_dt(data_md, engine_tgt);

    dnn_mem_t dst_fp(data_md, fp, tag, engine_tgt);
    dnn_mem_t dst_dt(data_md, engine_tgt);

    if (p->dir & FLAG_INF) SAFE(ws_md.ndims == 0 ? OK : FAIL, WARN);
    dnn_mem_t ws_fp(ws_md, engine_tgt);
    dnn_mem_t ws_dt(ws_md, engine_tgt);
    dnn_mem_t scratchpad_dt(scratchpad_md, engine_tgt);

    dnn_mem_t d_dst_dt, d_src_dt;

    SAFE(fill_src(p, src_dt, src_fp), WARN);

    args_t args_fwd, args_bwd;
    args_fwd.set(DNNL_ARG_SRC, src_dt);
    args_fwd.set(DNNL_ARG_DST, dst_dt);
    args_fwd.set(DNNL_ARG_WORKSPACE, ws_dt);
    args_fwd.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
    args_t &args = args_fwd;

    DNN_SAFE(execute_and_wait(l, stream_tgt, args), WARN);

    if (p->dir & FLAG_FWD) {
        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, dst_fp);
            dnn_mem_t dst(dst_dt, fp, tag, engine_tgt);
            SAFE(compare(p, dst, dst_fp, r), WARN);
        }
    }

    if (p->dir & FLAG_BWD) {
        SAFE(init_pd_bwd(p, lpd, const_fpd, r), WARN);
        DNN_SAFE(dnnl_primitive_destroy(l), CRIT);
        if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

        DNN_SAFE(dnnl_primitive_create(&l, lpd), WARN);
        DNN_SAFE(dnnl_primitive_desc_destroy(lpd), CRIT);

        const_dnnl_primitive_desc_t const_bpd;
        DNN_SAFE(dnnl_primitive_get_primitive_desc(l, &const_bpd), CRIT);

        if (dnn_mem_t::check_mem_size(const_bpd) != OK) {
            DNN_SAFE_V(dnnl_primitive_destroy(l));
            return r->state = SKIPPED, OK;
        }

        const auto &d_data_md = q(const_bpd, DNNL_ARG_DIFF_DST);
        const auto &d_scratchpad_md = q(const_bpd, DNNL_ARG_SCRATCHPAD);

        dnn_mem_t d_dst_fp(d_data_md, fp, tag, engine_tgt);
        d_dst_dt = dnn_mem_t(d_data_md, engine_tgt);

        dnn_mem_t d_src_fp(d_data_md, fp, tag, engine_tgt);
        d_src_dt = dnn_mem_t(d_data_md, engine_tgt);

        scratchpad_dt = dnn_mem_t(d_scratchpad_md, engine_tgt);

        SAFE(fill_dst(p, d_dst_dt, d_dst_fp), WARN);

        args_bwd.set(DNNL_ARG_SRC, src_dt);
        args_bwd.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args_bwd.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args_bwd.set(DNNL_ARG_WORKSPACE, ws_dt);
        args_bwd.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);
        args = args_bwd;

        DNN_SAFE(execute_and_wait(l, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_fp, d_dst_fp, d_src_fp);
            dnn_mem_t d_src(d_src_dt, fp, tag, engine_tgt);
            SAFE(compare(p, d_src, d_src_fp, r), WARN);
        }
    }

    measure_perf(r->timer, l, args);

    DNN_SAFE_V(dnnl_primitive_destroy(l));

    return OK;
}

} // namespace lrn
