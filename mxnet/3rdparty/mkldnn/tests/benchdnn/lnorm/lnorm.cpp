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

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "dnnl.h"

#include "src/common/dnnl_thread.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "norm.hpp"

#include "bnorm/bnorm.hpp"
#include "lnorm/lnorm.hpp"

using namespace bnorm;

namespace lnorm {

static int prepare_fwd(const prb_t *p, dnn_mem_t &src, dnn_mem_t &mean,
        dnn_mem_t &var, dnn_mem_t &ss) {
    /** Idea: choose src[] values so that both mean and variance are computed
     * exactly (independently of the order of the computations).
     *
     * The `exactness` is achieved via [a1]: src[i] + src[i+1] = 2 * mean.
     *
     * The variation in src is allowed in the last flex_bits bits.
     * If the sequence (L) is too big (flex_bits <= min_flex_bits), the mean
     * value is set to 0 and src is partially filled with zeros (according to
     * density so that at least want_flex_bits is reserved for src variation.
     * Once src is set, variance is computed.
     *
     * ALG_0: mean is set to 0
     * ALG_1: mean is set to 2^p, where p \in {-2, -1, ..., 4}
     * ALG_AUTO: choose between ALG_0 and ALG_1 automatically */
    const int64_t exact_bits = digits_dt(p->dt);
    const int64_t L = p->c;
    const int64_t logL = (int64_t)ceilf(log2f(L));

    assert(logL <= 0 || (1LL << (logL - 1)) < L);
    assert(L <= (1LL << logL));

    const int64_t min_flex_bits = 3;
    const int64_t want_flex_bits = MIN2(6, exact_bits / 2);

    check_alg_t alg = p->check_alg;
    if (alg == ALG_AUTO) /* choose appropriate checking algorithm */
        alg = (exact_bits - logL) / 2 - 1 >= min_flex_bits ? ALG_1 : ALG_0;

    const int64_t flex_bits = alg == ALG_0
            ? want_flex_bits /* BFloat16 has only 7 bits of mantissa */
            : MIN2(p->dt == dnnl_bf16 ? 7 : exact_bits,
                    (exact_bits - logL) / 2 - 1);

    if (flex_bits < min_flex_bits) return FAIL;

    const int64_t flex_mask = (1 << flex_bits) - 1;

    /* density: (exact_bits - log_2(L * density)) / 2 >= flex_bits */
    const float density = alg == ALG_0
            ? 1.f * (1 << (exact_bits - 2 * flex_bits)) / L
            : 1.f;
    assert((exact_bits - ceilf(log2f(L * density))) / 2 >= flex_bits);

    print(6, "check_alg: %s, density = %g, flex_bits = " IFMT "\n",
            check_alg2str(alg), density, flex_bits);

    dnnl::impl::parallel_nd(p->n, [&](int64_t n) {
        const float m = alg == ALG_0 ? 0.f : 0.25f * (1 << (n % 7));
        float v = 0; /* current variance */

        float *s = (float *)src + n * p->c;
        for (int64_t c = 0; c < p->c; ++c) {
            const int64_t l = c + n * 239 * 2; // l[0] must be even

            if (alg == ALG_0 && !flip_coin(l / 2 * 257ULL, density)) {
                s[c] = 0;
                continue;
            }

            const int64_t gen = (l / 2 * 1637) & flex_mask;
            const int sgn = l % 2 == 0 ? 1 : -1; /* [a1] */
            const float f = 1.f * sgn * gen / (1 << flex_bits);

            src.set_elem(n * p->c + c, alg == ALG_0 ? f : m * (1.f + f));
            if (L % 2 && (c == L - 1)) { s[c] = m; }
            v += (s[c] - m) * (s[c] - m);
        }
        mean.set_elem(n, m);
        var.set_elem(n, v / p->c);
    });

    dnnl::impl::parallel_nd(p->c, [&](int64_t c) {
        if (p->flags & USE_SCALESHIFT) {
            ((float *)ss)[c] = 1.f / 8 * (1 << (c % 7));
            ((float *)ss)[p->c + c] = ((c % 3) - 1) * ((float *)ss)[c] / 64;
        } else {
            ((float *)ss)[c] = 1;
            ((float *)ss)[p->c + c] = 0;
        }
    });
    return OK;
}
/** @brief L = 2^k * P, P % 2 != 0 */
static void decompose2(int64_t L, int64_t &k, int64_t &P) {
    P = L;
    for (k = 0; P % 2 == 0; ++k)
        P /= 2;
}
static int prepare_bwd(const prb_t *p, dnn_mem_t &src, dnn_mem_t &d_dst,
        dnn_mem_t &mean, dnn_mem_t &var, dnn_mem_t &ss) {
    const int64_t exact_bits = 24;

    if (p->c < 2) return FAIL;

    const int64_t L = p->c;
    /** Stabilization idea...
     * Layer Normalization (unlike batch normalization) features two types of
     * accumulations in bwd step:
     * First, accumulation over n:
     *      d_gamma[c] = sum_over_n ddst[n, c] * (src[n, c] - mean[n]) * inv_sigma
     *      d_beta[c] = ...
     * Second, accumulation over c:
     *      dd_gamma[n] = sum_over_c ddst[n, c] * (src[n, c] - mean[n])
     *          * inv_sigma * gamma
     *      dd_gamma_x[n] = ...
     * that is used when computing d_src:
     *      d_src = func(dd_gamma / C, dd_gamma_x / C, ...)
     * To avoid accumulation error in the first case we will force sparsity
     * of ddst over n if d_gamma and d_beta need to be computed.
     * To get exact result of division in the second case we use the same
     * approach as in batch normalization:
     * Try to make dd_gamma = L / 2^t_dd_gamma and dd_gamma_x = L / 2^t_dd_gamma_x,
     * where both t_dd_gamma and t_dd_gamma_x are in {1, .., max_k}.
     * Currently, with no obvious reason, max_k is set to 4 for
     * reasonably small problems and to 8 for big problems.
     *
     * We might hope that division by L would be exact in that case,
     * but that might happen iff L is less than 2^exact_bits, hence
     * restriction [r1].
     * */

    int64_t k, P;
    decompose2(L, k, P);

    int64_t log2P = (int64_t)ceilf(log2f(P));
    if (log2P >= exact_bits) return FAIL; /* [r1] */

    const int64_t max_k = 4;
    if (k > max_k && exact_bits - log2P > max_k + 4) {
        log2P += (k - max_k);
        P <<= k - max_k;
        k = max_k;
    }

    const int64_t param_dd_p2 = 7; // factor_dd <- 2^{0, .., -param_dd_p2+1}
    const int64_t param_dd_gen = 32; // gen_dd <- {1, .., param_dd_gen}

    const int64_t param_f_p2 = 1; // factor_f <- 2^{-1, ..., -param_f_p2}
    const int64_t param_f_gen = 16; // gen_f <- {2, ..., param_s_gen}

    const float density
            = p->flags & USE_SCALESHIFT ? MIN2(1.f, 10.f / p->n) : 1.f;

    print(5, "prep_bwd: k:" IFMT ", P:" IFMT " log2P:" IFMT ", density = %g\n",
            k, P, log2P, density);

    // fill gamma and beta
    for (int64_t c = 0; c < p->c; ++c) {
        if (p->flags & USE_SCALESHIFT) {
            ((float *)ss)[c] = 1.f / 2 * (1 << (c % 7));
            ((float *)ss)[p->c + c] = ((float *)ss)[c] / 64;
        } else {
            ((float *)ss)[c] = 1;
            ((float *)ss)[p->c + c] = 0;
        }
    }

    for (int64_t n = 0; n < p->n; ++n) {
        const float m = ((float *)mean)[n] = n % 2;

        /* var + eps \in {1/4, 1, 4} */
        const float ve_denom = 4.f / (1 << 2 * (n % 3));
        ((float *)var)[n] = ve_denom - p->eps;

        const int64_t dd_p2 = (n * 127 % param_dd_p2);
        const float factor_dd = 1.f / (1 << dd_p2);
        const int64_t f_p2 = 1 + (n % param_f_p2);
        const float factor_f = 1.f / (1 << f_p2);

        const float target_dd_g = factor_dd * P;
        const float target_dd_g_x = 2 * target_dd_g;

        if (!flip_coin(n, density) && n != 0 && n != p->n - 1) {
            for (int64_t c = 0; c < p->c; ++c) {
                ((float *)d_dst)[n * p->c + c] = 0;
                ((float *)src)[n * p->c + c] = m;
            }
            continue;
        }
        float dd_g = 0, dd_g_x = 0; /* current dd_gamma and dd_gamma_x */
        for (int64_t c = 0; c < p->c - 2; ++c) {
            const float g = ((float *)ss)[c];
            float &s = ((float *)src)[n * p->c + c];
            float &dd = ((float *)d_dst)[n * p->c + c];

            const int sgn_dd = dd_g < target_dd_g ? 1 : -1;
            dd = sgn_dd * factor_dd * (1 + ((c + n) * 3 % param_dd_gen));
            dd_g += dd * g;

            const int sgn_f = dd_g_x < target_dd_g_x ? 1 : -1;
            const float f = sgn_f * factor_f
                    * (2 + ((c + n) * 7 % (param_f_gen - 1)));

            dd_g_x += f * dd * g;
            s = f + m;
        }

        /* the last 2 elements in src and d_dst are set, so that:
         *      dd_gamma == target_dd_gamma
         *      dd_gamma_x == target_dd_gamma_x
         * For this we need to solve the system:
         *      d_dst[l1] * g[c1]           + d_dst[l0] * g[c0]
         *          = target_dd_gamma - dd_gamma
         *      d_dst[l1] * src[l1] * g[c1] + d_dst[l0] * src[l0] * g[c0]
         *          = target_dd_gamam_x - dd_gamma_x
         *
         * Here l0 -- last index, l1 -- last but one.
         * More over, let's assume src[l1] = 1 and src[l0] = -1. */
        int64_t l0 = n * p->c + p->c - 1;
        int64_t l1 = n * p->c + p->c - 2;

        ((float *)src)[l1] = 1.f + m;
        ((float *)src)[l0] = -1.f + m;
        const float g1 = ((float *)ss)[p->c - 2];
        const float g0 = ((float *)ss)[p->c - 1];

        float f1 = ((target_dd_g - dd_g) + (target_dd_g_x - dd_g_x)) / 2;
        float f0 = ((target_dd_g - dd_g) - (target_dd_g_x - dd_g_x)) / 2;

        ((float *)d_dst)[l1] = f1 / g1;
        ((float *)d_dst)[l0] = f0 / g0;

        if (p->dt == dnnl_bf16) { // truncate to bf16
            ((uint16_t *)(&((float *)d_dst)[l1]))[0] = 0;
            ((uint16_t *)(&((float *)d_dst)[l0]))[0] = 0;
        }
    }

    return OK;
}

static int compare(const prb_t *p, data_kind_t kind, const dnn_mem_t &fp_mem,
        const dnn_mem_t &dt_mem, res_t *r, const dnn_mem_t *ss = nullptr) {
    const char *skind = data_kind2str(kind);
    const int f32_mant_digits = 24;
    const float eps_coeff = (1 << (f32_mant_digits - digits_dt(p->dt)));
    const float eps = eps_coeff
            * (p->dir & FLAG_FWD ? (kind == DATA ? 5e-7 : 0)
                                 : (kind == DATA || kind == SS ? 2e-7 : 0));
    const int64_t N = kind == SS ? 1 : p->n;
    const int64_t C = kind == DATA ? p->c : (kind == SS ? 2 * p->c : 1);
    const auto nelems = N * C;
    r->total += nelems;

    diff_norm_t diff_norm;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            int64_t i = n * C + c;
            const float dt = dt_mem.get_elem(i);
            const float fp = fp_mem.get_elem(i);
            diff_norm.update(fp, dt);

            const float diff = fabsf(fp - dt);
            const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);
            bool ok = (fabsf(fp) > 1e-5 ? rel_diff : diff) <= eps;

            /* When the error is larger than eps, It could be
         * due to catastrophic cancellation in final result
         * which is computed as `Y = a * X + b`.
         * When `a * X`  is close to `b` and `sign(a * X) = - sign(b)`.
         * Then large error in `a * X` could result in a final
         * result (which has a cancellation i.e. `|Y| = |a*X - (-b)|`)
         * which has no meaningful digits left in mantissa.*/
            if (!ok && (p->dir & FLAG_FWD) && kind == DATA && ss) {
                const float beta = ((float *)*ss)[p->c + c];
                /* Using an empirically derived threshold,
             * check if cancellation error
             * in `|Y| = |a*X - (-b)|` is huge.*/
                bool maybe_cancellation_error
                        = (fabsf(fp - beta)
                                  / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1))
                        > 1.0f;
                if (maybe_cancellation_error) {
                    /* Check for error in `a * X` */
                    float diff_aX = fabsf((fp - beta) - (dt - beta));
                    float rel_diff_aX = diff_aX
                            / (fabsf(fp - beta) > FLT_MIN ? fabsf(fp - beta)
                                                          : 1);
                    ok = rel_diff_aX <= eps;
                }
            }

            r->errors += !ok;

            bool dump = false || (!ok && (r->errors < 10 || verbose >= 10))
                    || (verbose >= 50 && i < 30);
            if (dump) {
                std::stringstream ss;
                if (kind == SS) {
                    ss << i / p->c << "," << i % p->c;
                } else {
                    int64_t size = kind == DATA ? N * C : N;
                    size_t ndims = kind == DATA ? p->ndims : p->ndims - 1;
                    const char *separator = "";
                    for (size_t j = 0; j < ndims; j++) {
                        size /= p->dims[j];
                        ss << separator << (i / size) % p->dims[j];
                        separator = ",";
                    }
                }
                std::string ind_str = ss.str();
                print(0, "[%4ld][%s%s][%s] fp:%8g dt:%8g diff:%8g rdiff:%8g\n",
                        (long)i, p->dir & FLAG_BWD ? "D_" : "", skind,
                        ind_str.c_str(), fp, dt, diff, rel_diff);
            }
        }
    }

    diff_norm.done();

    if (r->errors || verbose >= 5) {
        const int vl = r->errors ? 0 : 2;
        print(vl,
                "@@@ [%s%s] diff: l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                p->dir & FLAG_BWD ? "D_" : "", skind,
                diff_norm.rel_diff(norm_t::L0), diff_norm.a_[norm_t::L1],
                diff_norm.b_[norm_t::L1], diff_norm.diff_[norm_t::L1],
                diff_norm.rel_diff(norm_t::L1), diff_norm.a_[norm_t::L2],
                diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
                diff_norm.rel_diff(norm_t::L2), diff_norm.a_[norm_t::L8],
                diff_norm.b_[norm_t::L8], diff_norm.diff_[norm_t::L8],
                diff_norm.rel_diff(norm_t::L8));
    }

    if (r->errors) r->state = FAILED;

    if (r->state == UNTESTED) r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

static int init_pd(const prb_t *p, dnnl_primitive_desc_t &lpd, res_t *r) {
    dnnl_layer_normalization_desc_t ld;
    dnnl_memory_desc_t data_d, stat_d;

    const int64_t *data_dims = &p->dims[0];

    DNN_SAFE(dnnl_memory_desc_init_by_tag(&data_d, p->ndims, data_dims, p->dt,
                     convert_tag(p->tag, p->ndims)),
            WARN);

    const dnnl_memory_desc_t *stat_d_ptr = NULL;
    if (p->stat_tag != tag::undef) {
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&stat_d, p->ndims - 1, data_dims,
                         dnnl_f32, convert_tag(p->stat_tag, p->ndims - 1)),
                WARN);
        stat_d_ptr = &stat_d;
    }

    auto flags = (dnnl_normalization_flags_t)p->flags;
    if (p->dir & FLAG_FWD) {
        auto prop = p->dir & FLAG_INF ? dnnl_forward_inference
                                      : dnnl_forward_training;
        DNN_SAFE(dnnl_layer_normalization_forward_desc_init(
                         &ld, prop, &data_d, stat_d_ptr, p->eps, flags),
                WARN);
    } else {
        dnnl_memory_desc_t diff_data_d;
        DNN_SAFE(dnnl_memory_desc_init_by_tag(&diff_data_d, p->ndims, data_dims,
                         p->dt, dnnl_format_tag_any),
                WARN);
        auto prop = p->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        DNN_SAFE(dnnl_layer_normalization_backward_desc_init(&ld, prop,
                         &diff_data_d, &data_d, stat_d_ptr, p->eps, flags),
                WARN);
    }

    dnnl_primitive_desc_t hint_fwd_pd = NULL;
    if (p->dir & FLAG_BWD) {
        dnnl_layer_normalization_desc_t ld_fwd;
        DNN_SAFE(dnnl_layer_normalization_forward_desc_init(&ld_fwd,
                         dnnl_forward_training, &data_d, stat_d_ptr, p->eps,
                         flags),
                WARN);
        dnnl_status_t init_fwd_status = dnnl_primitive_desc_create(
                &hint_fwd_pd, &ld_fwd, NULL, engine_tgt, NULL);
        if (init_fwd_status == dnnl_unimplemented)
            return r->state = UNIMPLEMENTED, OK;
        else
            SAFE(init_fwd_status, WARN);
    }

    auto dnnl_attr = create_dnnl_attr(p->attr);

    dnnl_status_t init_status = dnnl_primitive_desc_create(
            &lpd, &ld, dnnl_attr, engine_tgt, hint_fwd_pd);

    dnnl_primitive_desc_destroy(hint_fwd_pd);
    dnnl_primitive_attr_destroy(dnnl_attr);

    if (init_status == dnnl_unimplemented)
        return r->state = UNIMPLEMENTED, OK;
    else
        SAFE(init_status, WARN);

    const char *impl_str = query_impl_info(lpd);
    if (maybe_skip(skip_impl, impl_str)) {
        print(2, "SKIPPED: dnnl implementation: %s\n", impl_str);
        DNN_SAFE(dnnl_primitive_desc_destroy(lpd), WARN);
        return r->state = SKIPPED, OK;
    } else {
        print(5, "dnnl implementation: %s\n", impl_str);
        if (!strstr(impl_str, "jit")) {
            print(2, "WARNING: %s",
                    "accuracy of the implementation being tested "
                    "depends on the compiler and might give "
                    "false-positives.\n");
            print(2, "         %s",
                    "please consider recompiling the sources with"
                    " `-prec-div -fp-model precise` for a reliable testing.\n");
        }
    }

    return OK;
}

int doit(const prb_t *p, res_t *r) {
    if (bench_mode == LIST) return r->state = LISTED, OK;

    dnnl_primitive_desc_t lpd;
    SAFE(init_pd(p, lpd, r), WARN);
    if (r->state == SKIPPED || r->state == UNIMPLEMENTED) return OK;

    dnnl_primitive_t l;
    DNN_SAFE(dnnl_primitive_create(&l, lpd), WARN);
    DNN_SAFE(dnnl_primitive_desc_destroy(lpd), CRIT);

    const_dnnl_primitive_desc_t const_pd;
    DNN_SAFE(dnnl_primitive_get_primitive_desc(l, &const_pd), CRIT);

    if (dnn_mem_t::check_mem_size(const_pd) != OK) {
        DNN_SAFE_V(dnnl_primitive_destroy(l));
        return r->state = SKIPPED, OK;
    }

    const auto q = [&](int index = 0) -> const dnnl_memory_desc_t & {
        return *dnnl_primitive_desc_query_md(
                const_pd, dnnl_query_exec_arg_md, index);
    };

    const auto &data_md = q(DNNL_ARG_SRC);
    const auto &mean_md = q(DNNL_ARG_MEAN);
    const auto &var_md = q(DNNL_ARG_VARIANCE);
    const auto &ss_md = q(DNNL_ARG_SCALE_SHIFT);
    const auto &scratchpad_md = q(DNNL_ARG_SCRATCHPAD);

    const auto fp = dnnl_f32;
    const auto tag = get_abx_tag(p->ndims);

    dnn_mem_t src_fp(data_md, fp, tag, engine_tgt);
    dnn_mem_t src_dt(data_md, engine_tgt);

    dnn_mem_t &dst_fp = src_fp; // in-place reference
    dnn_mem_t placeholder_dst_dt;
    if (!p->inplace) { placeholder_dst_dt = dnn_mem_t(data_md, engine_tgt); }
    dnn_mem_t &dst_dt = p->inplace ? src_dt : placeholder_dst_dt;

    // On inference w/o global stats the layer norm doesn't require stat
    // memories. Hence, we need to prepare the mean_fp and var_fp ourselves.
    const auto stat_ndims = p->ndims - 1;
    const auto stat_tag = get_abx_tag(stat_ndims);
    dnn_mem_t mean_fp(stat_ndims, data_md.dims, fp, stat_tag, engine_tgt);
    dnn_mem_t mean_dt(mean_md, engine_tgt);

    dnn_mem_t var_fp(stat_ndims, data_md.dims, fp, stat_tag, engine_tgt);
    dnn_mem_t var_dt(var_md, engine_tgt);

    dnn_mem_t ss_fp(ss_md, fp, get_abx_tag(ss_md.ndims), engine_tgt);
    dnn_mem_t ss_dt(ss_md, engine_tgt);
    dnn_mem_t d_ss_fp(ss_md, fp, get_abx_tag(ss_md.ndims), engine_tgt);
    dnn_mem_t d_ss_dt(ss_md, engine_tgt);

    dnn_mem_t scratchpad_dt(scratchpad_md, engine_tgt);

    dnn_mem_t d_dst_dt, placeholder_d_src_dt;

    args_t args;

    if (p->dir & FLAG_FWD) {
        if (prepare_fwd(p, src_fp, mean_fp, var_fp, ss_fp) != OK) {
            DNN_SAFE_V(dnnl_primitive_destroy(l));
            return r->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        if (p->flags & GLOB_STATS) {
            /* prepare mean & var if they are inputs */
            SAFE(mean_dt.reorder(mean_fp), WARN);
            SAFE(var_dt.reorder(var_fp), WARN);
        }
        if (p->flags & USE_SCALESHIFT) { SAFE(ss_dt.reorder(ss_fp), WARN); }

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DST, dst_dt);
        args.set(DNNL_ARG_MEAN, mean_dt);
        args.set(DNNL_ARG_VARIANCE, var_dt);
        args.set(DNNL_ARG_SCALE_SHIFT, ss_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        DNN_SAFE(execute_and_wait(l, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_fwd(p, src_fp, mean_fp, var_fp, ss_fp, dst_fp);
            if (!(p->flags & GLOB_STATS) && !(p->dir & FLAG_INF)) {
                dnn_mem_t mean(mean_dt, fp, stat_tag, engine_tgt);
                dnn_mem_t var(var_dt, fp, stat_tag, engine_tgt);
                SAFE(compare(p, MEAN, mean_fp, mean, r), WARN);
                SAFE(compare(p, VAR, var_fp, var, r), WARN);
            }
            dnn_mem_t dst(dst_dt, fp, tag, engine_tgt);
            SAFE(compare(p, DATA, dst_fp, dst, r, &ss_fp), WARN);
        }
    } else {
        const auto &d_data_md = q(DNNL_ARG_DIFF_DST);

        dnn_mem_t d_dst_fp(d_data_md, fp, tag, engine_tgt);
        d_dst_dt = dnn_mem_t(d_data_md, engine_tgt);

        dnn_mem_t &d_src_fp = d_dst_fp; // in-place in ref code
        if (!p->inplace) {
            placeholder_d_src_dt = dnn_mem_t(d_data_md, engine_tgt);
        }
        dnn_mem_t &d_src_dt = p->inplace ? d_dst_dt : placeholder_d_src_dt;

        if (prepare_bwd(p, src_fp, d_dst_fp, mean_fp, var_fp, ss_fp) != OK) {
            DNN_SAFE_V(dnnl_primitive_destroy(l));
            return r->state = MISTRUSTED, OK;
        }

        SAFE(src_dt.reorder(src_fp), WARN);
        SAFE(d_dst_dt.reorder(d_dst_fp), WARN);
        SAFE(mean_dt.reorder(mean_fp), WARN);
        SAFE(var_dt.reorder(var_fp), WARN);
        if (p->flags & USE_SCALESHIFT) { SAFE(ss_dt.reorder(ss_fp), WARN); }

        args.set(DNNL_ARG_SRC, src_dt);
        args.set(DNNL_ARG_DIFF_DST, d_dst_dt);
        args.set(DNNL_ARG_DIFF_SRC, d_src_dt);
        args.set(DNNL_ARG_MEAN, mean_dt);
        args.set(DNNL_ARG_VARIANCE, var_dt);
        args.set(DNNL_ARG_SCALE_SHIFT, ss_dt);
        args.set(DNNL_ARG_DIFF_SCALE_SHIFT, d_ss_dt);
        args.set(DNNL_ARG_SCRATCHPAD, scratchpad_dt);

        DNN_SAFE(execute_and_wait(l, stream_tgt, args), WARN);

        if (bench_mode & CORR) {
            compute_ref_bwd(p, src_fp, mean_fp, var_fp, d_dst_fp, ss_fp,
                    d_src_fp, d_ss_fp);
            if ((p->flags & USE_SCALESHIFT) && (p->dir & FLAG_WEI)) {
                SAFE(compare(p, SS, d_ss_fp, d_ss_dt, r), WARN);
            }
            dnn_mem_t d_src(d_src_dt, fp, tag, engine_tgt);
            SAFE(compare(p, DATA, d_src_fp, d_src, r), WARN);
        }
    }

    measure_perf(r->timer, l, args);

    DNN_SAFE_V(dnnl_primitive_destroy(l));

    return OK;
}

} // namespace lnorm
