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

#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <iostream>

#include "dnnl.h"

#include "common.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace matmul {

typedef struct dt_conf_t {
    dnnl_data_type_t dt;
    double min, max; /* representative */
    double f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double f_scale; /* fill scale, scaling factor for integer generated data */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;
extern const _dt_conf_t conf_u8s8s32;
extern const _dt_conf_t conf_u8s8s8;
extern const _dt_conf_t conf_u8s8u8;
extern const _dt_conf_t conf_bf16bf16f32;
extern const _dt_conf_t conf_bf16bf16bf16;
extern const _dt_conf_t conf_f32bf16bf16;
extern const _dt_conf_t conf_bf16f32bf16;

const int64_t LD_GOOD = INT64_MAX;
const int64_t LD_NONE = INT64_MAX - 1;

// default driver setting
namespace defaults {
extern const dt_conf_t *cfg; // = conf_f32;
const std::string tag(tag::abx);
const int64_t ld = LD_NONE;
const bool runtime_val = false;
const dnnl_data_type_t bia_dt = dnnl_data_type_undef;
const int bia_mask = 2;
} // namespace defaults

struct desc_t {
    int ndims; // if 2, mb = 1.
    int64_t mb, m, n, k;

    const char *name;
};
int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, const dt_conf_t *cfg, const std::string &stag,
            const std::string &wtag, const std::string &dtag, int64_t ld_src,
            int64_t ld_wei, int64_t ld_dst, bool runtime_mb, bool runtime_m,
            bool runtime_n, bool runtime_k, dnnl_data_type_t bia_dt,
            int bia_mask, const attr_t &attr)
        : desc_t(desc)
        , cfg(cfg)
        , stag(stag)
        , wtag(wtag)
        , dtag(dtag)
        , ld_src(ld_src)
        , ld_wei(ld_wei)
        , ld_dst(ld_dst)
        , runtime_mb(runtime_mb)
        , runtime_m(runtime_m)
        , runtime_n(runtime_n)
        , runtime_k(runtime_k)
        , bia_dt(bia_dt)
        , bia_mask(bia_mask)
        , attr(attr)
        , ops(2. * mb * m * n * k)
        , scales(NULL) {
        generate_oscales();
    }
    ~prb_t() {
        if (scales) zfree(scales);
    }

    const dt_conf_t *cfg;

    std::string stag, wtag, dtag;
    int64_t ld_src, ld_wei, ld_dst;
    bool runtime_mb, runtime_m, runtime_n, runtime_k;
    dnnl_data_type_t bia_dt;
    int bia_mask;

    attr_t attr;

    double ops;
    float *scales;

    void generate_oscales();

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);
};
std::ostream &operator<<(std::ostream &s, const prb_t &p);

/* some extra control parameters which shouldn't be placed in prb_t */
extern const char *skip_impl; /* NULL or "" means do not skip anything */

const dt_conf_t *str2cfg(const char *str);
const char *cfg2str(const dt_conf_t *cfg);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        base_report(r, prb_str);
    }

    virtual void dump_cfg(std::ostream &s) const override {
        s << cfg2str(p_->cfg);
    }

    virtual void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    virtual void dump_desc_csv(std::ostream &s) const override {
        s << p_->ndims << ',' << p_->mb << ',' << p_->m << ',' << p_->n << ','
          << p_->k;
    }

    virtual double ops() const override { return p_->ops; }
    virtual const attr_t *attr() const override { return &p_->attr; }
    virtual const char *name() const override { return p_->name; }

private:
    const prb_t *p_ = NULL;
};

inline int64_t src_off_f(const prb_t *p, int64_t mb, int64_t m, int64_t k) {
    return (mb * p->m + m) * p->k + k;
}

inline int64_t wei_off_f(const prb_t *p, int64_t mb, int64_t k, int64_t n) {
    return (mb * p->k + k) * p->n + n;
}

inline int64_t dst_off_f(const prb_t *p, int64_t mb, int64_t m, int64_t n) {
    return (mb * p->m + m) * p->n + n;
}

inline int64_t bia_off_f(const prb_t *p, int64_t mb, int64_t m, int64_t n) {
    int64_t bia_stride[3] = {p->m * p->n, p->n, 1}, factor = 1;
    if ((p->bia_mask & (1 << ((p->ndims == 3) + 1))) == 0) {
        bia_stride[2] = 0;
        factor *= p->n;
    }
    if ((p->bia_mask & (1 << ((p->ndims == 3) + 0))) == 0) {
        bia_stride[1] = 0;
        factor *= p->m;
    } else {
        bia_stride[1] /= factor;
    }
    if (p->ndims == 2 || ((p->bia_mask & (1 << 0)) == 0)) {
        bia_stride[0] = 0;
    } else {
        bia_stride[0] /= factor;
    }
    return mb * bia_stride[0] + m * bia_stride[1] + n * bia_stride[2];
}

void compute_ref(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m);

int doit(const prb_t *p, res_t *res);

int bench(int argc, char **argv);

} // namespace matmul

#endif
