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

#ifndef RESAMPLING_HPP
#define RESAMPLING_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <iostream>

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "perf_report.hpp"

namespace resampling {

enum alg_t { nearest, linear };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
dnnl_alg_kind_t alg2alg_kind(alg_t alg);

struct desc_t {
    int64_t mb, ic;
    int64_t id, ih, iw;
    int64_t od, oh, ow;
    const char *name;
    int ndims;
};

int str2desc(desc_t *desc, const char *str);
std::ostream &operator<<(std::ostream &s, const desc_t &d);

struct prb_t : public desc_t {
    prb_t(const desc_t &desc, dir_t dir, dnnl_data_type_t dt,
            const std::string &tag, alg_t alg, int64_t mb = 0)
        : desc_t(desc), dir(dir), dt(dt), tag(tag), alg(alg) {
        if (mb) this->mb = mb;
    }
    ~prb_t() {}

    dir_t dir;
    dnnl_data_type_t dt;
    std::string tag;
    alg_t alg;

    BENCHDNN_DISALLOW_COPY_AND_ASSIGN(prb_t);
};
std::ostream &operator<<(std::ostream &s, const prb_t &p);

struct perf_report_t : public base_perf_report_t {
    using base_perf_report_t::base_perf_report_t;

    void report(const prb_t *p, const res_t *r, const char *prb_str) {
        p_ = p;
        base_report(r, prb_str);
    }

    virtual void dump_alg(std::ostream &s) const override {
        s << alg2str(p_->alg);
    }

    virtual void dump_desc(std::ostream &s) const override {
        s << static_cast<const desc_t &>(*p_);
    }

    virtual void dump_desc_csv(std::ostream &s) const override {
        s << p_->mb << ','

          << p_->ic << ',' << p_->id << ',' << p_->ih << ',' << p_->iw << ','

          << p_->od << ',' << p_->oh << ',' << p_->ow;
    }

    virtual const char *name() const override { return p_->name; }
    virtual const dir_t *dir() const override { return &p_->dir; }
    virtual const dnnl_data_type_t *dt() const override { return &p_->dt; }
    virtual const std::string *tag() const override { return &p_->tag; }

private:
    const prb_t *p_ = NULL;
};

/* some extra control parameters which shouldn't be placed in prb_t */
extern const char *skip_impl; /* NULL or "" means do not skip anything */
extern bool allow_unimpl; /* true means do not treat unimplemented as error */

inline int64_t src_off_f(const prb_t *p, int64_t mb, int64_t ic, int64_t id,
        int64_t ih, int64_t iw) {
    return (((mb * p->ic + ic) * p->id + id) * p->ih + ih) * p->iw + iw;
}

inline void inv_src_off_f(const prb_t *p, int64_t off, int64_t &mb, int64_t &ic,
        int64_t &id, int64_t &ih, int64_t &iw) {
    iw = off % p->iw;
    off /= p->iw;
    ih = off % p->ih;
    off /= p->ih;
    id = off % p->id;
    off /= p->id;
    ic = off % p->ic;
    off /= p->ic;
    mb = off % p->mb;
    off /= p->mb;
    assert(off == 0);
}

inline int64_t dst_off_f(const prb_t *p, int64_t mb, int64_t ic, int64_t od,
        int64_t oh, int64_t ow) {
    return (((mb * p->ic + ic) * p->od + od) * p->oh + oh) * p->ow + ow;
}

inline void inv_dst_off_f(const prb_t *p, int64_t off, int64_t &mb, int64_t &ic,
        int64_t &od, int64_t &oh, int64_t &ow) {
    ow = off % p->ow;
    off /= p->ow;
    oh = off % p->oh;
    off /= p->oh;
    od = off % p->od;
    off /= p->od;
    ic = off % p->ic;
    off /= p->ic;
    mb = off % p->mb;
    off /= p->mb;
    assert(off == 0);
}

void compute_ref_fwd(const prb_t *p, const dnn_mem_t &src, dnn_mem_t &dst);
void compute_ref_bwd(
        const prb_t *p, dnn_mem_t &diff_src, const dnn_mem_t &diff_dst);

int compare_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);
int compare_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);
int fill_dat(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *r);

int doit(const prb_t *p, res_t *res);
int bench(int argc, char **argv);

} // namespace resampling

#endif
