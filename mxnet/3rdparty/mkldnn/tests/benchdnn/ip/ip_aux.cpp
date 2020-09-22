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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"

#include "ip/ip.hpp"

namespace ip {

void prb_t::generate_oscales() {
    if (attr.oscale.policy != attr_t::scale_t::policy_t::PER_OC) return;

    scales = (float *)zmalloc(sizeof(float) * oc, 64);
    SAFE_V(scales != NULL ? OK : FAIL);

    const float K = 32;
    /* scale in [1/K .. K], with starting point at oscale.scale */
    float s[2] = {attr.oscale.scale, attr.oscale.scale / 2};
    for (int64_t i = 0; i < oc; ++i) {
        int64_t si = i % 2; // 0 -> left, 1 -> right
        scales[i] = s[si];
        if (si == 0) {
            s[si] /= 2.;
            if (s[si] < 1. / K) s[si] *= K * K; // turn around to become ~K
        } else {
            s[si] *= 2.;
            if (s[si] > K) s[si] /= K * K; // turn around to become ~K
        }
    }
}

int str2desc(desc_t *desc, const char *str) {
    // Canonical form: mbXicXidXihXiwXocXnS,
    // where
    //     X is integer
    //     S is string
    // note: symbol `_` is ignored.
    // Cubic/square shapes are supported by specifying just highest dimension.

    desc_t d {0};
    d.mb = 2;

    const char *s = str;
    assert(s);

#define CASE_NN(p, c) \
    do { \
        if (!strncmp(p, s, strlen(p))) { \
            ok = 1; \
            s += strlen(p); \
            char *end_s; \
            d.c = strtol(s, &end_s, 10); \
            s += (end_s - s); \
            if (d.c < 0) return FAIL; \
            /* printf("@@@debug: %s: %d\n", p, d. c); */ \
        } \
    } while (0)
#define CASE_N(c) CASE_NN(#c, c)
    while (*s) {
        int ok = 0;
        CASE_N(mb);
        CASE_N(ic);
        CASE_N(ih);
        CASE_N(iw);
        CASE_N(id);
        CASE_N(oc);
        if (*s == 'n') {
            d.name = s + 1;
            break;
        }
        if (*s == '_') ++s;
        if (!ok) return FAIL;
    }
#undef CASE_NN
#undef CASE_N

    if (d.ic == 0 || d.oc == 0) return FAIL;

    if (sanitize_desc(d.ndims, {d.id}, {d.ih}, {d.iw}, {1}) != OK) return FAIL;

    *desc = d;

    return OK;
}

std::ostream &operator<<(std::ostream &s, const desc_t &d) {
    bool print_d = true, print_h = true, print_w = true;
    print_dhw(print_d, print_h, print_w, d.ndims, {d.id}, {d.ih}, {d.iw});

    if (canonical || d.mb != 2) s << "mb" << d.mb;

    s << "ic" << d.ic;

    if (print_d) s << "id" << d.id;
    if (print_h) s << "ih" << d.ih;
    if (print_w) s << "iw" << d.iw;

    s << "oc" << d.oc;

    if (d.name) s << "n" << d.name;

    return s;
}

std::ostream &operator<<(std::ostream &s, const prb_t &p) {
    dump_global_params(s);

    if (canonical || p.dir != FWD_B) s << "--dir=" << dir2str(p.dir) << " ";
    if (canonical || p.cfg != conf_f32) s << "--cfg=" << cfg2str(p.cfg) << " ";
    if (canonical || p.stag != tag::any) s << "--stag=" << p.stag << " ";
    if (canonical || p.wtag != tag::any) s << "--wtag=" << p.wtag << " ";
    if (canonical || p.dtag != tag::any) s << "--dtag=" << p.dtag << " ";
    if (canonical || !p.attr.is_def()) s << "--attr=\"" << p.attr << "\" ";

    s << static_cast<const desc_t &>(p);

    return s;
}

} // namespace ip
