/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#include "dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "primitive_desc.hpp"
#include "primitive_iterator.hpp"
#include "type_helpers.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;

status_t dnnl_primitive_desc_iterator_create(
        primitive_desc_iterator_t **iterator, const_c_op_desc_t c_op_desc,
        const primitive_attr_t *attr, engine_t *engine,
        const primitive_desc_t *hint_fwd_pd) {
    const op_desc_t *op_desc = (const op_desc_t *)c_op_desc;
    if (utils::any_null(iterator, op_desc, engine)) return invalid_arguments;

    using namespace primitive_kind;
    bool known_primitive_kind = utils::one_of(op_desc->kind,
            batch_normalization, binary, convolution, deconvolution, eltwise,
            gemm, inner_product, layer_normalization, lrn, logsoftmax, matmul,
            pooling, resampling, rnn, shuffle, softmax);
    if (!known_primitive_kind) return invalid_arguments;

    auto it = new primitive_desc_iterator_t(engine, op_desc, attr, hint_fwd_pd);
    if (it == nullptr) return out_of_memory;

    ++(*it);
    if (*it == it->end()) {
        delete it;
        return unimplemented;
    }

    *iterator = it;
    return success;
}

status_t dnnl_primitive_desc_iterator_next(
        primitive_desc_iterator_t *iterator) {
    if (iterator == nullptr) return invalid_arguments;
    ++(*iterator);
    return *iterator == iterator->end() ? iterator_ends : success;
}

primitive_desc_t *dnnl_primitive_desc_iterator_fetch(
        const primitive_desc_iterator_t *iterator) {
    if (iterator == nullptr) return nullptr;
    return *(*iterator);
}

status_t dnnl_primitive_desc_clone(primitive_desc_t **primitive_desc,
        const primitive_desc_t *existing_primitive_desc) {
    if (utils::any_null(primitive_desc, existing_primitive_desc))
        return invalid_arguments;
    return safe_ptr_assign<primitive_desc_t>(
            *primitive_desc, existing_primitive_desc->clone());
}

status_t dnnl_primitive_desc_iterator_destroy(
        primitive_desc_iterator_t *iterator) {
    if (iterator != nullptr) delete iterator;
    return success;
}

status_t dnnl_primitive_desc_create(primitive_desc_t **primitive_desc,
        const_c_op_desc_t c_op_desc, const primitive_attr_t *attr,
        engine_t *engine, const primitive_desc_t *hint_fwd_pd) {
    primitive_desc_iterator_t *it;
    status_t status = dnnl_primitive_desc_iterator_create(
            &it, c_op_desc, attr, engine, hint_fwd_pd);
    if (status != status::success) return status;

    primitive_desc_t *pd = it->fetch_once();
    dnnl_primitive_desc_iterator_destroy(it);
    if (pd == nullptr) return out_of_memory;

    *primitive_desc = pd;

    return success;
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
