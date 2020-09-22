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

#include "gpu/gpu_impl_list.hpp"

#include "gpu/ocl/ref_concat.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

using cpd_create_f = dnnl::impl::engine_t::concat_primitive_desc_create_f;

namespace {
#define INSTANCE(...) __VA_ARGS__::pd_t::create
static const cpd_create_f concat_impl_list[] = {
        INSTANCE(ocl::ref_concat_t),
        nullptr,
};
#undef INSTANCE
} // namespace

const cpd_create_f *gpu_impl_list_t::get_concat_implementation_list() {
    return concat_impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl
