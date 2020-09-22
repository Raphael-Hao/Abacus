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

#ifndef PRIMITIVE_EXEC_TYPES_HPP
#define PRIMITIVE_EXEC_TYPES_HPP

#include <unordered_map>

#include "dnnl_types.h"

#include "c_types_map.hpp"
#include "memory.hpp"
#include "memory_storage.hpp"
#include "primitive_desc.hpp"

#define CTX_IN_STORAGE(arg) \
    (ctx.input(arg) ? *(ctx.input(arg)->memory_storage()) \
                    : dnnl::impl::memory_storage_t::empty_storage())

#define CTX_OUT_STORAGE(arg) \
    (ctx.output(arg) ? *(ctx.output(arg)->memory_storage()) \
                     : dnnl::impl::memory_storage_t::empty_storage())

namespace dnnl {
namespace impl {

namespace memory_tracking {
struct grantor_t;
} // namespace memory_tracking

struct memory_arg_t {
    memory_t *mem;
    bool is_const;
};

using exec_args_t = std::unordered_map<int, memory_arg_t>;

status_t cvt_primtive_args(const primitive_desc_t *pd, int nargs,
        const dnnl_exec_arg_t *c_args, exec_args_t &args);

/** Primitive execution context (helps passing stream, memories, and events. */
struct exec_ctx_t {
    exec_ctx_t(stream_t *stream) : stream_(stream) {}
    exec_ctx_t(stream_t *stream, exec_args_t &&args)
        : stream_(stream), args_(std::move(args)) {}

    stream_t *stream() const { return stream_; }
    const exec_args_t &args() const { return args_; }

    memory_t *input(int arg) const;
    memory_t *output(int arg) const;
    memory_t *memory(int arg) const;

    // Returns memory descriptor wrapper for the corresponding memory argument.
    //
    // To support sub-memory flow (when primitive descriptor was created with
    // a sub-memory, but the primitive is executed on the original memory),
    // it is recommended to pass the memory descriptor from the primitive
    // descriptor. If this memory descriptor is fully defined (i.e. no reason
    // to use memory descriptor from the input memory), exactly it will be
    // returned.
    //
    // Note: fully defined memory descriptor mentioned above is a synonym to
    //       `mdw::has_runtime_dims_or_strides() == false`.
    //
    // XXX: revisit this behavior in DNNL 2.0. It would be more consistent to
    //      take memory description from the incoming argument. This will
    //      require a sub-memory object, though...
    memory_desc_wrapper memory_mdw(int arg,
            const memory_desc_t *md_from_primitive_desc = nullptr) const;

    void set_scratchpad_grantor(
            const memory_tracking::grantor_t &scratchpad_grantor);
    const memory_tracking::grantor_t &get_scratchpad_grantor() const;

private:
    stream_t *stream_;
    exec_args_t args_;
    std::unique_ptr<memory_tracking::grantor_t> scratchpad_grantor_;
};

} // namespace impl
} // namespace dnnl

#endif
