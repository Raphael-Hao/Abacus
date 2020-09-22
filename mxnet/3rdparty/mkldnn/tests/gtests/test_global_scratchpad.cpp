/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "dnnl.hpp"

namespace dnnl {

using dt = memory::data_type;
using tag = memory::format_tag;

// This test checks that globally defined primitive_t object will be
// successfully destroyed after finishing the program despite the order of
// internal objects destruction.
// The cause was thread-local non-trivially-constructed object in
// global_scratchpad_t object which got destroyed before global_scratchpad_t
// causing a crash.
class global_scratchpad : public ::testing::Test {};

struct conv_ctx_t {
    conv_ctx_t() : eng_(engine::kind::cpu, 0), c_() {}

    struct conv_t {
        conv_t()
            : src_md()
            , wei_md()
            , dst_md()
            , pd()
            , src_mem()
            , wei_mem()
            , dst_mem()
            , prim() {}

        memory::desc src_md;
        memory::desc wei_md;
        memory::desc dst_md;
        convolution_forward::primitive_desc pd;
        memory src_mem;
        memory wei_mem;
        memory dst_mem;
        primitive prim;
    };

    void Setup(memory::dims src_dims, memory::dims wei_dims,
            memory::dims dst_dims, memory::dims strides_dims,
            memory::dims dilations_dims, memory::dims padding_left,
            memory::dims padding_right) {
        c_.src_md = memory::desc(src_dims, dt::f32, tag::any);
        c_.wei_md = memory::desc(wei_dims, dt::f32, tag::any);
        c_.dst_md = memory::desc(dst_dims, dt::f32, tag::any);

        auto desc = convolution_forward::desc(prop_kind::forward,
                algorithm::convolution_direct, c_.src_md, c_.wei_md, c_.dst_md,
                strides_dims, dilations_dims, padding_left, padding_right);

        c_.pd = convolution_forward::primitive_desc(desc, eng_);

        c_.src_mem = memory(c_.pd.src_desc(), eng_);
        c_.wei_mem = memory(c_.pd.weights_desc(), eng_);
        c_.dst_mem = memory(c_.pd.dst_desc(), eng_);

        c_.prim = convolution_forward(c_.pd);
    }

    engine eng_;
    struct conv_t c_;
};

conv_ctx_t global_conv_ctx1;
conv_ctx_t global_conv_ctx2;

TEST(global_scratchpad, TestGlobalScratchpad) {
    memory::dims src1 = {1, 1, 3, 4};
    memory::dims wei1 = {1, 1, 3, 3};
    memory::dims dst1 = {1, 1, 8, 5};
    memory::dims str1 = {1, 1};
    memory::dims dil1 = {0, 0};
    memory::dims pad_l1 = {3, 1};
    memory::dims pad_r1 = {4, 2};
    global_conv_ctx1.Setup(src1, wei1, dst1, str1, dil1, pad_l1, pad_r1);

    memory::dims src2 = {256, 3, 227, 227};
    memory::dims wei2 = {96, 3, 11, 11};
    memory::dims dst2 = {256, 96, 55, 55};
    memory::dims str2 = {4, 4};
    memory::dims dil2 = {0, 0};
    memory::dims pad_l2 = {0, 0};
    memory::dims pad_r2 = {0, 0};
    global_conv_ctx2.Setup(src2, wei2, dst2, str2, dil2, pad_l2, pad_r2);

    // if something goes wrong, test should return 139 on Linux.
};

} // namespace dnnl
