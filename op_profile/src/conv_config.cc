/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: conv_config.cc
 * \brief:
 * Created Date: Monday, September 14th 2020, 1:31:44 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Monday, September 14th 2020, 1:45:17 pm
 * Modified By: raphael hao
 */

#include "conv_config.h"

DMLC_REGISTER_PARAMETER(ConvolutionParam);

template <>
std::map<cudnnConvolutionFwdAlgo_t, std::string> CuDNNAlgo<cudnnConvolutionFwdAlgo_t>::algo_infos_ =
    {{CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"},
     {CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
      "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM"},
     {CUDNN_CONVOLUTION_FWD_ALGO_GEMM, "CUDNN_CONVOLUTION_FWD_ALGO_GEMM"},
     {CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"},
     {CUDNN_CONVOLUTION_FWD_ALGO_FFT, "CUDNN_CONVOLUTION_FWD_ALGO_FFT"},
     {CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING, "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING"},
     {CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"},
     {CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"},
     {CUDNN_CONVOLUTION_FWD_ALGO_COUNT, "CUDNN_CONVOLUTION_FWD_ALGO_COUNT"}};