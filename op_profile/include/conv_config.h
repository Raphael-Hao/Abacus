/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: conv_config.h
 * \brief: configuration of convolution
 * Created Date: Tuesday, June 23rd 2020, 10:17:32 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Monday, September 14th 2020, 1:49:44 pm
 * Modified By: raphael hao
 */
#pragma once

#include <dmlc/parameter.h>

#include "cuda_utils.h"
#include "shape.h"
namespace conv {
enum ConvolutionOpInputs { kData, kWeight, kBias };
enum ConvolutionOpOutputs { kOut };
enum ConvolutionOpResource { kTempSpace };
}  // namespace conv

struct ConvolutionParam : public dmlc::Parameter<ConvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint32_t workspace;
  TShape input_shape;
  TShape output_shape;
  bool no_bias;
  bool if_output_set = false;
  bool cudnn_tune;
  DMLC_DECLARE_PARAMETER(ConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride)
        .set_default(TShape(1, 1))
        .describe(
            "Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate)
        .set_default(TShape(1, 1))
        .describe(
            "Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad)
        .set_default(TShape(0, 0))
        .describe("Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter)
        .set_range(1, 100000)
        .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1).describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192).describe(
        "Maximum temporary workspace allowed (MB) in convolution."
        "This parameter has two usages. When CUDNN is not used, it determines the "
        "effective batch size of the convolution kernel. When CUDNN is used, it controls "
        "the maximum temporary storage used for tuning the best CUDNN kernel when "
        "`limited_workspace` strategy is used.");
    DMLC_DECLARE_FIELD(no_bias).set_default(false).describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(input_shape)
        .set_default(TShape(std::vector<dim_t>{3, 128, 128}))
        .describe("dimension of input shape");
    DMLC_DECLARE_FIELD(output_shape)
        .set_default(TShape(std::vector<dim_t>{8, 126, 126}))
        .describe("dimension of output shape");
    DMLC_DECLARE_FIELD(cudnn_tune).set_default(true).describe("whether use cudnn tune?");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  void SetOutShape() {
    output_shape[1] = (input_shape[1] - kernel[0] + 2 * pad[0]) / stride[0] + 1;
    output_shape[2] = (input_shape[2] - kernel[1] + 2 * pad[1]) / stride[1] + 1;
    if_output_set = true;
  }
  template <typename KType>
  ShapeVector GetShape(int bs) {
    if (if_output_set == false) {
      SetOutShape();
    }
    if (std::is_same<KType, conv::ConvolutionOpInputs>::value) {
      TShape dshape{static_cast<dim_t>(bs), input_shape[0], input_shape[1], input_shape[2]};

      TShape wshape{output_shape[0], input_shape[0], kernel[0], kernel[1]};

      TShape bshape{output_shape[0]};

      return ShapeVector{dshape, wshape, bshape};
    } else if (std::is_same<KType, conv::ConvolutionOpOutputs>::value) {
      TShape oshape{static_cast<dim_t>(bs), output_shape[0], output_shape[1], output_shape[2]};

      return ShapeVector{oshape};
    } else {
      LOG(FATAL) << "tensor shape not supported!!!";
    }
  }
  bool operator==(const ConvolutionParam& other) const {
    return this->kernel == other.kernel && this->stride == other.stride &&
           this->dilate == other.dilate && this->pad == other.pad &&
           this->num_filter == other.num_filter && this->num_group == other.num_group &&
           this->workspace == other.workspace && this->no_bias == other.no_bias;
  }
};


class RunContext {
 public:
  RunContext() {
    CUDA_CALL(cudaStreamCreate(&dnn_stream_));
    CUDNN_CALL(cudnnCreate(&dnn_handle_));
    CUDNN_CALL(cudnnSetStream(dnn_handle_, dnn_stream_));
  }
  ~RunContext() {
    CUDNN_CALL(cudnnDestroy(dnn_handle_));
    CUDA_CALL(cudaStreamDestroy(dnn_stream_));
  }

  cudnnHandle_t dnn_handle_;
  cudaStream_t dnn_stream_;
};

template <typename CuDNNAlgoType>
class CuDNNAlgo {
 public:
  CuDNNAlgo() : algo_number_(static_cast<CuDNNAlgoType>(0)), is_tensor_core_algo_(false) {}
  void Set(CuDNNAlgoType algo, bool is_tensor_core) {
    algo_number_ = algo;
    is_tensor_core_algo_ = is_tensor_core;
  }
  CuDNNAlgoType AlgoNumber() const {
    return algo_number_;
  }
  bool IsTensorCoreAlgo() const {
    return is_tensor_core_algo_;
  }
  cudnnMathType_t MathType() {
    return IsTensorCoreAlgo() ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
  }

  static std::string AlgoInfo(CuDNNAlgoType algo) {
    return algo_infos_[algo];
  }

 private:
  static std::map<CuDNNAlgoType, std::string> algo_infos_;
  CuDNNAlgoType algo_number_;
  bool is_tensor_core_algo_;
};