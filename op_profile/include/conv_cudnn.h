/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: conv.hpp
 * \brief: cudnn convolution implementation
 * Created Date: Sunday, June 14th 2020, 7:24:40 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Monday, September 14th 2020, 4:36:37 pm
 * Modified By: raphael hao
 */
#pragma once
#include "conv_config.h"
#include "cuda_utils.h"

template <typename DType>
class CuDNNConvolutionOp {
 public:
  CuDNNConvolutionOp() {
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&forward_conv_desc_));
  }

  // initiate the operators
  void Init(const ConvolutionParam &param,
            const ShapeVector &in_shape,
            const ShapeVector &out_shape,
            const RunContext &rctx,
            bool add_to_weight) {
    this->param_ = param;
    this->add_to_weight_ = add_to_weight;
    auto cudnn_forward_compute_type = CUDNN_DATA_FLOAT;
    // convert MB to words
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    dtype_ = CUDNN_DATA_FLOAT;
    // TensorCore algos only allowed on fp16-I/O convolutions if permitted by the global policy.
    cudnn_tensor_core_ = true;

    auto effective_layout = CUDNN_TENSOR_NCHW;
    this->format_ = effective_layout;

    // initiate the tensor descriptors of cudnn
    InitDescriptors(in_shape, out_shape, cudnn_forward_compute_type);

    // select algorithm for convolution
    SelectAlgo(rctx, in_shape, out_shape, cudnn_forward_compute_type);
    GetTempSize(rctx);
    workspace_ = AllocateTempWorkspace(rctx, forward_workspace_byte_);
  }
  //clear global memory of workspace
  void Clear() {
    CUDA_CALL(cudaFree(workspace_));
    workspace_ = nullptr;
  }

  ~CuDNNConvolutionOp() {
    CUDA_CALL(cudaFree(workspace_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bias_desc_));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc_));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(forward_conv_desc_));
  }

  void Forward(const RunContext &ctx,
               const std::vector<DType *> &in_data,
               const std::vector<DType *> &out_data) {
    size_t expected = param_.no_bias ? 2 : 3;
    CHECK_EQ(in_data.size(), expected);
    CHECK_EQ(out_data.size(), 1U);

    // I/O's should have 2 more dims than the kernel dim
    DType *data_ptr = in_data[conv::kData];
    DType *wmat_ptr = in_data[conv::kWeight];
    DType *out_ptr = out_data[conv::kOut];

    float alpha = 1.0f;
    float beta = 0.0f;
    float beta_add = 1.0f;
    CUDNN_CALL(cudnnConvolutionForward(ctx.dnn_handle_,
                                       &alpha,
                                       in_desc_,
                                       data_ptr,
                                       filter_desc_,
                                       wmat_ptr,
                                       forward_conv_desc_,
                                       forward_algo_.AlgoNumber(),
                                       workspace_,
                                       forward_workspace_byte_,
                                       param_.no_bias ? &beta : &beta_add,
                                       out_desc_,
                                       out_ptr));

    if (!param_.no_bias) {
      DType *bias = in_data[conv::kBias];
      CUDNN_CALL(
          cudnnAddTensor(ctx.dnn_handle_, &alpha, bias_desc_, bias, &beta_add, out_desc_, out_ptr));
    }
  }

 private:
  void InitDescriptors(const ShapeVector &in_shape,
                       const ShapeVector &out_shape,
                       cudnnDataType_t cudnn_forward_compute_type) {
    size_t expected = param_.no_bias ? 2 : 3;
    DLOG(INFO) << std::string(10, '-') << "params" << std::string(10, '-');
    DLOG(INFO) << param_.kernel;
    DLOG(INFO) << param_.pad;
    DLOG(INFO) << param_.dilate;
    DLOG(INFO) << param_.no_bias;
    DLOG(INFO) << param_.workspace;
    DLOG(INFO) << param_.num_filter;
    DLOG(INFO) << param_.num_group;
    DLOG(INFO) << std::string(26, '-');
    CHECK_EQ(in_shape.size(), expected);
    CHECK_EQ(out_shape.size(), 1U);

    TShape dshape = in_shape[conv::kData];
    DLOG(INFO) << dshape[0] << " " << dshape[1] << " " << dshape[2] << " " << dshape[3];

    TShape wshape = in_shape[conv::kWeight];
    DLOG(INFO) << wshape[0] << " " << wshape[1] << " " << wshape[2] << " " << wshape[3];

    TShape oshape = out_shape[conv::kOut];
    DLOG(INFO) << oshape[0] << " " << oshape[1] << " " << oshape[2] << " " << oshape[3];

    TShape dstride{dshape[1] * dshape[2] * dshape[3], dshape[2] * dshape[3], dshape[3], 1};
    TShape ostride{oshape[1] * oshape[2] * oshape[3], oshape[2] * oshape[3], oshape[3], 1};

    DLOG(INFO) << dstride[0] << " " << dstride[1] << " " << dstride[2] << " " << dstride[3];

    DLOG(INFO) << ostride[0] << " " << ostride[1] << " " << ostride[2] << " " << ostride[3];

    // 1d or 2d conv
    auto pad = param_.pad;
    auto stride = param_.stride;
    auto dilate = param_.dilate;

    DLOG(INFO) << std::string(10, '-') << "conv desc" << std::string(10, '-');
    DLOG(INFO) << pad << " " << stride << " " << dilate;

    CUDNN_CALL(cudnnSetConvolution2dDescriptor(forward_conv_desc_,
                                               pad[0],
                                               pad[1],
                                               stride[0],
                                               stride[1],
                                               dilate[0],
                                               dilate[1],
                                               CUDNN_CONVOLUTION,
                                               cudnn_forward_compute_type));
    DLOG(INFO) << std::string(26, '-');
    DLOG(INFO) << std::string(10, '-') << "conv desc" << std::string(10, '-');
    DLOG(INFO) << wshape[0] << " " << wshape[1] << " " << wshape[2] << " " << wshape[3];
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc_,
                                          dtype_,
                                          format_,
                                          wshape[0],
                                          wshape[1],
                                          wshape[2],
                                          wshape[3]));

    DLOG(INFO) << std::string(26, '-');
    // Set "allow tensor core" flag in convolution descriptors, if available.
    cudnnMathType_t math_type =
        cudnn_tensor_core_ ? CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION : CUDNN_DEFAULT_MATH;

    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, math_type));
    CUDNN_CALL(cudnnSetConvolutionGroupCount(forward_conv_desc_, param_.num_group));

    std::vector<int> dshape_buffer(dshape.ndim());
    ShapeTypeCast(dshape.begin(), dshape.end(), dshape_buffer.data());
    std::vector<int> dstride_buffer(dstride.ndim());
    ShapeTypeCast(dstride.begin(), dstride.end(), dstride_buffer.data());
    std::vector<int> oshape_buffer(oshape.ndim());
    ShapeTypeCast(oshape.begin(), oshape.end(), oshape_buffer.data());
    std::vector<int> ostride_buffer(ostride.ndim());
    ShapeTypeCast(ostride.begin(), ostride.end(), ostride_buffer.data());

    CUDNN_CALL(cudnnSetTensorNdDescriptor(in_desc_,
                                          dtype_,
                                          static_cast<int>(dshape.ndim()),
                                          dshape_buffer.data(),
                                          dstride_buffer.data()));

    CUDNN_CALL(cudnnSetTensorNdDescriptor(out_desc_,
                                          dtype_,
                                          static_cast<int>(oshape.ndim()),
                                          oshape_buffer.data(),
                                          ostride_buffer.data()));

    if (!param_.no_bias) {
      TShape bias = in_shape[conv::kBias];
      std::vector<int> bias_shape = {1, static_cast<int>(bias[0]), 1, 1};
      std::vector<int> bias_stride = {static_cast<int>(bias[0]), 1, 1, 1};
      CUDNN_CALL(cudnnSetTensorNdDescriptor(bias_desc_,
                                            dtype_,
                                            static_cast<int>(bias_shape.size()),
                                            &bias_shape[0],
                                            &bias_stride[0]));
    }
  }

  void CuDNNAlgoSetter(const RunContext &rctx,
                       const ShapeVector &in_shape,
                       const ShapeVector &out_shape,
                       cudnnDataType_t cudnn_forward_compute_type,
                       CuDNNAlgo<cudnnConvolutionFwdAlgo_t> *fwd) {
    // Not in algo registry, must determine via *Get*() or *Find*()
    size_t workspace_byte = static_cast<size_t>(param_.workspace * sizeof(DType));

    // Forward Algorithm Find/Get() v7
    std::vector<cudnnConvolutionFwdAlgoPerf_t> fwd_results(MaxForwardAlgos(rctx.dnn_handle_));
    int actual_fwd_algos = 0;
    auto fwd_algo_discoverer = param_.cudnn_tune == true ? cudnnFindConvolutionForwardAlgorithm :
                                                           cudnnGetConvolutionForwardAlgorithm_v7;
    CUDNN_CALL((*fwd_algo_discoverer)(rctx.dnn_handle_,
                                      in_desc_,
                                      filter_desc_,
                                      forward_conv_desc_,
                                      out_desc_,
                                      fwd_results.size(),
                                      &actual_fwd_algos,
                                      fwd_results.data()));
    fwd_results.resize(actual_fwd_algos);
    AlgoFinalSelect<cudnnConvolutionFwdAlgoPerf_t, cudnnConvolutionFwdAlgo_t>(fwd_results,
                                                                              "forward",
                                                                              workspace_byte,
                                                                              fwd);
  }

  void SelectAlgo(const RunContext &rctx,
                  const ShapeVector &in_shape,
                  const ShapeVector &out_shape,
                  cudnnDataType_t cudnn_forward_compute_type) {
    // The routine will only be calling cudnnGet, so no need to grab the Storage lock.
    this->CuDNNAlgoSetter(rctx, in_shape, out_shape, cudnn_forward_compute_type, &forward_algo_);

    // If we're allowing Tensor Core variants of the algos to be considered in
    // *Find*() or *Get*(), but a non-Tensor-Core algo variant is the fastest,
    // we must change the descriptor to preclude Tensor Core.  Simplest is to
    // once again set the mathType in all cases.
    CUDNN_CALL(cudnnSetConvolutionMathType(forward_conv_desc_, forward_algo_.MathType()));
  }

  // Look over the results from *Find*() or *Get*() and pick the fastest algo given possible
  // workspace constraints.
  template <typename PerfType, typename AlgoType>
  void AlgoFinalSelect(const std::vector<PerfType> &perf_results,
                       std::string kernel_name,
                       size_t workspace_byte,
                       CuDNNAlgo<AlgoType> *algo,
                       int32_t algo_exclude = -1) {
    // Determine the fastest acceptable algo that matches the algo_preference (-1 = any),
    // regardless of mathType.
    for (decltype(perf_results.size()) i = 0; i != perf_results.size(); ++i) {
      const auto &result = perf_results[i];
      bool algo_exclusion = static_cast<int32_t>(result.algo) == algo_exclude;
      bool algo_is_tensor_core = false;
      algo_is_tensor_core = result.mathType == CUDNN_TENSOR_OP_MATH;
      DLOG(INFO) << result.status;
      DLOG(INFO) << result.determinism;
      DLOG(INFO) << result.memory;
      std::string tensor_core_info = algo_is_tensor_core ? "true" : "false";
      DLOG(INFO) << tensor_core_info << " matype: " << result.mathType;
      if (result.status == CUDNN_STATUS_SUCCESS &&
          result.determinism == cudnnDeterminism_t::CUDNN_DETERMINISTIC &&
          result.memory <= workspace_byte && !algo_exclusion) {
        // LOG(INFO) << "CUDNN Convolution Forward Algorithm Selected: "
        //  << CuDNNAlgo<AlgoType>::AlgoInfo(result.algo);
        // algo->Set(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED, algo_is_tensor_core);
        // algo->Set(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD, algo_is_tensor_core);
        // algo->Set(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, algo_is_tensor_core);
        // algo->Set(, algo_is_tensor_core);
        algo->Set(result.algo, algo_is_tensor_core);
        return;
      }
    }
    auto mode = " get ";
    LOG(FATAL) << "Failed to" << mode << "any " << kernel_name << " convolution algorithm. "
               << " with workspace size of " << workspace_byte << " bytes,"
               << " please consider reducing batch/model size or increasing the workspace size";
  }

  void GetTempSize(const RunContext &rctx) {
    // cudaMalloc returns addresses that are aligned for large accesses (e.g. to 512 bytes).
    // Since we only make one allocation and divide it into two parts when we parallelize
    // the dgrad and wgrad kernels, we round the sizes up to this alignment size so the
    // dptrs respect this alignment, even if the separate areas are stacked.

    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(rctx.dnn_handle_,
                                                       in_desc_,
                                                       filter_desc_,
                                                       forward_conv_desc_,
                                                       out_desc_,
                                                       forward_algo_.AlgoNumber(),
                                                       &forward_workspace_byte_));
  }

  int *CastTShapeToIntPtr(const TShape &s, std::vector<int> *buffer) {
    buffer->resize(s.ndim());
    std::copy(s.begin(), s.end(), buffer->data());
    return buffer->data();
  }

  // void InitBufferForParam() {
  //   CastTShapeToIntPtr(param_.stride, &param_stride_);
  //   CastTShapeToIntPtr(param_.dilate, &param_dilate_);
  //   CastTShapeToIntPtr(param_.pad, &param_pad_);
  // }

  // Round a value 'x' up to the next multiple of 'multiple'
  size_t RoundToMultiple(size_t x, size_t multiple) {
    size_t retVal = ((x + multiple - 1) / multiple) * multiple;
    return retVal;
  }

  // Allocates a 1D Tensor of words with size in bytes >= `size_bytes`.
  // Always allocates at least one word.
  DType *AllocateTempWorkspace(const RunContext &ctx, size_t size_bytes) {
    DType *tmp_dptr;
    CUDA_CALL(cudaMalloc(&tmp_dptr, size_bytes));
    return tmp_dptr;
  }

  // std::vector<int> param_stride_;
  // std::vector<int> param_dilate_;
  // std::vector<int> param_pad_;

  // Temp workspace size in bytes needed for Forward() operation.
  size_t forward_workspace_byte_;
  DType *workspace_ = nullptr;
  cudnnDataType_t dtype_;
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  // Convolution descriptor for forward inference operation
  cudnnConvolutionDescriptor_t forward_conv_desc_;
  // Algorithm for the forward inference operation
  CuDNNAlgo<cudnnConvolutionFwdAlgo_t> forward_algo_;

  cudnnTensorFormat_t format_;
  // Allow TensorCore algo policy
  bool cudnn_tensor_core_;
  // Is req[kWeight] == conv::kAddTo ?
  bool add_to_weight_;
  // Is there a dgrad algo that should be avoided (-1 == none)?
  int32_t exclude_dgrad_algo_ = -1;
  ConvolutionParam param_;
};