/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: profiler.h
 * \brief: profiler of merge batch
 * Created Date: Friday, June 26th 2020, 10:02:15 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Monday, September 14th 2020, 1:42:34 pm
 * Modified By: raphael hao
 */

#pragma once

#include <dmlc/io.h>

#include "conv_config.h"
#include "conv_cudnn.h"

using RCTXVector = std::vector<RunContext>;

template <typename DType>
using OPVector = std::vector<CuDNNConvolutionOp<DType>>;

template <typename DType>
using DPtrVector = std::vector<DType *>;

template <typename DType>
using DataVector = std::vector<DPtrVector<DType>>;

using ResultMap = std::map<std::string, std::vector<std::unordered_map<std::string, double>>>;

using ResultVal = std::vector<std::unordered_map<std::string, double>>;
template <typename DType>
class Profiler {
 public:
  Profiler(const size_t &epoch, const size_t &max_bs) : epoch_(epoch), max_bs_(max_bs) {
    CUDA_CALL(cudaEventCreate(&start_event_));
    CUDA_CALL(cudaEventCreate(&end_event_));
  }

  void Init(const int bs, const ConvolutionParam &param, const bool split = false) {
    split_ = split;
    this->cleared_ = false;
    this->param_ = param;
    this->bs_ = bs;
    af_inshape_ = param_.GetShape<conv::ConvolutionOpInputs>(bs_);
    af_outshape_ = param_.GetShape<conv::ConvolutionOpOutputs>(bs_);
    af_op_.Init(param_, af_inshape_, af_outshape_, global_rctx_, true);
    std::tie(af_indata_, af_outdata_) = AllocateData(af_inshape_, af_outshape_);
    if (split_) {
      bf_inshape_ = param_.GetShape<conv::ConvolutionOpInputs>(1);
      bf_outshape_ = param_.GetShape<conv::ConvolutionOpOutputs>(1);
      bf_op_vec_.resize(bs_);
      bf_rctx_vec_.resize(bs_);
      bf_indata_vec_.resize(bs_);
      bf_outdata_vec_.resize(bs_);
      for (size_t i = 0; i < bs_; i++) {
        // bf_rctx_vec_.emplace_back();
        bf_op_vec_[i].Init(this->param_, bf_inshape_, bf_outshape_, bf_rctx_vec_[i], true);
        std::tie(bf_indata_vec_[i], bf_outdata_vec_[i]) = AllocateData(bf_inshape_, bf_outshape_);
      }
    }
  }

  void Clear() {
    FREE_DPTR_VEC(af_indata_);
    FREE_DPTR_VEC(af_outdata_);
    af_op_.Clear();
    if (split_) {
      for (auto dptr_vec : bf_indata_vec_) {
        FREE_DPTR_VEC(dptr_vec);
      }
      for (auto dptr_vec : bf_outdata_vec_) {
        FREE_DPTR_VEC(dptr_vec);
      }
      bf_op_vec_.clear();
      bf_rctx_vec_.clear();
      split_ = false;
    }
    cleared_ = true;
  }

  void DumpResult() {
    std::string outputfile = "results.json";
    std::unique_ptr<dmlc::Stream> fo(dmlc::Stream::Create(outputfile.c_str(), "w"));
    dmlc::ostream out_stream(fo.get());
    dmlc::JSONWriter writer(&out_stream);
    writer.Write(results_);
  }

  void Profile(bool perf_all = false) {
    // LG << "Profiling: " << GetConvKey() << " batch size: " << bs_;
    CHECK_EQ(cleared_, false) << "profiler needs to be initiated before profiling";
    if (split_) {
      if (perf_all) {
        this->ProfileAll();
      } else {
        this->ProfileMergeOrNot();
      }
    } else {
      LG << this->bs_ << "," << this->ProfileOnce(0);
    }
    // this->DumpResult();
  }

  void ProfileAll() {
    this->WarmUp();
    if (split_) {
      this->ProfileOnce(0);
    } else {
      for (auto i = 0; i < this->epoch_; i++) {
        this->ProfileOnce(i);
        // this->ProfileEmpty();
      }
    }
  }
  void ProfileMergeOrNot() {
    this->WarmUp();
    double ms_bs, ms_split;
    ms_bs = this->ProfileOnce(0);
    ms_split = this->ProfileOnce(this->epoch_);
    LG << this->bs_ << "," << ms_bs << "," << ms_split;
  }

  ~Profiler() {
    CUDA_CALL(cudaEventDestroy(start_event_));
    CUDA_CALL(cudaEventDestroy(end_event_));
  }

 private:
  void WarmUp() {
    // LG << std::string(20, '-') << "warming up" << std::string(20, '-');
    for (auto i = 0; i < this->epoch_; i++) {
      if (split_) {
        for (size_t i = 0; i < bs_; i++) {
          bf_op_vec_[i].Forward(bf_rctx_vec_[i], bf_indata_vec_[i], bf_outdata_vec_[i]);
        }
      }
      af_op_.Forward(global_rctx_, af_indata_, af_outdata_);
    }
  }

  void ProfileEmpty() {
    CUDA_CALL(cudaEventRecord(this->start_event_));
    for (auto i = 0; i < this->cur_index_; i++) {
      for (size_t i = 0; i < bs_; i++) {
        // bf_op_vec_[i].Forward(bf_rctx_vec_[i], bf_indata_vec_[i], bf_outdata_vec_[i]);
      }
    }
    CUDA_CALL(cudaDeviceSynchronize());
    for (auto i = this->cur_index_; i < this->epoch_; i++) {
      // af_op_.Forward(global_rctx_, af_indata_, af_outdata_);
    }
    CUDA_CALL(cudaEventRecord(this->end_event_));
    CUDA_CALL(cudaEventSynchronize(this->end_event_));
    CUDA_CALL(cudaEventElapsedTime(&this->time_elapsed_, this->start_event_, this->end_event_));
    LG << "Empty Profile---current index: " << this->cur_index_
       << " elapsed time: " << this->time_elapsed_;
  }

  double ProfileOnce(int const &index) {
    CUDA_CALL(cudaEventRecord(this->start_event_));
    this->cur_index_ = index;
    for (auto i = 0; i < index; i++) {
      for (size_t i = 0; i < bs_; i++) {
        bf_op_vec_[i].Forward(bf_rctx_vec_[i], bf_indata_vec_[i], bf_outdata_vec_[i]);
      }
    }
    CUDA_CALL(cudaDeviceSynchronize());
    for (auto i = index; i < this->epoch_; i++) {
      af_op_.Forward(global_rctx_, af_indata_, af_outdata_);
    }
    CUDA_CALL(cudaEventRecord(this->end_event_));
    CUDA_CALL(cudaEventSynchronize(this->end_event_));
    CUDA_CALL(cudaEventElapsedTime(&this->time_elapsed_, this->start_event_, this->end_event_));
    // LG << "True Profile---current index: " << index << " elapsed time: " << this->time_elapsed_;
    UpdateResult();
    return this->time_elapsed_;
  }

  std::tuple<std::vector<DType *>, std::vector<DType *>> AllocateData(ShapeVector &in_shape,
                                                                      ShapeVector &out_shape) {
    DType *ddptr, *wdptr, *bdptr, *odptr;
    auto dshape = in_shape[conv::kData];
    auto wshape = in_shape[conv::kWeight];
    auto bshape = in_shape[conv::kBias];
    auto oshape = out_shape[conv::kOut];
    CUDA_CALL(cudaMalloc(&ddptr, dshape[0] * dshape[1] * dshape[2] * dshape[3] * sizeof(DType)));
    CUDA_CALL(cudaMalloc(&wdptr, wshape[0] * wshape[1] * wshape[2] * wshape[3] * sizeof(DType)));
    CUDA_CALL(cudaMalloc(&bdptr, bshape[0] * sizeof(DType)));
    CUDA_CALL(cudaMalloc(&odptr, oshape[0] * oshape[1] * oshape[2] * oshape[3] * sizeof(DType)));
    return std::make_tuple(std::vector<DType *>{ddptr, wdptr, bdptr}, std::vector<DType *>{odptr});
  }

  std::string GetConvKey() {
    auto input_str = std::to_string(param_.input_shape[0]) + 'x' +
                     std::to_string(param_.input_shape[1]) + 'x' +
                     std::to_string(param_.input_shape[2]);

    auto kernel_str = std::to_string(param_.output_shape[0]) + 'x' +
                      std::to_string(param_.input_shape[0]) + 'x' +
                      std::to_string(param_.kernel[0]) + 'x' + std::to_string(param_.kernel[1]);
    auto pad_str = std::to_string(param_.pad[0]) + 'x' + std::to_string(param_.pad[1]);
    auto stride_str = std::to_string(param_.stride[0]) + 'x' + std::to_string(param_.stride[1]);
    return input_str + '_' + kernel_str + '_' + pad_str + '_' + stride_str;
  }

  void UpdateResult() {
    std::string conv_key = GetConvKey();
    auto ret_it = results_.find(conv_key);
    if (ret_it == results_.end()) {
      results_[conv_key].resize(max_bs_ - 1);
    }
    results_[conv_key][bs_ - 2][std::to_string(this->cur_index_)] = this->time_elapsed_;
  }

  // timer
  bool split_;
  cudaEvent_t start_event_, end_event_;
  float time_elapsed_;
  ResultMap results_;
  // params
  ConvolutionParam param_;

  const size_t epoch_;
  const size_t max_bs_;

  size_t cur_index_ = 0;
  size_t bs_;

  bool cleared_ = true;
  // cudnn and cuda object
  RunContext global_rctx_;
  CuDNNConvolutionOp<DType> af_op_;
  ShapeVector af_inshape_;
  ShapeVector af_outshape_;
  DPtrVector<DType> af_indata_;
  DPtrVector<DType> af_outdata_;

  std::vector<RunContext> bf_rctx_vec_;
  OPVector<DType> bf_op_vec_;
  ShapeVector bf_inshape_;
  ShapeVector bf_outshape_;
  DataVector<DType> bf_indata_vec_;
  DataVector<DType> bf_outdata_vec_;
};
