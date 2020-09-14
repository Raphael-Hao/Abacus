/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: cuda_utils.h
 * \brief: ease coding for NVIDIA GPU
 * Created Date: Saturday, June 13th 2020, 8:38:49 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Tuesday, July 7th 2020, 8:35:39 pm
 * Modified By: raphael hao
 */

#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>
#include <dmlc/logging.h>

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                                                            \
  {                                                                                                \
    cudaError_t e = (func);                                                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) << "CUDA: " << cudaGetErrorString(e); \
  }

#define CUDNN_CALL(func)                                                                           \
  {                                                                                                \
    cudnnStatus_t e = (func);                                                                      \
    CHECK_EQ(e, CUDNN_STATUS_SUCCESS) << "cuDNN: " << cudnnGetErrorString(e);                      \
  }

/*!
 * \brief Return max number of perf structs
 * cudnnFindConvolutionForwardAlgorithm() may want to populate. \param
 * cudnn_handle cudnn handle needed to perform the inquiry. \return max number
 * of perf structs cudnnFindConvolutionForwardAlgorithm() may want to populate.
 */
inline int MaxForwardAlgos(cudnnHandle_t cudnn_handle) {
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
}

/*!
 * \brief Return max number of perf structs
 * cudnnFindConvolutionBackwardFilterAlgorithm() may want to populate. \param
 * cudnn_handle cudnn handle needed to perform the inquiry. \return max number
 * of perf structs cudnnFindConvolutionBackwardFilterAlgorithm() may want to
 * populate.
 */
inline int MaxBackwardFilterAlgos(cudnnHandle_t cudnn_handle) {
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
}

/*!
 * \brief Return max number of perf structs
 * cudnnFindConvolutionBackwardDataAlgorithm() may want to populate. \param
 * cudnn_handle cudnn handle needed to perform the inquiry. \return max number
 * of perf structs cudnnFindConvolutionBackwardDataAlgorithm() may want to
 * populate.
 */
inline int MaxBackwardDataAlgos(cudnnHandle_t cudnn_handle) {
  int max_algos = 0;
  CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn_handle, &max_algos));
  return max_algos;
}

#define FREE_DPTR_VEC(dptr_vec)                                                                    \
  {                                                                                                \
    for (auto tmp_dptr : dptr_vec) {                                                               \
      CUDA_CALL(cudaFree(tmp_dptr));                                                               \
    }                                                                                              \
  }

#define GET_DEVICE_MEM_INFO(STAGE)                                                                 \
  {                                                                                                \
    size_t free_space, total_space;                                                                \
    CUDA_CALL(cudaMemGetInfo(&free_space, &total_space));                                          \
    LG << "Stage: " << #STAGE << " Used global memory size: " << ((total_space - free_space) >> 20) \
       << " Mbytes";                                                                               \
  }
