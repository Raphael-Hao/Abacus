/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: test_conv.cc
 * \brief: convolution test
 * Created Date: Thursday, June 25th 2020, 2:14:32 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Monday, September 14th 2020, 5:11:59 pm
 * Modified By: raphael hao
 */

#include <dmlc/io.h>
#include <dmlc/json.h>

#include <algorithm>
#include <iostream>
#include <tuple>

#include "conv_config.h"
#include "conv_cudnn.h"
#include "profiler.h"
#include "utils.h"

using param_obj = std::unordered_map<std::string, std::string>;
using param_vec = std::vector<param_obj>;

// register the class as dmlc:any before using class as dmlc:any
DMLC_JSON_ENABLE_ANY(param_obj, ParamOBJ);
DMLC_JSON_ENABLE_ANY(param_vec, ParamVec);
DMLC_JSON_ENABLE_ANY(int, Int);

// load params from json file and remove duplicate configuration of operators
std::tuple<int, int, param_vec> LoadParamFile(std::string filename) {
  std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(filename.c_str(), "r"));

  dmlc::istream is(fi.get());

  dmlc::JSONReader reader(&is);

  std::unordered_map<std::string, dmlc::any> kwargs;

  reader.Read(&kwargs);

  int layer_cnt = dmlc::get<int>(kwargs["layer_cnt"]);
  int max_bs = dmlc::get<int>(kwargs["max_batchsize"]);

  param_vec conv_candidate = dmlc::get<param_vec>(kwargs["conv"]);

  param_vec conv_candi_nodup;

  for (auto it = conv_candidate.begin(); it < conv_candidate.end(); it++) {
    bool if_same_conv = false;
    for (auto jt = conv_candi_nodup.begin(); jt < conv_candi_nodup.end(); jt++) {
      if_same_conv = mapComp(*it, *jt);
      if (if_same_conv) {
        LG << std::string(10, '-') << "Duplicate Convlutions Parameters" << std::string(10, '-');
        LG << "Input Shape: " << it->at("input");
        LG << "Weight Shpae: " << it->at("weight");
        LG << "Pad: " << it->at("pad");
        LG << "stride: " << it->at("stride");
        LG << "dilate: " << it->at("dilate");
        LG << "group: " << it->at("group");
        LG << std::string(30, '-');
        break;
      }
    }
    if (!if_same_conv) {
      conv_candi_nodup.emplace_back(*it);
    }
  }
  LG << "Convolution Candidates Candidates, Original: " << conv_candidate.size()
     << " after Deduplication: " << conv_candi_nodup.size();
  kwargs["conv"] = conv_candi_nodup;

  return std::make_tuple(layer_cnt, max_bs, conv_candi_nodup);
}

int main(int argc, char const *argv[]) {
  param_vec conv_candidate;
  int layer_cnt, max_bs;

  std::tie(layer_cnt, max_bs, conv_candidate) = LoadParamFile(argv[1]);
  Profiler<float> profiler(layer_cnt, max_bs);
  int cnt = 0;
  for (auto it = conv_candidate.begin(); it < conv_candidate.end(); it++) {
    it->operator[]("cudnn_tune") = "false";
    ConvolutionParam param;
    param.Init(*it);
    for (auto i = 2; i <= max_bs; i++) {
      profiler.Init(i, param, true);
      profiler.Profile();
      profiler.Clear();
    }
  }
  return 0;
}
