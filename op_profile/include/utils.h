/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: utils.hpp
 * \brief: common used headers
 * Created Date: Monday, June 15th 2020, 11:42:05 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Wednesday, July 8th 2020, 5:09:08 pm
 * Modified By: raphael hao
 */

#pragma once

#include <assert.h>
#include <dmlc/logging.h>

template <typename ContainerType>
bool mapComp(ContainerType const &lhs, ContainerType const &rhs) {
  return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
}