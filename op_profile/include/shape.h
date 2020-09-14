/*
 * Copyright (c) 2020 by Mogic
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * --------
 * \file: shape.hpp
 * \brief: tensor shape implementaion
 * Created Date: Sunday, June 14th 2020, 10:57:16 pm
 * Author: raphael hao
 * Email: raphaelhao@outlook.com
 * --------
 * Last Modified: Monday, September 14th 2020, 2:09:52 pm
 * Modified By: raphael hao
 */

#pragma once
#include <dmlc/logging.h>
#include <algorithm>

template <typename ValueType>
class Tuple {
 public:
  Tuple() = default;

  ~Tuple() {}

  inline Tuple(const int ndim, const int value) {
    this->SetDim(ndim);
    if (ndim > 0) {
      std::fill_n(begin(), ndim, value);
    }
  }

  inline Tuple(const Tuple<ValueType>& s) {
    if (s.ndim_ == -1) {
      this->SetDim(-1);
    } else {
      this->assign(s.begin(), s.end());
    }
  }

  inline Tuple(std::initializer_list<ValueType> init) {
    this->assign(init.begin(), init.end());
  }

  inline Tuple(std::vector<ValueType> init) {
    this->assign(init.begin(), init.end());
  }

  inline Tuple(Tuple<ValueType>&& src) {
    this->swap(src);
  }

  template <typename RandomAccessIterator>
  inline Tuple(RandomAccessIterator begin, RandomAccessIterator end) {
    this->assign(begin, end);
  }

  template <typename RandomAccessIterator>
  inline void assign(RandomAccessIterator begin, RandomAccessIterator end) {
    this->SetDim(end - begin);
    CHECK_GE(ndim(), 0);
    std::copy(begin, end, this->begin());
  }

  inline void swap(Tuple<ValueType>& other) {  // NOLINT(*)
    std::swap(ndim_, other.ndim_);
    std::swap(data_stack_, other.data_stack_);
  }

  inline Tuple<ValueType>& operator=(const Tuple<ValueType>& src) {
    if (src.ndim() == -1) {
      this->SetDim(-1);
    } else {
      this->assign(src.begin(), src.end());
    }
    return *this;
  }

  inline Tuple<ValueType>& operator=(Tuple<ValueType>&& src) {
    Tuple<ValueType>(std::move(src)).swap(*this);
    return *this;
  }

  inline Tuple<ValueType>& operator=(std::initializer_list<ValueType> init) {
    this->assign(init.begin(), init.end());
    return *this;
  }

  inline bool operator==(const Tuple<ValueType>& s) const {
    if (ndim_ != s.ndim_)
      return false;
    if (ndim() == -1)
      return true;
    return std::equal(begin(), end(), s.begin());
  }

  inline bool operator!=(const Tuple<ValueType>& s) const {
    return !(*this == s);
  }

  inline ValueType* begin() {
    return data_stack_;
  }

  inline const ValueType* begin() const {
    return data_stack_;
  }

  inline ValueType* end() {
    return data_stack_ + ndim_;
  }

  inline const ValueType* end() const {
    return data_stack_ + ndim_;
  }

  inline int ndim() const {
    return ndim_;
  }

  inline ValueType& operator[](int i) {
// it fixes the false alarm of assuming signed overflow does not occur
// when assuming that (X - c) > X is always false [-Werror=strict-overflow]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
    CHECK(i >= 0 && i < ndim()) << "index = " << i << " must be in range [0, " << ndim() << ")";
#pragma GCC diagnostic pop
    return begin()[i];
  }

  inline const ValueType& operator[](int i) const {
// it fixes the false alarm of assuming signed overflow does not occur
// when assuming that (X - c) > X is always false [-Werror=strict-overflow]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-overflow"
    CHECK(i >= 0 && i < ndim()) << "index = " << i << " must be in range [0, " << ndim() << ")";
#pragma GCC diagnostic pop
    return begin()[i];
  }

  friend std::ostream& operator<<(std::ostream& os, const Tuple<ValueType>& t) {
    if (t.ndim() == -1) {
      // If t is an unknown shape, return string "None".
      // This is consistent with returning unknown shape in Python and generating
      // C++ operator APIs by OpWrapperGenerator.py (defaultString) in cpp-package.
      os << "None";
      return os;
    }
    os << '[';
    const ValueType* begin = t.begin();
    const ValueType* end = t.end();
    for (const ValueType* it = begin; it != end; ++it) {
      if (it != begin)
        os << ',';
      os << *it;
    }
    os << ']';
    return os;
  }

  friend std::istream& operator>>(std::istream& is, Tuple<ValueType>& t) {
    while (true) {
      char ch = is.peek();
      if (isdigit(ch) || ch == '-') {
        ValueType idx;
        if (is >> idx) {
          t.assign(&idx, &idx + 1);
        }
        return is;
      }
      is.get();
      if (ch == '(' || ch == '[')
        break;
      if (!isspace(ch)) {
        if (ch == 'N') {
          std::string tmp_val;
          is >> tmp_val;
          if (tmp_val == "one") {  // is stores "None"
            t.SetDim(-1);
            return is;
          }
        }
        is.setstate(std::ios::failbit);
        return is;
      }
    }
    while (isspace(is.peek())) {
      is.get();
    }
    if (is.peek() == ')' || is.peek() == ']') {
      is.get();
      t.SetDim(0);
      return is;
    }
    // Handle non-empty tuple
    ValueType idx;
    std::vector<ValueType> tmp;
    while (is >> idx) {
      tmp.push_back(idx);
      char ch;
      do {
        ch = is.get();
      } while (isspace(ch));
      if (std::is_integral<ValueType>::value && ch == 'L') {
        ch = is.get();
      }
      if (ch == ',') {
        while (true) {
          ch = is.peek();
          if (isspace(ch)) {
            is.get();
            continue;
          }
          if (ch == ')' || ch == ']') {
            is.get();
            break;
          }
          break;
        }
        if (ch == ')' || ch == ']')
          break;
      } else if (ch == ')' || ch == ']') {
        break;
      } else {
        is.setstate(std::ios::failbit);
        return is;
      }
    }
    t.assign(tmp.begin(), tmp.end());
    return is;
  }

 private:
  static const int dimStackCache = 4;
  int ndim_{0};
  ValueType data_stack_[dimStackCache];
  inline void SetDim(int ndim) {
    CHECK_GE(ndim, -1) << "dimension should greater than 0, received" << ndim;
    ndim_ = ndim;
  }
};

template <typename SrcIter, typename DstIter>
inline DstIter ShapeTypeCast(const SrcIter begin, const SrcIter end, DstIter dst_begin) {
  typedef typename std::iterator_traits<SrcIter>::value_type SrcDType;
  typedef typename std::iterator_traits<DstIter>::value_type DstDType;
  auto cast = [](const SrcDType& dim) {
    return static_cast<DstDType>(dim);
  };
  return std::transform(begin, end, dst_begin, cast);
}


using dim_t = int64_t;
using TShape = Tuple<dim_t>;
using ShapeVector = std::vector<TShape>;