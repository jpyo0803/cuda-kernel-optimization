#pragma once

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class Matmul1DBlockTiling : public MatmulBase {
 public:
  Matmul1DBlockTiling() = default;

  ~Matmul1DBlockTiling() override;

  void ComputeCore(int M, int K, int N) override;
};

}  // namespace jpyo0803