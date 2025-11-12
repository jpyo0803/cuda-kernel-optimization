#pragma once

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class Matmul2DBlockTiling : public MatmulBase {
 public:
  Matmul2DBlockTiling() = default;

  ~Matmul2DBlockTiling() override;

  void ComputeCore(int M, int K, int N) override;
};

}  // namespace jpyo0803