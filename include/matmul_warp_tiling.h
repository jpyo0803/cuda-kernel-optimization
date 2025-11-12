#pragma once

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class MatmulWarpTiling : public MatmulBase {
 public:
  MatmulWarpTiling() = default;

  ~MatmulWarpTiling() override;

  void ComputeCore(int M, int K, int N) override;
};

}  // namespace jpyo0803