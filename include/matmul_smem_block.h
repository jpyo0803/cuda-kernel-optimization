#pragma once

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class MatmulSmemBlock : public MatmulBase {
 public:
  MatmulSmemBlock() = default;

  ~MatmulSmemBlock() override;

  void ComputeCore(int M, int K, int N) override;
};

}  // namespace jpyo0803