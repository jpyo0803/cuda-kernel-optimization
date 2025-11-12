#pragma once

#include <iostream>

#include "matmul_base.h"

namespace jpyo0803 {

class MatmulCoalescing : public MatmulBase {
 public:
  MatmulCoalescing() = default;

  ~MatmulCoalescing() override;

  void ComputeCore(int M, int K, int N) override;
};

}  // namespace jpyo0803