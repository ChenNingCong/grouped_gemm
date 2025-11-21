#include "grouped_gemm.h"

#include <torch/extension.h>

namespace grouped_gemm {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gmm", &GroupedGemm, "Grouped GEMM.");
  m.def("gmm_sm89", &grouped_gemm_sm89::GroupedGemm_sm89, "Grouped GEMM for sm89 architecture.");
};

}  // namespace grouped_gemm
