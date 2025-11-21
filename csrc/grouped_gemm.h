#include <torch/extension.h>

namespace grouped_gemm {

void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b);
	 
}  // namespace grouped_gemm

namespace grouped_gemm_sm89 {
void GroupedGemm_sm89(torch::Tensor a,
		torch::Tensor b,
		torch::Tensor c,
		torch::Tensor batch_sizes,
		bool trans_a, bool trans_b);

}  // namespace grouped_gemm_sm89