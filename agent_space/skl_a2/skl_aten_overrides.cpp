#include <torch/library.h>
#include <ATen/ATen.h>
#include <cstdio>

// stand-in for the SKL vendor gemm: naive loops, NO at::mm anywhere (recursion!)
static void skl_sgemm(const float* a, const float* b, float* c,
                      int64_t m, int64_t k, int64_t n) {
  std::fprintf(stderr, "[SKL] sgemm %ldx%ldx%ld\n", (long)m, (long)k, (long)n);
  for (int64_t i = 0; i < m; i++)
    for (int64_t j = 0; j < n; j++) {
      float acc = 0.f;
      for (int64_t p = 0; p < k; p++) acc += a[i*k+p] * b[p*n+j];
      c[i*n+j] = acc;
    }
}

static at::Tensor& skl_mm_out(const at::Tensor& self, const at::Tensor& mat2, at::Tensor& out) {
  TORCH_CHECK(self.dim()==2 && mat2.dim()==2, "skl_mm_out: 2-D only");
  TORCH_CHECK(self.scalar_type()==at::kFloat, "skl_mm_out: float32 only (demo)");
  auto a = self.contiguous(); auto b = mat2.contiguous();
  out.resize_({a.size(0), b.size(1)});
  skl_sgemm(a.const_data_ptr<float>(), b.const_data_ptr<float>(),
            out.mutable_data_ptr<float>(), a.size(0), a.size(1), b.size(1));
  return out;
}

static at::Tensor skl_mm(const at::Tensor& self, const at::Tensor& mat2) {
  auto out = at::empty({self.size(0), mat2.size(1)}, self.options());
  return skl_mm_out(self, mat2, out);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("mm",     skl_mm);       // eager a @ b
  m.impl("mm.out", skl_mm_out);   // extern_kernels.mm path
}
