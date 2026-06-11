// SKL extension glue — the analog of aten/src/ATen/native/mkldnn/Linear.cpp.
// In production this links against libskl.so; here skl_sgemm is a stand-in.
#include <torch/library.h>
#include <ATen/ATen.h>

// ---- stand-in for the vendor library (swap point: declare the real symbol,
// ---- delete this, and link with -lskl) ----
static void skl_sgemm(const float* a, const float* b, float* c,
                      int64_t m, int64_t k, int64_t n) {
  for (int64_t i = 0; i < m; i++)
    for (int64_t j = 0; j < n; j++) {
      float acc = 0.f;
      for (int64_t p = 0; p < k; p++) acc += a[i*k+p] * b[p*n+j];
      c[i*n+j] = acc;
    }
}

static at::Tensor skl_mm_cpu(const at::Tensor& a_, const at::Tensor& b_) {
  TORCH_CHECK(a_.dim() == 2 && b_.dim() == 2, "skl::mm: 2-D only");
  TORCH_CHECK(a_.scalar_type() == at::kFloat, "skl::mm: float32 only (demo)");
  auto a = a_.contiguous(); auto b = b_.contiguous();
  auto out = at::empty({a.size(0), b.size(1)}, a.options());
  skl_sgemm(a.const_data_ptr<float>(), b.const_data_ptr<float>(),
            out.mutable_data_ptr<float>(), a.size(0), a.size(1), b.size(1));
  return out;
}

// Meta kernel: shape inference for torch.compile (the C++ analog of register_fake)
static at::Tensor skl_mm_meta(const at::Tensor& a, const at::Tensor& b) {
  return at::empty({a.size(0), b.size(1)}, a.options());
}

TORCH_LIBRARY(skl, m) {
  m.def("mm(Tensor a, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(skl, CPU, m)  { m.impl("mm", skl_mm_cpu); }
TORCH_LIBRARY_IMPL(skl, Meta, m) { m.impl("mm", skl_mm_meta); }
