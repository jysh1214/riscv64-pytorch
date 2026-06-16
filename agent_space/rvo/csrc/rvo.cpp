// RVO: the whole native backend in one file. READING SKELETON, not a standalone
// compile target -- the bodies marked "... // see <file>" are copied verbatim from
// torch_openreg, whose paths are given so you can lift the real, compilable code.
// RVO's trick: "device memory" IS host memory, so allocate==malloc, copy==memcpy,
// and the ~10 core ops are one-liners. Everything else rides the CPU fallback.
//
// Minimum-effort build note: RVO uses torch.utils.cpp_extension (one .cpp, see
// setup.py) instead of openreg's CMake + two libraries -- the smallest path that works.

#include <torch/extension.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>

using c10::DeviceType;

// ---- 1. Device guard -------------------------------------------------------
// Drives "which device am I on / switch / streams". RVO is single-device, so
// most methods are trivial. Real, complete body: torch_openreg/csrc/runtime/OpenRegGuard.h
struct RVOGuard final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::PrivateUse1;
  RVOGuard() = default;
  explicit RVOGuard(DeviceType t) { TORCH_CHECK(t == static_type); }  // as OpenReg/CUDA do
  DeviceType type() const override { return static_type; }
  c10::Device getDevice() const override { return c10::Device(static_type, 0); }
  c10::Device exchangeDevice(c10::Device) const override { return getDevice(); }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  // getStream/exchangeStream: return a default stream; event methods inherit the
  // base's throwing defaults (fine until you use streams). // see OpenRegGuard.h
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream exchangeStream(c10::Stream) const noexcept override { return getStream(getDevice()); }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, RVOGuard);   // <-- drops RVOGuard into the guard table

// ---- 2. Allocator (device memory == host memory) ---------------------------
// Same base as OpenReg/CUDA: c10::DeviceAllocator. The caching/stats methods below
// are no-op stubs; OpenReg/CUDA put real per-device caching + stream-aware reuse
// there. // see OpenRegDeviceAllocator.cpp for the full subclass.
static void rvo_free(void* p) { free(p); }
struct RVOAllocator final : public c10::DeviceAllocator {
  c10::DataPtr allocate(size_t n) override {
    void* p = n ? malloc(n) : nullptr;                 // a real driver would call its malloc
    return {p, p, &rvo_free, c10::Device(DeviceType::PrivateUse1, 0)};
  }
  c10::DeleterFnPtr raw_deleter() const override { return &rvo_free; }
  void copy_data(void* dst, const void* src, size_t n) const override { memcpy(dst, src, n); }
  // DeviceAllocator pure virtuals -- minimal no-op stubs (no caching, no stats):
  bool initialized() override { return true; }
  void emptyCache(c10::MempoolId_t = {0, 0}) override {}
  void recordStream(const c10::DataPtr&, c10::Stream) override {}
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex) override { return {}; }
  void resetAccumulatedStats(c10::DeviceIndex) override {}
  void resetPeakStats(c10::DeviceIndex) override {}
};
static RVOAllocator g_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_alloc);  // <-- drops it into the allocator table

// ---- 3. Hooks (answers generic questions; lazily throws if not overridden) --
struct RVOHooks : public at::PrivateUse1HooksInterface {
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  // getDefaultGenerator/getPinnedMemoryAllocator/isPinnedPtr: override as needed.
  // see torch_openreg/csrc/runtime/OpenRegHooks.h
};
static bool _hooks = [] {
  at::RegisterPrivateUse1HooksInterface(new RVOHooks());   // <-- exactly once, heap-allocated
  return true;
}();

// ---- 4. Core operators: ~10 byte/layout ops, then a catch-all fallback -----
namespace {
// empty.memory_format: allocate device bytes via our allocator.
// see torch_openreg/csrc/aten/native/Minimal.cpp:7 for the real wrapper.
at::Tensor rvo_empty_memory_format(c10::SymIntArrayRef size, std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>,
    std::optional<c10::MemoryFormat>) {
  return at::detail::empty_generic(
      C10_AS_INTARRAYREF_SLOW(size), &g_alloc,
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      dtype.value_or(at::kFloat), std::nullopt);
}
// _copy_from: the load-bearing one (powers .copy_, .to, and the fallback round-trip).
// Since device memory == host memory, this is a strided memcpy. // see Minimal.cpp:81
at::Tensor rvo__copy_from(const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
  /* memcpy self -> dst honoring dtype/contiguity ... // see OpenReg _copy_from */ return dst;
}
// The CPU fallback: move inputs to CPU, run the op there, copy results back.
void rvo_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}
}  // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {        // the ~10 core ops you must write
  m.impl("empty.memory_format", rvo_empty_memory_format);
  m.impl("_copy_from", rvo__copy_from);
  // empty_strided, as_strided, view, _reshape_alias, resize_, _local_scalar_dense,
  // _copy_from_and_resize, set_.source_Storage(_storage_offset) ... // see OpenRegMinimal.cpp:119
}
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {           // the wildcard "_" = every other op
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&rvo_cpu_fallback>());
}

// ---- 5. The _C module: device-runtime helpers the Python module forwards to -
PYBIND11_MODULE(_C, m) {
  m.def("get_device_count", [] { return 1; });
  m.def("get_device", [] { return 0; });
  m.def("set_device", [](int) {});
}
