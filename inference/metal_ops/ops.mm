#include <torch/extension.h>
#include <ATen/native/mps/OperationUtils.h>  // mtl_setBuffer, mtl_dispatch1DJob, getMTLBufferStorage
#include <ATEN/mps/MPSStream.h>              // get_command_buffer(), commit()  (include if needed)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

using Tensor = at::Tensor;

static std::string load_metal_source(const char* filename) {
  NSString* folder = [[NSString stringWithUTF8String:__FILE__] stringByDeletingLastPathComponent];
  NSString* path = [folder stringByAppendingPathComponent:[NSString stringWithUTF8String:filename]];
  NSError* err = nil;
  NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
  if (err || !src) throw std::runtime_error(std::string("Failed to read ") + filename);
  return std::string([src UTF8String]);
}

struct Pipelines {
  id<MTLDevice> dev = nil;
  id<MTLLibrary> lib = nil;
  id<MTLComputePipelineState> pso_add2 = nil;
};
static Pipelines G;
static std::once_flag G_once;

static void ensure_pso() {
  std::call_once(G_once, []{
    G.dev = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(G.dev, "No Metal device");
    NSError* e = nil;
    auto src = load_metal_source("add_kernel.metal");
    G.lib = [G.dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:nil error:&e];
    TORCH_CHECK(G.lib, "Compile failed: ", e ? e.localizedDescription.UTF8String : "unknown");
    id<MTLFunction> fn = [G.lib newFunctionWithName:@"add2"];
    TORCH_CHECK(fn, "Kernel 'add2' not found");
    G.pso_add2 = [G.dev newComputePipelineStateWithFunction:fn error:&e];
    TORCH_CHECK(G.pso_add2, "PSO failed: ", e ? e.localizedDescription.UTF8String : "unknown");
  });
}

Tensor add2_mps(const Tensor& a, const Tensor& b) {
  TORCH_CHECK(a.device().is_mps() && b.device().is_mps(), "inputs must be on mps");
  TORCH_CHECK(a.scalar_type()==at::kFloat && b.scalar_type()==at::kFloat, "float32 only for demo");
  TORCH_CHECK(a.sizes()==b.sizes(), "shape mismatch");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "must be contiguous");

  ensure_pso();
  Tensor out = at::empty_like(a);
  const NSUInteger N = (NSUInteger)a.numel();

  id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
  TORCH_CHECK(cb, "no MPS command buffer");

  // Encode on PyTorch's MPS stream.
  auto dq = torch::mps::get_dispatch_queue();
  dispatch_sync(dq, ^{
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:G.pso_add2];

    // Bind A,B,Out as Metal buffers directly from Tensors:
    at::native::mps::mtl_setBuffer(enc, a, 0);
    at::native::mps::mtl_setBuffer(enc, b, 1);
    at::native::mps::mtl_setBuffer(enc, out, 2);
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:3];

    // 1D grid; helper picks a good threadgroup size:
    at::native::mps::mtl_dispatch1DJob(enc, G.pso_add2, N);
    [enc endEncoding];
    torch::mps::commit();  // trigger execution on MPS stream
  });

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add2_mps", &add2_mps, "out = a + b (Metal/MPS)");
}
