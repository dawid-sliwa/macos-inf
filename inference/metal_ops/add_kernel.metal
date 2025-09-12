#include <metal_stdlib>
using namespace metal;

// Elementwise add: out[i] = a[i] + b[i] for i in [0, N)
kernel void add2(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float*       out [[ buffer(2) ]],
    constant uint&      N [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]) {

    if (gid >= N) return;
    out[gid] = a[gid] + b[gid];
}
