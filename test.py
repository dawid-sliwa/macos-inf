import torch
from inference.metal_ops import add_kernel



a = torch.randn(2,3,4, device="mps", dtype=torch.float32)
b = torch.randn(2,3,4, device="mps", dtype=torch.float32)

y = add_kernel(a, b)
torch.testing.assert_close(y, a + b)
print("ok", y.shape, y.device)