from torch.utils.cpp_extension import load
import pathlib

_THIS_DIR = pathlib.Path(__file__).parent


def _build():
    name = "metal_ops_ext"
    srcs = [str(_THIS_DIR / "ops.mm")]
    extra_cflags = ["-std=gnu++17", "-fno-objc-arc"]
    extra_ldflags = ["-framework", "Metal", "-framework", "Foundation"]
    mod = load(
        name=name,
        sources=srcs,
        extra_cflags=extra_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=False,
        verbose=True,
    )
    return mod


try:
    _ext = _build()
except Exception as e:
    raise RuntimeError(f"Failed to build Metal extension: {e}")


# Python wrappers (export names you want to use)
def add_kernel(a, b):
    return _ext.add2_mps(a, b)
