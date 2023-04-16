import types, gc, os, time, re
import paddle
from paddle.nn import functional as F

current_path = os.path.dirname(os.path.abspath(__file__))

from paddle.utils.cpp_extension import CUDAExtension, setup

# python -m rwkv_paddle.setup install
# todo: we need move wkv_cuda.so to wkv_cuda_pd_.so
setup(
    name='wkv_cuda',
    ext_modules=CUDAExtension(
        sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
        verbose=True,
        extra_cuda_cflags=["-t 4", "-std=c++17", "--use_fast_math", "-O3", "--extra-device-vectorization"]
    )
)
