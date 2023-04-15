#include <paddle/extension.h>
#include <paddle/phi/backends/gpu/gpu_info.h>
// #include <paddle/utils/pybind.h>
#include <cuda_fp16.h>
#include <iostream>

typedef __half fp16;

// cast const data pointer to mutable data pointer
template <typename data_t>
data_t* ptr(const paddle::Tensor &tensor)
{
    return const_cast<data_t*>(tensor.data<data_t>());
}
template <>
fp16* ptr(const paddle::Tensor &tensor)
{
    return reinterpret_cast<fp16*>(const_cast<phi::dtype::float16*>(tensor.data<phi::dtype::float16>()));
}
template <typename data_t>
data_t* ptr2(paddle::Tensor &tensor)
{
    return tensor.data<data_t>();
}
template <>
fp16* ptr2(paddle::Tensor &tensor)
{
    return reinterpret_cast<fp16*>(tensor.data<phi::dtype::float16>());
}

void cuda_wkv_forward(int B, int T, int C, float *w, float *u, fp16 *k, fp16 *v, fp16 *y, float *aa, float *bb, float *pp);
void cuda_mm8_seq(int B, int N, int M,
                  fp16 *x, int x_stride,
                  uint8_t *w, int w_stride,
                  fp16 *mx, fp16 *rx,
                  fp16 *my, fp16 *ry,
                  fp16 *y, int y_stride);
void cuda_mm8_one(int N, int M,
                  fp16 *x,
                  uint8_t *w, int w_stride,
                  fp16 *mx, fp16 *rx,
                  fp16 *my, fp16 *ry,
                  float *y);

std::vector<paddle::Tensor> wkv_forward(const paddle::Tensor &w, const paddle::Tensor &u, const paddle::Tensor &k, const paddle::Tensor &v, const paddle::Tensor &aa, const paddle::Tensor &bb, const paddle::Tensor &pp, int64_t B, int64_t T, int64_t C) {
    const phi::backends::gpu::GPUDeviceGuard device_guard(w.place().GetDeviceId());
    auto y = paddle::empty({T, C}, paddle::DataType::FLOAT16, w.place());
    cuda_wkv_forward(B, T, C, ptr<float>(w), ptr<float>(u), ptr<fp16>(k), ptr<fp16>(v), ptr2<fp16>(y), ptr<float>(aa), ptr<float>(bb), ptr<float>(pp));
    return {y};
}
std::vector<paddle::Tensor> mm8_seq(const paddle::Tensor &x, const paddle::Tensor &w,
             const paddle::Tensor &mx, const paddle::Tensor &rx,
             const paddle::Tensor &my, const paddle::Tensor &ry,
             int64_t B, int64_t N, int64_t M) {
    auto y = paddle::empty({B, M}, paddle::DataType::FLOAT16, w.place());
    assert(x.shape()[1] == 1);
    assert(w.shape()[1] == 1);
    assert(mx.shape()[0] == 1 && rx.shape()[0] == 1);
    assert(my.shape()[0] == 1 && ry.shape()[0] == 1);
    assert(y.shape()[1] == 1);
    const phi::backends::gpu::GPUDeviceGuard device_guard(w.place().GetDeviceId());
    cuda_mm8_seq(
        B, N, M,
        ptr<fp16>(x), x.shape()[0],
        ptr<uint8_t>(w), w.shape()[0],
        ptr<fp16>(mx), ptr<fp16>(rx),
        ptr<fp16>(my), ptr<fp16>(ry),
        ptr2<fp16>(y), y.shape()[0]);
    return {y};
}
std::vector<paddle::Tensor> mm8_one(const paddle::Tensor &x, const paddle::Tensor &w,
             const paddle::Tensor &mx, const paddle::Tensor &rx,
             const paddle::Tensor &my, const paddle::Tensor &ry,
             int64_t N, int64_t M) {
    auto y = paddle::zeros({M}, paddle::DataType::FLOAT32, w.place());
    assert(x.shape()[0] == 1);
    assert(w.shape()[1] == 1);
    assert(mx.shape()[0] == 1 && rx.shape()[0] == 1);
    assert(my.shape()[0] == 1 && ry.shape()[0] == 1);
    assert(y.shape()[0] == 1);
    const phi::backends::gpu::GPUDeviceGuard device_guard(w.place().GetDeviceId());
    cuda_mm8_one(
        N, M,
        ptr<fp16>(x),
        ptr<uint8_t>(w), w.shape()[0],
        ptr<fp16>(mx), ptr<fp16>(rx),
        ptr<fp16>(my), ptr<fp16>(ry),
        ptr2<float>(y));
    return {y};
}

// PYBIND11_MODULE(PADDLE_EXTENSION_NAME, m) {
//     m.def("wkv_forward", &wkv_forward, "wkv forward");
//     m.def("mm8_seq", &mm8_seq, "mm8 seq");
//     m.def("mm8_one", &mm8_one, "mm8 one");
// }

std::vector<std::vector<int64_t>> wkv_forward_shape(std::vector<int64_t> w, std::vector<int64_t> u, std::vector<int64_t> k, std::vector<int64_t> v, std::vector<int64_t> aa, std::vector<int64_t> bb, std::vector<int64_t> pp, int64_t B, int64_t T, int64_t C) {
  return {{T, C}};
}
std::vector<paddle::DataType> wkv_forward_dtype(paddle::DataType w, paddle::DataType u, paddle::DataType k, paddle::DataType v, paddle::DataType aa, paddle::DataType bb, paddle::DataType pp) {
  return {paddle::DataType::FLOAT16};
}
std::vector<std::vector<int64_t>> mm8_seq_shape(std::vector<int64_t> x, std::vector<int64_t> w,
             std::vector<int64_t> mx, std::vector<int64_t> rx,
             std::vector<int64_t> my, std::vector<int64_t> ry,
             int64_t B, int64_t N, int64_t M) {
    return {{B, M}};
}
std::vector<paddle::DataType> mm8_seq_dtype(paddle::DataType x, paddle::DataType w,
             paddle::DataType mx, paddle::DataType rx,
             paddle::DataType my, paddle::DataType ry) {
    return {paddle::DataType::FLOAT16};
}
std::vector<std::vector<int64_t>> mm8_one_shape(std::vector<int64_t> x, std::vector<int64_t> w,
             std::vector<int64_t> mx, std::vector<int64_t> rx,
             std::vector<int64_t> my, std::vector<int64_t> ry,
             int64_t N, int64_t M) {
    return {{M}};
}
std::vector<paddle::DataType> mm8_one_dtype(paddle::DataType x, paddle::DataType w,
             paddle::DataType mx, paddle::DataType rx,
             paddle::DataType my, paddle::DataType ry) {
    return {paddle::DataType::FLOAT32};
}

PD_BUILD_OP(wkv_forward)
    .Inputs({"w", "u", "k", "v", "aa", "bb", "pp"})
    .Outputs({"Out"})
    .Attrs({"b: int64_t", "t: int64_t", "c: int64_t"})
    .SetKernelFn(PD_KERNEL(wkv_forward))
    .SetInferShapeFn(PD_INFER_SHAPE(wkv_forward_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(wkv_forward_dtype));
PD_BUILD_OP(mm8_seq)
    .Inputs({"x", "w", "mx", "rx", "my", "ry"})
    .Outputs({"Out"})
    .Attrs({"b: int64_t", "n: int64_t", "m: int64_t"})
    .SetKernelFn(PD_KERNEL(mm8_seq))
    .SetInferShapeFn(PD_INFER_SHAPE(mm8_seq_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(mm8_seq_dtype));
PD_BUILD_OP(mm8_one)
    .Inputs({"x", "w", "mx", "rx", "my", "ry"})
    .Outputs({"Out"})
    .Attrs({"n: int64_t", "m: int64_t"})
    .SetKernelFn(PD_KERNEL(mm8_one))
    .SetInferShapeFn(PD_INFER_SHAPE(mm8_one_shape))
    .SetInferDtypeFn(PD_INFER_DTYPE(mm8_one_dtype));
