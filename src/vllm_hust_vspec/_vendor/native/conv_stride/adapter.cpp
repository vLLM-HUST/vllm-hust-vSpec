// Standalone, default-unloaded diagnostic; same native BF16 convolution.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <cstdint>

extern "C" void launch_vspec_conv_native_stride(void*, void*, void*, void*, void*, void*, void*, void*, void*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

at::Tensor project_strided(at::Tensor& output, const at::Tensor& value, const at::Tensor& weight,
    at::Tensor& state, const at::Tensor& queries, const at::Tensor& indices,
    const at::Tensor& accepted, at::Tensor& counters, int64_t tile)
{
    TORCH_CHECK(tile == 256 || tile == 512 || tile == 1024 || tile == 1376
        || tile == 1536 || tile == 2048 || tile == 4096, "Pinned aligned tile choices only");
    const int64_t batch = accepted.numel();
    TORCH_CHECK(accepted.dim() == 1 && batch >= 1 && batch <= 16, "Batch1..16 only");
    TORCH_CHECK(value.dim() == 2 && value.size(1) == 4096
        && (value.size(0) == batch * 5 || value.size(0) == batch * 7), "Pinned speculation geometry");
    TORCH_CHECK(output.sizes() == value.sizes() && weight.sizes() == at::IntArrayRef({4, 4096}), "Output/weight geometry");
    TORCH_CHECK(state.dim() == 3 && state.size(0) >= batch * 7 + 2
        && (state.size(1) == 9 || state.size(1) == 11) && state.size(2) == 4096, "Full native state geometry");
    TORCH_CHECK(queries.sizes() == at::IntArrayRef({batch + 1}) && indices.dim() == 2
        && indices.size(0) == batch && (indices.size(1) == 5 || indices.size(1) == 7 || indices.size(1) == 9), "Native metadata geometry");
    TORCH_CHECK(counters.sizes() == at::IntArrayRef({34}) && counters.scalar_type() == at::kInt
        && counters.is_contiguous() && counters.device() == value.device(), "Private replay witness geometry");
    TORCH_CHECK(value.stride(1) == 1 && (value.stride(0) == 4096 || value.stride(0) == 6144)
        && value.device() == output.device(), "Admitted dense or QKVZ split input row stride only");
    for (const auto& t : {output, weight, state, queries, indices, accepted}) {
        TORCH_CHECK(t.is_contiguous() && t.device() == value.device()
            && t.device().type() == c10::DeviceType::PrivateUse1, "Contiguous same-device NPU inputs only");
    }
    for (const auto& t : {output, value, weight, state}) {
        TORCH_CHECK(t.scalar_type() == at::kBFloat16, "Native BF16 only");
    }
    for (const auto& t : {queries, indices, accepted}) {
        TORCH_CHECK(t.scalar_type() == at::kInt || t.scalar_type() == at::kLong, "Native integer metadata only");
    }
    uint32_t flags = (queries.scalar_type() == at::kLong)
        | ((indices.scalar_type() == at::kLong) << 1) | ((accepted.scalar_type() == at::kLong) << 2);
    uint32_t tokens = value.size(0), n = batch, lines = state.size(0), stride = indices.stride(0);
    uint32_t input_stride = value.stride(0);
    uint32_t state_len = state.size(1);
    void* out = output.data_ptr(); void* x = value.data_ptr(); void* w = weight.data_ptr();
    void* s = state.data_ptr(); void* q = queries.data_ptr(); void* i = indices.data_ptr(); void* a = accepted.data_ptr();
    void* witness = counters.data_ptr();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand command;
    command.Name("vspec_native_conv_tile_diagnostic");
    command.SetCustomHandler([=]() -> int {
        int device = 0; int64_t cores = 0;
        TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "Cannot resolve current NPU");
        TORCH_CHECK(aclGetDeviceCapability(device, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores) == ACL_SUCCESS
            && cores > 0, "Cannot resolve vector core count");
        launch_vspec_conv_native_stride(stream, out, x, w, s, q, i, a, witness, tokens, n, lines, stride, tile, flags, cores, input_stride, state_len);
        return 0;
    });
    command.Run();
    return output;
}

TORCH_LIBRARY(vspec_conv_strided, library) {
    library.def("project(Tensor(b!) output, Tensor value, Tensor weight, Tensor(a!) state, Tensor queries, Tensor indices, Tensor accepted, Tensor(c!) counters, int tile) -> Tensor(b!)");
}
TORCH_LIBRARY_IMPL(vspec_conv_strided, PrivateUse1, library) { library.impl("project", &project_strided); }
