// Standalone torch operator. Not loaded or selected by any model experiment.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <cstdint>

extern "C" void launch_vspec_rgdr_pack6(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t);

at::Tensor full128(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                   const at::Tensor& beta, at::Tensor& state, const at::Tensor& lengths,
                   const at::Tensor& indices, const at::Tensor& g, const at::Tensor& accepted,
                   at::Tensor& counters, int64_t v_step, int64_t output_buffers)
{
    TORCH_CHECK(v_step == 128 && output_buffers == 1, "Original128 single-bank path with exact packed-delta FMA only");
    TORCH_CHECK(query.dim() == 3 && query.size(0) >= 7 && query.size(0) <= 112 && query.size(0) % 7 == 0, "Only B1..16 gamma6; gamma4/8 unchanged");
    const int64_t tokens = query.size(0);
    const int64_t requests = tokens / 7;
    TORCH_CHECK(query.sizes() == at::IntArrayRef({tokens, 8, 128}) && key.sizes() == query.sizes(), "Q/K geometry");
    TORCH_CHECK(value.sizes() == at::IntArrayRef({tokens, 16, 128}), "Value geometry");
    TORCH_CHECK(beta.sizes() == at::IntArrayRef({tokens, 16}) && g.sizes() == beta.sizes(), "Gate geometry");
    TORCH_CHECK(state.dim() == 4 && state.size(0) >= tokens && state.size(1) == 16 && state.size(2) == 128 && state.size(3) == 128, "State geometry");
    TORCH_CHECK(lengths.sizes() == at::IntArrayRef({requests + 1}) && accepted.sizes() == at::IntArrayRef({requests}) && indices.sizes() == at::IntArrayRef({tokens}), "Native metadata geometry");
    TORCH_CHECK(counters.sizes() == at::IntArrayRef({17}) && counters.scalar_type() == at::kInt, "Folded replay counters only");
    for (const auto& tensor : {query, key, value, beta, state, lengths, indices, g, accepted, counters}) {
        TORCH_CHECK(tensor.is_contiguous() && tensor.device() == query.device() && tensor.device().type() == c10::DeviceType::PrivateUse1, "Contiguous same-device NPU inputs only");
    }
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 && value.scalar_type() == at::kBFloat16 && beta.scalar_type() == at::kBFloat16, "BF16 inputs only");
    TORCH_CHECK(state.scalar_type() == at::kFloat && g.scalar_type() == at::kFloat, "FP32 state/gate only");
    TORCH_CHECK(lengths.scalar_type() == at::kInt && indices.scalar_type() == at::kInt && accepted.scalar_type() == at::kInt, "INT32 native metadata only");
    auto output = at::empty(value.sizes(), value.options());
    void* q = query.data_ptr(); void* k = key.data_ptr(); void* v = value.data_ptr();
    void* b = beta.data_ptr(); void* s = state.data_ptr(); void* l = lengths.data_ptr();
    void* i = indices.data_ptr(); void* gate = g.data_ptr(); void* a = accepted.data_ptr();
    void* out = output.data_ptr();
    void* witness = counters.data_ptr();
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand command;
    command.Name("vspec_rgdr_pack6_prototype");
    command.SetCustomHandler([=]() -> int {
        int device = 0;
        int64_t cores = 0;
        TORCH_CHECK(aclrtGetDevice(&device) == ACL_SUCCESS, "Cannot resolve current device");
        TORCH_CHECK(aclGetDeviceCapability(device, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &cores) == ACL_SUCCESS && cores > 0, "Cannot resolve vector core count");
        launch_vspec_rgdr_pack6(stream, q, k, v, b, s, l, i, gate, a, out, witness, tokens, cores, v_step, output_buffers);
        return 0;
    });
    command.Run();
    return output;
}

TORCH_LIBRARY(vspec_rgdr_pack6, library) {
    library.def("project(Tensor query, Tensor key, Tensor value, Tensor beta, Tensor(a!) state, Tensor lengths, Tensor indices, Tensor g, Tensor accepted, Tensor(b!) counters, int v_step, int output_buffers) -> Tensor");
}
TORCH_LIBRARY_IMPL(vspec_rgdr_pack6, PrivateUse1, library) {
    library.impl("project", &full128);
}
