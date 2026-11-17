// Host-only batching of the SAME installed torch-npu native FIA function.
// No ACL tensor packing, raw device kernel arguments, approximate masks,
// alternative attention implementation, or graph ordering changes.
#include <torch/extension.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch_npu/csrc/core/npu/NPUGraph.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace py = pybind11;

bool valid_lengths(int64_t rows, const std::vector<int64_t>& qs,
                   const std::vector<int64_t>& ks)
{
    if (qs.size() != ks.size() || qs.empty() || qs.back() != rows) return false;
    bool admitted_rows = false;
    for (auto requests : {1, 2, 3, 14}) {
        for (auto width : {5, 7}) {
            if (rows == requests * width) admitted_rows = true;
        }
    }
    if (!admitted_rows) return false;
    if (qs.size() == static_cast<size_t>(rows)) {
        for (size_t i = 0; i < qs.size(); ++i) if (qs[i] != static_cast<int64_t>(i + 1)) return false;
        return true;
    }
    for (auto requests : {1, 2, 3, 14}) {
        for (auto width : {5, 7}) {
            if (rows != requests * width || qs.size() != static_cast<size_t>(requests)) continue;
            for (int i = 0; i < requests; ++i) if (qs[i] != (i + 1) * width) return false;
            return true;
        }
    }
    return false;
}

using TensorOption = const std::optional<at::Tensor>&;
using HostLengths = at::OptionalSymIntArrayRef;
// Exact signature reported by the installed runtime registration. Its
// .out schema takes Tensor[] and returns Tensor values, unlike the stale
// generated CustomFunctions.h wrapper bundled with this installation.
using FIAFunction = std::tuple<at::Tensor, at::Tensor>(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    TensorOption, TensorOption, HostLengths, HostLengths,
    TensorOption, TensorOption, TensorOption, TensorOption, TensorOption, TensorOption, TensorOption,
    TensorOption, TensorOption, TensorOption, TensorOption,
    TensorOption, TensorOption, TensorOption, TensorOption, TensorOption, HostLengths,
    TensorOption, TensorOption, TensorOption,
    int64_t, double, int64_t, int64_t, c10::string_view, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, bool, TensorOption, at::TensorList);

struct NativeFIA {
    at::Tensor query, key, value, output, lse;
    std::optional<at::Tensor> mask, workspace;
    int64_t heads, kv_heads, block_size, sparse, pre_tokens, next_tokens;
    double scale;
    c10::TypedOperatorHandle<FIAFunction> operation;
    std::array<at::Tensor, 2> outputs;

    NativeFIA(at::Tensor q, at::Tensor k, at::Tensor v,
              std::optional<at::Tensor> m, std::optional<at::Tensor> ws,
              at::Tensor out, at::Tensor softmax_lse, int64_t nheads,
              int64_t nkv, int64_t block, int64_t sparse_mode,
              int64_t pre, int64_t next, double factor)
        : query(q), key(k), value(v), output(out), lse(softmax_lse), mask(m), workspace(ws),
          heads(nheads), kv_heads(nkv), block_size(block), sparse(sparse_mode),
          pre_tokens(pre), next_tokens(next), scale(factor),
          operation(c10::Dispatcher::singleton().findSchemaOrThrow(
              "npu::npu_fused_infer_attention_score", "out").typed<FIAFunction>()),
          outputs{out, softmax_lse}
    {
        TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                    "Pinned unquantized BF16 attention only");
        TORCH_CHECK(q.dim() == 3 && q.size(1) == 8 && q.size(2) == 256 &&
                    k.dim() == 3 && k.size(1) == 128 && k.size(2) == 256 &&
                    v.sizes() == k.sizes() && out.sizes() == q.sizes(), "Pinned TND Q35 TP2 geometry");
        TORCH_CHECK(heads == 8 && kv_heads == 1 && block_size == 128 && sparse == 3,
                    "Pinned native causal TND paged attention only");
    }

    void invoke(const std::vector<int64_t>& query_lengths, const std::vector<int64_t>& kv_lengths,
                const at::Tensor& table)
    {
        TORCH_CHECK(valid_lengths(query.size(0), query_lengths, kv_lengths),
                    "Only exact N1/N2/N3/N14 MTP4/6 native and compact-padding lengths admitted");
        std::vector<c10::SymInt> qs, ks;
        for (auto x : query_lengths) qs.emplace_back(x);
        for (auto x : kv_lengths) ks.emplace_back(x);
        const auto none = std::nullopt;
        operation.call(
            query, key, value, none, mask,
            at::OptionalSymIntArrayRef(c10::SymIntArrayRef(qs)),
            at::OptionalSymIntArrayRef(c10::SymIntArrayRef(ks)),
            none, none, none, none, none, none, none,  // dequant1 through antiquant offset
            none, none, none, none,                  // K/V antiquant scale/offset
            table, none, none,                      // block table, Q/KV padding
            none, none, none,                       // shared prefixes and their lengths
            none, none, none,                       // Q/K rope and rope antiquant
            heads, scale, pre_tokens, next_tokens, "TND", kv_heads, sparse,
            0, block_size, 0, 0, 0, false, workspace, at::TensorList(outputs));
    }
};

struct NativeFIABatch {
    std::vector<std::shared_ptr<NativeFIA>> bindings;
    std::vector<c10_npu::NPUTaskGroupHandle> handles;
    std::vector<py::object> events;

    NativeFIABatch(std::vector<std::shared_ptr<NativeFIA>> ops, py::list hs, py::list es)
        : bindings(std::move(ops))
    {
        TORCH_CHECK((bindings.size() == 10 || bindings.size() == 4 || bindings.size() == 6) &&
                    hs.size() == bindings.size() && es.size() == bindings.size(),
                    "All original target10 or draft4/6 groups required");
        for (size_t i = 0; i < bindings.size(); ++i) {
            // This uses the installed, typed torch-npu public C++ handle.
            // Unknown pybind layouts fail here, before beginning an update.
            handles.push_back(hs[i].cast<c10_npu::NPUTaskGroupHandle>());
            events.push_back(py::reinterpret_borrow<py::object>(es[i]));
        }
    }

    void update(py::object python_stream, const std::vector<std::vector<int64_t>>& qs,
                const std::vector<std::vector<int64_t>>& ks, const std::vector<at::Tensor>& tables)
    {
        TORCH_CHECK(qs.size() == bindings.size() && ks.size() == bindings.size() &&
                    tables.size() == bindings.size(), "Exact per-task argument cardinality required");
        auto stream = c10_npu::getCurrentNPUStream();
        auto supplied = python_stream.attr("npu_stream").cast<uint64_t>();
        TORCH_CHECK(supplied == reinterpret_cast<uint64_t>(stream.stream()),
                    "Caller must retain the original update-stream context");
        for (size_t i = 0; i < bindings.size(); ++i) {
            c10_npu::graph_task_update_begin(stream, handles[i]);
            bindings[i]->invoke(qs[i], ks[i], tables[i]);
            c10_npu::graph_task_update_end(stream);
            // Retain the actual native ExternalEvent object and per-group
            // record position; no late batched event or removed wait.
            events[i].attr("record")(python_stream);
        }
    }
};

PYBIND11_MODULE(vspec_fia_host, module) {
    module.def("valid_lengths", &valid_lengths);
    py::class_<NativeFIA, std::shared_ptr<NativeFIA>>(module, "NativeFIA")
        .def(py::init<at::Tensor, at::Tensor, at::Tensor,
             std::optional<at::Tensor>, std::optional<at::Tensor>, at::Tensor, at::Tensor,
             int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, double>())
        .def("invoke", &NativeFIA::invoke);
    py::class_<NativeFIABatch>(module, "NativeFIABatch")
        .def(py::init<std::vector<std::shared_ptr<NativeFIA>>, py::list, py::list>())
        .def("update", &NativeFIABatch::update);
}
