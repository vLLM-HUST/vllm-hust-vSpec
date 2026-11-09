// Native convolution arithmetic/events/writeback unchanged; five input address
// calculations use the real physical row stride. Private standalone ABI only.
#include <cstdio>
#include "native-kernel/causal_conv1d_tiling_key.h"
#include "native-kernel/causal_conv1d_update.h"
using namespace AscendC;
using namespace NsCausalConv1d;

extern "C" __global__ __aicore__ void vspec_conv_native_stride(
    GM_ADDR output, GM_ADDR value, GM_ADDR weight, GM_ADDR state,
    GM_ADDR queries, GM_ADDR indices, GM_ADDR accepted, GM_ADDR counters,
    uint32_t tokens, uint32_t batch, uint32_t cache_lines, uint32_t index_stride,
    uint32_t tile, uint32_t flags, uint32_t input_stride, uint32_t state_len)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CausalConv1dTilingData tiling{};
    tiling.dim = 4096;
    tiling.vspecInputRowStride = input_stride;
    tiling.cuSeqlen = tokens;
    tiling.seqLen = 0;
    tiling.inputMode = 0;
    tiling.width = 4;
    tiling.stateLen = state_len;
    tiling.numCacheLines = cache_lines;
    tiling.batch = batch;
    tiling.activationMode = 1;
    tiling.padSlotId = -1;
    tiling.hasBias = 0;
    tiling.baseDim = tile;
    tiling.baseDimCnt = (4096 + tile - 1) / tile;
    tiling.hasNumAcceptedTokens = 1;
    tiling.hasQueryStartLoc = 1;
    tiling.hasCacheIndices = 1;
    tiling.cacheIndicesStride = index_stride;
    tiling.queryStartLocUseInt64 = flags & 1;
    tiling.cacheIndicesUseInt64 = (flags >> 1) & 1;
    tiling.numAcceptedTokensUseInt64 = (flags >> 2) & 1;
    RunCausalConv1dUpdate<bfloat16_t>(value, weight, nullptr, state, queries,
        indices, nullptr, accepted, output, nullptr, &tiling);
    if (GetBlockIdx() == 0) {
        // The original update object and its TPipe have already finished.
        // Fold a real replay witness into this same launch, with integer
        // atomic DMA instead of cached scalar GM reads/writes.
        TPipe counterPipe;
        TBuf<TPosition::VECCALC> counterBuffer;
        counterPipe.InitBuffer(counterBuffer, 32);
        LocalTensor<int32_t> one = counterBuffer.Get<int32_t>();
        Duplicate(one, static_cast<int32_t>(1), 8);
        GlobalTensor<int32_t> counts;
        counts.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(counters));
        TEventID ready = counterPipe.FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        uint32_t index = batch + ((tokens / batch == 7) ? 17 : 0);
        SetAtomicAdd<int32_t>();
        DataCopyParams params{1, 4, 0, 0};
        DataCopyPad(counts[index], one, params);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void launch_vspec_conv_native_stride(
    void* stream, void* output, void* value, void* weight, void* state,
    void* queries, void* indices, void* accepted, void* counters, uint32_t tokens, uint32_t batch,
    uint32_t cache_lines, uint32_t index_stride, uint32_t tile, uint32_t flags,
    uint32_t cores, uint32_t input_stride, uint32_t state_len)
{
    uint32_t grid = batch * ((4096 + tile - 1) / tile);
    uint32_t blocks = grid < cores ? grid : cores;
    vspec_conv_native_stride<<<blocks, nullptr, stream>>>((uint8_t*)output, (uint8_t*)value,
        (uint8_t*)weight, (uint8_t*)state, (uint8_t*)queries, (uint8_t*)indices,
        (uint8_t*)accepted, (uint8_t*)counters, tokens, batch, cache_lines, index_stride, tile, flags, input_stride, state_len);
}
