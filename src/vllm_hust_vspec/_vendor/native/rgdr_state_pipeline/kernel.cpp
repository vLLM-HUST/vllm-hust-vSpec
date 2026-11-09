// Private direct-launch prototype, reusing the original native arithmetic.
#include "rgdr_full128.h"
using namespace AscendC;
using namespace RecurrentGatedDeltaRule;

extern "C" __global__ __aicore__ void vspec_rgdr_full128(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR beta, GM_ADDR state,
    GM_ADDR lengths, GM_ADDR indices, GM_ADDR g, GM_ADDR accepted, GM_ADDR out,
    uint32_t tokens, uint32_t cores, uint32_t v_step, uint32_t output_buffers)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    RecurrentGatedDeltaRuleTilingData tiling{};
    tiling.vectorCoreNum = cores;
    tiling.ubCalSize = 196608;
    tiling.ubRestBytes = 24576;
    tiling.t = tokens;
    tiling.nk = 8;
    tiling.dk = 128;
    tiling.nv = 16;
    tiling.dv = 128;
    tiling.b = 14;
    tiling.vStep = v_step;
    tiling.stateOutBufferNum = output_buffers;
    tiling.attnOutBufferNum = 1;
    tiling.scale = 0.08838834764831844055f;
    tiling.hasGama = 1;
    tiling.hasGamaK = 0;
    tiling.hasAcceptedTokens = 1;
    TPipe pipe;
    RGDR<bfloat16_t, bfloat16_t, float> op(&tiling);
    RGDRInitParams params{query, key, value, g, nullptr, beta, state, lengths,
                          indices, accepted, out, state};
    op.Init(params, &pipe);
    op.Process();
}

extern "C" void launch_vspec_rgdr_full128(
    void* stream, void* query, void* key, void* value, void* beta, void* state,
    void* lengths, void* indices, void* g, void* accepted, void* out,
    uint32_t tokens, uint32_t cores, uint32_t v_step, uint32_t output_buffers)
{
    vspec_rgdr_full128<<<cores, nullptr, stream>>>(
        (uint8_t*)query, (uint8_t*)key, (uint8_t*)value, (uint8_t*)beta, (uint8_t*)state,
        (uint8_t*)lengths, (uint8_t*)indices, (uint8_t*)g, (uint8_t*)accepted, (uint8_t*)out,
        tokens, cores, v_step, output_buffers);
}
