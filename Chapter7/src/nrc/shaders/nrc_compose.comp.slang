// Compose pass (paper §3.5 "Amortization"): after cache-query has written
// g_nrcContrib for this frame's query records, sum it with the direct-
// lighting prefix the ray-gen shader wrote into g_pathPrefix, and deposit
// the accumulated result into the ray-tracer's output image.
//
// Frame accumulation: the host renderer accumulates sample history into
// `outImage` across frames. Ray-gen skips its imageStore when NRC is on;
// we perform the accumulation here so the per-frame sample (prefix +
// contrib) is correctly blended with history.
//
// Descriptor set 0:
//   binding 0 : RWStructuredBuffer<float4> g_pathPrefix  (ray-gen writes)
//   binding 1 : RWStructuredBuffer<float4> g_nrcContrib  (cache-query writes)
//   binding 2 : RWTexture2D<float4>        g_outImage    (gbuf, VK_FORMAT_R32G32B32A32_SFLOAT)

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> g_pathPrefix;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> g_nrcContrib;
[[vk::binding(2, 0)]] RWTexture2D<float4>        g_outImage;

struct Pc {
    uint imageWidth;
    uint imageHeight;
    uint numSamples;        // samples this frame (typically 1)
    uint totalSamples;      // samples accumulated before this frame
    uint firstFrameOrReset; // 1 => no history to blend with (replace)
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]] Pc pc;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.imageWidth || tid.y >= pc.imageHeight) return;
    uint pixelIdx = tid.y * pc.imageWidth + tid.x;

    float4 prefix  = g_pathPrefix[pixelIdx];
    float3 contrib = g_nrcContrib[pixelIdx].xyz;

    // This-frame sample = direct prefix (ray-gen output, pre-contrib) plus
    // the same-frame MLP contribution. Preserve the prefix's alpha (ray-gen
    // writes alpha=1 for hit pixels, alpha=0 for sky via the `solid` branch).
    float4 sampleValue = float4(prefix.xyz + contrib, prefix.w);

    // Guard against NaN/Inf from an early-frame under-trained MLP or degenerate paths.
    // Without this, a single bad sample poisons the accumulation history forever.
    if (any(isnan(sampleValue)) || any(isinf(sampleValue))) {
        sampleValue = float4(0.0);
    }

    int2 pixel = int2(tid.xy);

    if (pc.firstFrameOrReset != 0u || pc.totalSamples == 0u) {
        g_outImage[pixel] = sampleValue;
    } else {
        float4 old = g_outImage[pixel];
        float  tA  = float(pc.totalSamples);
        float  tB  = float(pc.numSamples);
        g_outImage[pixel] = (old * tA + sampleValue * tB) / (tA + tB);
    }
}
