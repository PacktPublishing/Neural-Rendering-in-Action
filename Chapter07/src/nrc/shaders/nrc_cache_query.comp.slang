// Cache-query compute pass.
//
// Reads each QueryRecord emitted by the ray-gen shader and writes a "cache
// contribution" into g_nrcContrib at record.payload.pixelIdx. The
// contribution is consumed by the compose pass in the same frame after ray
// tracing.
//
// Each query is evaluated with the current NRC MLP through the RTXNS-backed
// inference facade. The network predicts normalized reflected radiance; this
// pass applies diffuse+specular reflectance, multiplies by the stored path
// throughput, and stores the result for composition.
//
// Descriptor set 0:
//   binding 0 : RWStructuredBuffer<QueryRecord> g_queries
//   binding 1 : RWStructuredBuffer<uint>        g_queryCount
//   binding 2 : ByteAddressBuffer               g_inferenceWeights
//   binding 3 : RWStructuredBuffer<float4>      g_nrcContrib

import nrc;

[[vk::binding(0, 0)]] RWStructuredBuffer<nrc::QueryRecord> g_queries;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>        g_queryCount;
[[vk::binding(2, 0)]] ByteAddressBuffer               g_inferenceWeights;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4>      g_nrcContrib;

struct Pc {
    // Per-layer byte offsets into g_inferenceWeights. Five hidden layers plus
    // the output layer make these 6-element arrays: 12 uints = 48 B, plus four
    // uints = 64 B. Well within the 128 B Vulkan push-constant budget.
    uint matrixOffset[nrc::kNetworkLayerCount];
    uint biasOffset[nrc::kNetworkLayerCount];

    uint maxQueries;
    uint imageWidth;
    uint imageHeight;
    uint pad;
};
[[vk::push_constant]] Pc pc;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint nQueries = g_queryCount[0];
    // The append counter can exceed storage capacity; both limits are required.
    if (idx >= nQueries || idx >= pc.maxQueries) return;

    nrc::QueryRecord q = g_queries[idx];

    // Real MLP inference via the RTXNS facade.
    nrc::NrcVertexFeatures features = q.getFeatures();
    nrc::NrcNetwork network;
    float3 mlpOut = network.inference(
        g_inferenceWeights, pc.matrixOffset, pc.biasOffset, features);

    // reflectance factorization — paper §4.1 multiplies MLP output by
    // (α + β) before using as cache value. The MLP learns normalized
    // reflected radiance; scaling by (diffuseAlbedo + specularAlbedo) brings
    // it back to physical HDR magnitude.
    float3 reflectance = nrc::reflectanceFactor(q.features);
    float3 cacheRadiance = mlpOut * reflectance;

    float3 contribution = cacheRadiance * q.payload.pathThroughput;

    // Preserve HDR magnitude; clamp the final composed physical estimate.
    contribution = max(contribution, float3(0));
    if(any(isnan(contribution)) || any(isinf(contribution)))
        contribution = float3(0);

    uint pixelIdx = q.payload.pixelIdx;
    // The renderer enforces 1 spp and at most one query per path, so each pixel
    // has at most one writer here. This assignment relies on that invariant;
    // pixels without queries retain the zero written by ray generation.
    if (pixelIdx < pc.imageWidth * pc.imageHeight)
        g_nrcContrib[pixelIdx] = float4(contribution, 1.0);
}
