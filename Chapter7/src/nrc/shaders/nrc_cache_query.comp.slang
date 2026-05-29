// Cache-query compute pass.
//
// Reads each QueryRecord emitted by the ray-gen shader and writes a "cache
// contribution" into g_nrcContrib at record.pixelIdx. The contribution is
// consumed by the compose pass in the same frame after ray tracing.
//
// Each query is evaluated with the current NRC MLP through the RTXNS-backed
// inference facade. The network predicts normalized reflected radiance; this
// pass applies diffuse+specular reflectance, multiplies by the stored path
// throughput, clamps fireflies, and stores the result for composition.
//
// Descriptor set 0:
//   binding 0 : RWStructuredBuffer<QueryRecord> g_queries
//   binding 1 : RWStructuredBuffer<uint>        g_queryCount
//   binding 2 : ByteAddressBuffer               g_inferenceWeights
//   binding 3 : RWStructuredBuffer<float4>      g_nrcContrib

#include "nrc_common.h.slang"
#include "nrc_features.slang"
#include "nrc_mlp_rtxns_inference.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<QueryRecord> g_queries;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>        g_queryCount;
[[vk::binding(2, 0)]] ByteAddressBuffer               g_inferenceWeights;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4>      g_nrcContrib;

struct Pc {
    // Per-layer byte offsets into g_inferenceWeights. NRC_HIDDEN_LAYERS = 5 so
    // these are 6-element arrays. Total 12 uints = 48 B, plus the 4 trailing
    // uints = 64 B. Well within the 128 B Vulkan push-constant budget.
    uint matrixOffset[NRC_HIDDEN_LAYERS + 1];
    uint biasOffset[NRC_HIDDEN_LAYERS + 1];

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
    if (idx >= nQueries || idx >= pc.maxQueries) return;

    QueryRecord q = g_queries[idx];

    // Real MLP inference via the RTXNS facade.
    MlpFeatures f = packFromQuery(q);
    float3 mlpOut = nrcInference(
        g_inferenceWeights, pc.matrixOffset, pc.biasOffset, f);

    // reflectance factorization — paper §4.1 multiplies MLP output by
    // (α + β) before using as cache value. The MLP learns normalized
    // reflected radiance; scaling by (diffuseAlbedo + specularAlbedo) brings
    // it back to physical HDR magnitude.
    float3 reflectance = q.diffuseAlbedo + q.specularAlbedo;
    float3 cacheRadiance = mlpOut * reflectance;

    float3 contribution  = cacheRadiance * q.pathThroughput;

    // Clamp firefly.
    contribution = clamp(contribution, 0.0, 20.0);

    uint pixelIdx = q.pixelIdx;
    if (pixelIdx < pc.imageWidth * pc.imageHeight)
        g_nrcContrib[pixelIdx] = float4(contribution, 1.0);
}
