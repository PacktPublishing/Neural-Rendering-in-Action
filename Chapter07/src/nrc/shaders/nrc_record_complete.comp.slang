// Complete local scattered-radiance targets. A traced suffix contributes
// suffixDirect; a suffix ending at an explicit cache endpoint additionally
// uses the primary (unfiltered) network. EMA is for rendering only.

import nrc;

[[vk::binding(0, 0)]] RWStructuredBuffer<nrc::TrainingRecord> g_trainingRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           g_trainingCount;
[[vk::binding(2, 0)]] ByteAddressBuffer                  g_primaryWeights;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4>         g_trainingTargets;

struct Pc {
    uint matrixOffset[nrc::kNetworkLayerCount];
    uint biasOffset  [nrc::kNetworkLayerCount];
    uint maxRecords;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
[[vk::push_constant]] Pc pc;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx  = tid.x;
    uint nRec = g_trainingCount[0];
    if (idx >= nRec || idx >= pc.maxRecords) return;

    nrc::TrainingRecord r = g_trainingRecords[idx];
    if (!r.isValid()) {
        g_trainingTargets[idx] = float4(0, 0, 0, 0);
        return;
    }

    // Unbiased paths trace to natural termination and never bootstrap.
    bool isUnbiasedRR = r.isUnbiasedRr();
    bool suffixTerminated = r.isSuffixTerminated();

    float3 target;
    if (isUnbiasedRR || suffixTerminated) {
        // Unbiased: full path-traced radiance past v_i already lives in
        // suffixDirect. No MLP tail query — this is the whole point.
        target = r.payload.suffixDirect;
    } else {
        // The emitter excludes the endpoint's own scattered contribution
        // from suffixDirect, so this tail covers exactly the untraced part.
        nrc::NrcVertexFeatures endFeatures = r.getEndFeatures();
        nrc::NrcNetwork network;
        float3 mlpTailOut = network.inference(
            g_primaryWeights, pc.matrixOffset, pc.biasOffset, endFeatures);

        // reflectance factorization at the suffix-end vertex.
        float3 tailReflectance = nrc::reflectanceFactor(endFeatures);
        float3 tail = mlpTailOut * tailReflectance;

        target = r.payload.suffixDirect
            + r.payload.suffixThroughput * tail;
    }

    // The MLP learns Ls NORMALIZED by v_i's reflectance, so when we
    // reconstruct "the target the MLP should predict at v_i features," we
    // divide by (alpha_{v_i} + beta_{v_i}). The training shader then
    // compares MLP output directly to this normalized target.
    float3 viReflectance = nrc::reflectanceFactor(r.features);
    target = target / viReflectance;

    // Do not clamp in reflectance-normalized space: it imposes a lower
    // physical-radiance ceiling on dark materials. The renderer applies its
    // configurable firefly clamp to the final physical pixel estimate.
    target = max(target, float3(0));
    if(any(isnan(target)) || any(isinf(target))) {
        g_trainingTargets[idx] = float4(0);
        return;
    }
    g_trainingTargets[idx] = float4(target, 1.0);
}
