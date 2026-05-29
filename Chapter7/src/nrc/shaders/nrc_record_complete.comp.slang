// Record-completion compute pass (Task 11b).
//
// For each TrainingRecord emitted by the ray-gen shader this frame, compute
// the "target radiance" — the self-supervised target the MLP will be trained
// to match at the record's v0 features.
//
// Target formula:
//   L_target = suffixDirect + suffixThroughput * MLP_EMA(suffixEnd features)
//
// Where:
//   - suffixDirect is the direct-lighting sum along the 2-bounce suffix,
//     already throughput-scaled (accumulated in ray-gen).
//   - MLP tail is queried with the current inference weights. With Use EMA
//     enabled this is the EMA buffer; otherwise it is the primary buffer
//   - suffixThroughput is the product of BSDF*cos/pdf along the suffix.
//
// Output is written to g_trainingTargets[idx] so the training shader
// (Task 11c) can read both the record (via g_trainingRecords[idx]) and its
// target (via g_trainingTargets[idx]) in one pass.
//
// Descriptor set 0:
//   binding 0 : RWStructuredBuffer<TrainingRecord> g_trainingRecords
//   binding 1 : RWStructuredBuffer<uint>           g_trainingCount
//   binding 2 : ByteAddressBuffer                  g_inferenceWeights
//   binding 3 : RWStructuredBuffer<float4>         g_trainingTargets  (float3 + pad)

#include "nrc_common.h.slang"
#include "nrc_features.slang"
#include "nrc_mlp_rtxns_inference.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<TrainingRecord> g_trainingRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           g_trainingCount;
[[vk::binding(2, 0)]] ByteAddressBuffer                  g_inferenceWeights;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4>         g_trainingTargets;

struct Pc {
    uint matrixOffset[NRC_HIDDEN_LAYERS + 1];
    uint biasOffset  [NRC_HIDDEN_LAYERS + 1];
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

    TrainingRecord r = g_trainingRecords[idx];
    if ((r.flags & 0x1u) == 0u) {
        g_trainingTargets[idx] = float4(0, 0, 0, 0);
        return;
    }

    // §4 γ=1/16 unbiased-RR flag. Set once per training path in ray-gen
    // (nrcIsUnbiasedRRPath) and stamped into every record's flags bit 1 by
    // nrcEmitTrainingStack. When set, the target drops the MLP-tail term
    // and uses suffixDirect alone — this is unbiased because our training
    // paths already run to natural RR end, so suffixDirect IS the full
    // path-traced radiance from v_i onward. Injects unbiased signal into
    // the otherwise self-referential training (paper §3.2).
    bool isUnbiasedRR = (r.flags & 0x2u) != 0u;
    // We only asks the MLP for a tail when the suffix did not already
    // terminate on a light/environment. If the sampled suffix ended by
    // environment hit, unlit surface, absorption, or RR kill, suffixDirect is
    // already the complete sampled continuation and adding an MLP tail would
    // double-count/brighten the target.
    bool suffixTerminated = (r.flags & 0x100u) != 0u;

    float3 target;
    if (isUnbiasedRR || suffixTerminated) {
        // Unbiased: full path-traced radiance past v_i already lives in
        // suffixDirect. No MLP tail query — this is the whole point.
        target = r.suffixDirect;
    } else {
        // add MLP-tail self-reference term at v_last.
        // End-of-suffix features at v_last. For single-record paths (stack
        // had exactly one push before termination) v_i == v_last; the MLP
        // tail then evaluates at v_i itself and contributes only a weak
        // self-reference term — acceptable.
        MlpFeatures fEnd = packFromTrainingEnd(r);
        float3 mlpTailOut = nrcInference(g_inferenceWeights, pc.matrixOffset, pc.biasOffset, fEnd);

        // reflectance factorization at the suffix-end vertex.
        float3 tailReflectance = r.suffixEndDiffuse + r.suffixEndSpecular;
        float3 tail = mlpTailOut * tailReflectance;

        // §2: suffixDirect = Σ_{j > i} T_{i->j} · direct(v_j), and
        // suffixThroughput = T_{i -> last}. Target is the sum of those plus the
        // MLP-evaluated tail beyond v_last, all relative to this record's own v_i.
        target = r.suffixDirect + r.suffixThroughput * tail;
    }

    // The MLP learns Ls NORMALIZED by v_i's reflectance, so when we
    // reconstruct "the target the MLP should predict at v_i features," we
    // divide by (alpha_{v_i} + beta_{v_i}). The training shader then
    // compares MLP output directly to this normalized target.
    float3 viReflectance = r.diffuseAlbedo + r.specularAlbedo;
    target = target / max(viReflectance, float3(1e-3));

    // Firefly clamp in NORMALIZED space (after dividing by reflectance).
    target = clamp(target, 0.0, 20.0);

    g_trainingTargets[idx] = float4(target, 1.0);
}
