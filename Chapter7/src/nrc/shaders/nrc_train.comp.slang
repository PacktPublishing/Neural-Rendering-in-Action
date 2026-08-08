// Training compute pass (Task 11c).
//
// For each TrainingRecord with a valid target (from Task 11b):
//  1. Run MLP forward with the *primary* weights at v0 features.
//  2. Compute the relative-L2 loss gradient against the target radiance:
//        L = ((pred - target)^2) / (stop_grad(pred)^2 + eps)
//     The output activation is linear.
//     So we provide the gradient w.r.t. the MLP's *output* (pred), and
//     autodiff cascades the chain rule through the linear output, ReLUs,
//     and LinearOps.
//  3. Call nrcTrainBackward(upstreamGrad = dL/d(pred)) to atomically
//     accumulate derivatives into g_gradients.
//
// Descriptor set 0:
//   binding 0 : RWStructuredBuffer<TrainingRecord> g_trainingRecords
//   binding 1 : RWStructuredBuffer<uint>           g_trainingCount
//   binding 2 : RWStructuredBuffer<float4>         g_trainingTargets
//   binding 3 : ByteAddressBuffer                  g_primaryWeights
//   binding 4 : RWByteAddressBuffer                g_gradients

#include "nrc_common.h.slang"
#include "nrc_features.slang"
#include "nrc_mlp_rtxns.slang"  // training facade (training + inference)

[[vk::binding(0, 0)]] RWStructuredBuffer<TrainingRecord> g_trainingRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           g_trainingCount;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4>         g_trainingTargets;
[[vk::binding(3, 0)]] ByteAddressBuffer                  g_primaryWeights;
[[vk::binding(4, 0)]] RWByteAddressBuffer                g_gradients;

// §3 (Task 3): 4 Adam steps/frame, each seeing a disjoint 16384-record slice
// of the 65536-record pool shuffled by the Knuth LCG constant 2654435761u.
// pc.stepIdx is 0..3; pc.recordCount is the actual number of records emitted
// by ray-gen this frame (host reads it back from g_trainingCount[0] one frame
// late — one frame of staleness is acceptable). When recordCount < 65536,
// the modulo wraps and the 4 steps will partially overlap — acceptable per
// spec §3.
struct Pc {
    uint matrixOffset[kNrcHiddenLayers + 1];
    uint biasOffset  [kNrcHiddenLayers + 1];
    uint maxRecords;     // buffer capacity (== trainingRecordsPerFrame == 65536)
    float lossScale;     // scales gradient (acts like inverse learning rate)
    uint stepIdx;        // 0..adamStepsPerFrame-1
    uint recordCount;    // live record count this frame (for LCG modulo)
};
[[vk::push_constant]] Pc pc;

static const uint kNrcRecordsPerBatch = 16384u;
static const uint kNrcLcgMultiplier   = 2654435761u;  // Knuth

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Each Adam step dispatches kNrcRecordsPerBatch threads (16384) — thread
    // tid.x in [0, 16384). Early-out on the padding threads in the final
    // workgroup (16384 % 128 == 0, so none today, but guard defensively).
    if (tid.x >= kNrcRecordsPerBatch) return;

    uint nRec = min(g_trainingCount[0], pc.maxRecords);
    if (nRec == 0u) return;

    // Disjoint LCG slice: different stepIdx → different starting offset.
    uint shuffled = ((tid.x + pc.stepIdx * kNrcRecordsPerBatch) * kNrcLcgMultiplier) % nRec;
    if (shuffled >= pc.maxRecords) return;  // buffer-capacity guard

    TrainingRecord r = g_trainingRecords[shuffled];
    if ((r.flags & 0x1u) == 0u) return;
    uint idx = shuffled;  // for trainingTargets indexing below

    float4 target4 = g_trainingTargets[idx];
    if (target4.w == 0.0) return;  // record had no valid target

    float3 target = target4.xyz;
    MlpFeatures f = packFromTrainingStart(r);

    // Forward pass with primary weights.
    var predCv = nrcTrainForward(
        g_primaryWeights, g_gradients,
        pc.matrixOffset, pc.biasOffset, f);

    float3 pred = float3(float(predCv[0]), float(predCv[1]), float(predCv[2]));

    // relative-L2 with shared luminance normalizer (paper eq. 5).
    // Paper: L = ||pred - target||^2 / (sg(Y(pred))^2 + eps)
    //   where Y(rgb) = 0.2126*r + 0.7152*g + 0.0722*b  (Rec.709 luminance)
    //   and sg(.) is stop-gradient (treat as constant during backprop).
    // We compute Y(max(pred, 0)) here. That matters for untrained
    // linear-output networks: negative channels should not create a large
    // denominator that weakens the first useful gradient.
    // dL/d(pred_c) = 2 * (pred_c - target_c) / (sg(Y(max(pred,0)))^2 + eps)
    // All channels share the scalar denominator — paper says this is
    // important for photometrically-correct gradient weighting.
    const float kEps = 1e-2;
    float3 predNonNegative = max(pred, float3(0));
    float Ypred = dot(float3(0.299, 0.587, 0.114), predNonNegative);
    float denom = Ypred * Ypred + kEps;
    float3 dLdPred = 2.0 * (pred - target) / denom;

    // Apply loss scale (a common NRC technique — scaling the gradient stably
    // across fp16 dynamic range).
    dLdPred *= pc.lossScale;

    // Clamp gradient to prevent divergence.
    dLdPred = clamp(dLdPred, -1e3, 1e3);

    // Upstream gradient in CoopVec form.
    var upstream = CoopVec<float16_t, kNrcOutputNeurons>(float16_t(0));
    upstream[0] = float16_t(dLdPred.x);
    upstream[1] = float16_t(dLdPred.y);
    upstream[2] = float16_t(dLdPred.z);

    nrcTrainBackward(
        g_primaryWeights, g_gradients,
        pc.matrixOffset, pc.biasOffset, f, upstream);
}
