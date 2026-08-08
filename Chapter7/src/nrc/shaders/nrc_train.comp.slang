// Each Adam step trains on a shuffled batch (16384 records by default). Forward and
// reverse matrix-vector products use FP16, while all parameter derivatives
// accumulate into a separate FP32 TrainingOptimal buffer.

import nrc;

[[vk::binding(0, 0)]] RWStructuredBuffer<nrc::TrainingRecord> g_trainingRecords;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint>           g_trainingCount;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4>         g_trainingTargets;
[[vk::binding(3, 0)]] ByteAddressBuffer                  g_primaryWeights;
[[vk::binding(4, 0)]] RWByteAddressBuffer                g_gradients;

struct Pc {
    uint matrixOffset[nrc::kNetworkLayerCount];
    uint biasOffset  [nrc::kNetworkLayerCount];
    uint maxRecords;     // trainingRecordsPerFrame capacity; 65536 by default
    float lossScale;     // mixed-precision scale, undone by optimizer
    uint stepIdx;        // 0..adamStepsPerFrame-1
    uint recordsPerBatch; // must match the host dispatch and normalization
    uint gradientMatrixOffset[nrc::kNetworkLayerCount];
    uint gradientBiasOffset[nrc::kNetworkLayerCount];
    uint shuffleSeed;    // shared by every minibatch in this frame
};
[[vk::push_constant]] Pc pc;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // A partial final workgroup must not change the effective batch size.
    if (tid.x >= pc.recordsPerBatch) return;

    uint nRec = min(g_trainingCount[0], pc.maxRecords);
    if (nRec == 0u) return;

    // Count dispatches with a nonempty record pool on the GPU, even if later
    // validation rejects every sample. The CPU cannot know whether
    // the current frame emitted any records without a blocking readback.
    // Only this invocation writes the persistent second counter; Adam reads
    // it after the dispatch barrier, so empty frames advance no optimizer state.
    if (tid.x == 0u) g_trainingCount[1] = g_trainingCount[1] + 1u;

    // Exhaust one shuffled population before repeating records to fill the
    // fixed-size batches. The frame seed changes which records repeat in a
    // partial final cycle, without gaps or collisions inside a full cycle.
    uint shuffled = nrc::shuffledTrainingIndex(
        tid.x + pc.stepIdx * pc.recordsPerBatch, nRec, pc.shuffleSeed);
    if (shuffled >= pc.maxRecords) return;  // buffer-capacity guard

    nrc::TrainingRecord r = g_trainingRecords[shuffled];
    if (!r.isValid()) return;
    uint idx = shuffled;  // for trainingTargets indexing below

    float4 target4 = g_trainingTargets[idx];
    if (target4.w == 0.0) return;  // record had no valid target

    nrc::TrainingSample sample = nrc::makeTrainingSample(
        r.getFeatures(), target4.xyz);

    // Forward pass with primary weights.
    nrc::NrcNetwork network;
    float3 pred = network.forward(
        g_primaryWeights, g_gradients,
        pc.matrixOffset, pc.biasOffset, sample.features);

    // Relative-L2 (paper Eq. 5), sharing a detached luminance normalizer
    // across RGB. Clamp only the normalizer's radiance to nonnegative values;
    // retain negative linear outputs in the residual so they can recover.
    const float kEps = 1e-2;
    if(any(isnan(pred)) || any(isinf(pred))) return;
    float Ypred = dot(float3(0.299, 0.587, 0.114), max(pred, float3(0)));
    float denom = Ypred * Ypred + kEps;
    float3 dLdPred = 2.0 * (pred - sample.target) / denom;
    if(any(isnan(dLdPred)) || any(isinf(dLdPred))) return;

    // Limit an extreme sample's norm before loss scaling. Normalize the batch
    // before narrowing to FP16; doing it after the accumulation was too late
    // to prevent overflow. FP32 accumulation preserves the small additions.
    float largest = max(max(abs(dLdPred.x), abs(dLdPred.y)), abs(dLdPred.z));
    dLdPred *= min(1.0, 1000.0 / max(largest, 1e-20));
    // Normalize by configured sample slots, including any skipped samples;
    // no separate count of accepted samples is used.
    dLdPred *= pc.lossScale / float(pc.recordsPerBatch);

    network.backward(
        g_primaryWeights, g_gradients,
        pc.matrixOffset, pc.biasOffset,
        pc.gradientMatrixOffset, pc.gradientBiasOffset,
        sample.features, dLdPred);
}
