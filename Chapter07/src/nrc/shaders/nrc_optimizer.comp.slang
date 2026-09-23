// Adam and temporal filtering use FP32 persistent state. Both master
// parameter buffers have the same driver-queried FP32 TrainingOptimal layout
// as the accumulated gradients. Host conversions publish FP16 matrices and
// biases for inference after this dispatch; rounding never feeds back into
// the optimizer or the EMA recurrence.

[[vk::binding(0, 0)]] RWByteAddressBuffer g_primaryWeights;
[[vk::binding(1, 0)]] RWByteAddressBuffer g_emaWeights;
[[vk::binding(2, 0)]] RWByteAddressBuffer g_gradients;
[[vk::binding(3, 0)]] RWByteAddressBuffer g_adamMoments;
[[vk::binding(4, 0)]] StructuredBuffer<uint> g_trainingCount;

struct Pc {
    uint  elementCount;  // number of FP32 layout elements (= totalBytes / 4)
    uint  _pad0;         // Adam step comes from the persistent GPU counter
    float learningRate;
    float beta1;
    float beta2;
    float eps;
    float emaAlpha;
    float gradientScale;  // inverse of lossScale used in training shader
};
[[vk::push_constant]] Pc pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= pc.elementCount) return;
    // A zero gradient is still an Adam update: it decays moments, moves
    // weights with existing momentum, and changes EMA. Empty record batches
    // must bypass the whole update, not merely supply zero derivatives.
    if (g_trainingCount[0] == 0u) return;
    uint adamStep = g_trainingCount[1];
    if (adamStep == 0u) return;

    uint weightByteOffset = i * 4;
    // Each moment pair is [m, v] float32 pair at byte offset i*8.
    uint momentByteOffset = i * 8;

    // Load
    float w    = g_primaryWeights.Load<float>(weightByteOffset);
    float we   = g_emaWeights.Load<float>(weightByteOffset);
    float gRaw = g_gradients.Load<float>(weightByteOffset);
    float g    = gRaw * pc.gradientScale;
    float m    = asfloat(g_adamMoments.Load<uint>(momentByteOffset + 0));
    float v    = asfloat(g_adamMoments.Load<uint>(momentByteOffset + 4));

    // Preserve finite state on overflow instead of advancing stale momentum.
    if (isnan(g) || isinf(g)) return;
    if (isnan(w) || isinf(w)) w = 0.0;
    if (isnan(we) || isinf(we)) we = w;
    if (isnan(m) || isinf(m)) m = 0.0;
    if (isnan(v) || isinf(v) || v < 0.0) v = 0.0;

    // Adam step
    m = pc.beta1 * m + (1.0 - pc.beta1) * g;
    v = pc.beta2 * v + (1.0 - pc.beta2) * g * g;
    float biasC1 = 1.0 - pow(pc.beta1, float(adamStep));
    float biasC2 = 1.0 - pow(pc.beta2, float(adamStep));
    float mHat   = m / max(biasC1, 1e-8);
    float vHat   = v / max(biasC2, 1e-8);
    float wNew   = w - pc.learningRate * mHat / (sqrt(vHat) + pc.eps);
    if (isnan(wNew) || isinf(wNew)) wNew = w;

    // EMA update
    float weNew = pc.emaAlpha * we + (1.0 - pc.emaAlpha) * wNew;
    if (isnan(weNew) || isinf(weNew)) weNew = wNew;

    // Persist FP32 state. Host clears gradients with an explicit dependency.
    g_primaryWeights.Store<float>(weightByteOffset, wNew);
    g_emaWeights.Store<float>(weightByteOffset, weNew);
    g_adamMoments.Store<uint>(momentByteOffset + 0, asuint(m));
    g_adamMoments.Store<uint>(momentByteOffset + 4, asuint(v));
}
