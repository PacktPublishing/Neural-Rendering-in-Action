// Optimizer compute pass (Task 11d).
//
// Adam update + EMA update. Runs once per backward pass (i.e., 4 times per
// frame). Operates byte-level on the weight buffer; Adam is element-wise
// so it doesn't care about the weight buffer's (TrainingOptimal tiled)
// layout — we just walk bytes.
//
// Adam (Kingma & Ba 2014):
//   m' = beta1 * m + (1 - beta1) * g
//   v' = beta2 * v + (1 - beta2) * g^2
//   m_hat = m' / (1 - beta1^t)
//   v_hat = v' / (1 - beta2^t)
//   w' = w - lr * m_hat / (sqrt(v_hat) + eps)
//
// EMA:
//   w_ema' = alpha * w_ema + (1 - alpha) * w'
//
// We treat all weights as fp16 for storage (matches TrainingOptimal) and
// fp32 for the intermediate computation (moments). The moments buffer is
// 2x the weight buffer size: [m0, v0, m1, v1, ...].
//
// Descriptor set 0:
//   binding 0 : RWByteAddressBuffer g_primaryWeights (fp16)
//   binding 1 : RWByteAddressBuffer g_emaWeights     (fp16)
//   binding 2 : RWByteAddressBuffer g_gradients      (fp16)
//   binding 3 : RWByteAddressBuffer g_adamMoments    (fp32 m,v pairs per weight)

[[vk::binding(0, 0)]] RWByteAddressBuffer g_primaryWeights;
[[vk::binding(1, 0)]] RWByteAddressBuffer g_emaWeights;
[[vk::binding(2, 0)]] RWByteAddressBuffer g_gradients;
[[vk::binding(3, 0)]] RWByteAddressBuffer g_adamMoments;

struct Pc {
    uint  elementCount;  // number of fp16 weights (= totalBytes / 2)
    uint  adamStep;      // t (for bias correction), starts at 1
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

    // Each weight is an fp16 at byte offset i*2.
    uint weightByteOffset = i * 2;
    // Each moment pair is [m, v] float32 pair at byte offset i*8.
    uint momentByteOffset = i * 8;

    // Load
    float w    = float(asfloat16(uint16_t(g_primaryWeights.Load<uint16_t>(weightByteOffset))));
    float we   = float(asfloat16(uint16_t(g_emaWeights.Load<uint16_t>(weightByteOffset))));
    float gRaw = float(asfloat16(uint16_t(g_gradients.Load<uint16_t>(weightByteOffset))));
    float g    = gRaw * pc.gradientScale;
    float m    = asfloat(g_adamMoments.Load<uint>(momentByteOffset + 0));
    float v    = asfloat(g_adamMoments.Load<uint>(momentByteOffset + 4));

    if (isnan(g) || isinf(g)) g = 0.0;
    if (isnan(w) || isinf(w)) w = 0.0;
    if (isnan(we) || isinf(we)) we = w;
    if (isnan(m) || isinf(m)) m = 0.0;
    if (isnan(v) || isinf(v) || v < 0.0) v = 0.0;

    // Adam step
    m = pc.beta1 * m + (1.0 - pc.beta1) * g;
    v = pc.beta2 * v + (1.0 - pc.beta2) * g * g;
    float biasC1 = 1.0 - pow(pc.beta1, float(pc.adamStep));
    float biasC2 = 1.0 - pow(pc.beta2, float(pc.adamStep));
    float mHat   = m / max(biasC1, 1e-8);
    float vHat   = v / max(biasC2, 1e-8);
    float wNew   = w - pc.learningRate * mHat / (sqrt(vHat) + pc.eps);
    if (isnan(wNew) || isinf(wNew)) wNew = w;

    // EMA update
    float weNew = pc.emaAlpha * we + (1.0 - pc.emaAlpha) * wNew;
    if (isnan(weNew) || isinf(weNew)) weNew = wNew;

    // Store weights (fp16), moments (fp32), reset gradients.
    g_primaryWeights.Store<uint16_t>(weightByteOffset, uint16_t(asuint16(float16_t(wNew))));
    g_emaWeights.Store<uint16_t>(weightByteOffset, uint16_t(asuint16(float16_t(weNew))));
    g_gradients.Store<uint16_t>(weightByteOffset, uint16_t(0));
    g_adamMoments.Store<uint>(momentByteOffset + 0, asuint(m));
    g_adamMoments.Store<uint>(momentByteOffset + 4, asuint(v));
}
