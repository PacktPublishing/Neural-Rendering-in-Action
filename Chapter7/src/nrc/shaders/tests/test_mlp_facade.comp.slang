// Compile-only sanity check for nrc_mlp_rtxns.slang. Does not run — just
// ensures the facade types instantiate cleanly against RTXNS v1.3.1.
//
// Build target: nrc_shader_compile_test in src/nrc/CMakeLists.txt.

#include "nrc_mlp_rtxns.slang"

[[vk::binding(0, 0)]] ByteAddressBuffer   g_weights;
[[vk::binding(1, 0)]] RWByteAddressBuffer g_derivatives;
[[vk::binding(2, 0)]] RWByteAddressBuffer g_image;

// Push-constants for the per-layer offsets.
struct Pc {
    uint matrixOffset[kNrcHiddenLayers + 1];
    uint biasOffset  [kNrcHiddenLayers + 1];
};
[[vk::push_constant]] Pc pc;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Build a dummy feature vector.
    MlpFeatures f;
    [ForceUnroll]
    for (int i = 0; i < 14; ++i)
        f.f[i] = float(i);

    // Inference path
    float3 radiance = nrcInference(g_weights, pc.matrixOffset, pc.biasOffset, f);

    // Training forward path
    var forwardOut = nrcTrainForward(
        g_weights, g_derivatives, pc.matrixOffset, pc.biasOffset, f);

    // Synthesize an upstream gradient and backprop it.
    var upstreamCv = CoopVec<float16_t, 3>(float16_t(0.01));
    nrcTrainBackward(
        g_weights, g_derivatives, pc.matrixOffset, pc.biasOffset, f, upstreamCv);

    // Sink the radiance so it isn't dead-code-eliminated.
    g_image.Store<float3>(tid.x * 12, radiance);
}
