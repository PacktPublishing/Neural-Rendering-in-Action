// Compile-only sanity check for the NRC Slang modules. This does not run; it
// ensures the semantic types, estimator helpers, and RTXNS facades instantiate
// together under the same Slang compiler used by the production shaders.
//
// Build target: nrc_shader_compile_test in src/nrc/CMakeLists.txt.

import nrc;

[[vk::binding(0, 0)]] ByteAddressBuffer   g_weights;
[[vk::binding(1, 0)]] RWByteAddressBuffer g_derivatives;
[[vk::binding(2, 0)]] RWByteAddressBuffer g_image;

// Push-constants for the per-layer offsets.
struct Pc {
    uint matrixOffset[nrc::kNetworkLayerCount];
    uint biasOffset  [nrc::kNetworkLayerCount];
};
[[vk::push_constant]] Pc pc;

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    nrc::NrcVertexFeatures features = nrc::makeVertexFeatures(
        float3(tid.x, 1.0, 2.0),
        float3(0.0, 0.0, 1.0),
        float3(0.0, 1.0, 0.0),
        0.5,
        float3(0.5),
        float3(0.04));

    nrc::QueryRecord query = nrc::makeQueryRecord(features, tid.x, float3(1.0));
    nrc::PendingTrainingRecord pending = nrc::makePendingTrainingRecord(
        features, 0u);
    nrc::TrainingRecord record = nrc::completeTrainingRecord(
        pending, features, false, false);
    float spread = nrc::spreadContribution(1.0, 0.5, 0.75);

    // Every encoder implementation is used through the same generic API.
    nrc::FrequencyEncoder<12> frequencyEncoder;
    nrc::ScalarEncoding<12> frequency = nrc::encodeWith(
        frequencyEncoder, features.position.x);
    nrc::OneBlobEncoder<4> oneBlobEncoder;
    nrc::ScalarEncoding<4> oneBlob = nrc::encodeWith(
        oneBlobEncoder, features.roughness);

    nrc::NrcNetwork network;

    // Inference path
    float3 radiance = network.inference(
        g_weights, pc.matrixOffset, pc.biasOffset, query.getFeatures());

    // Training forward path
    float3 forwardOutput = network.forward(
        g_weights,
        g_derivatives,
        pc.matrixOffset,
        pc.biasOffset,
        record.getFeatures());

    // Synthesize an upstream gradient and backprop it.
    network.backward(
        g_weights,
        g_derivatives,
        pc.matrixOffset,
        pc.biasOffset,
        pc.matrixOffset,
        pc.biasOffset,
        record.getFeatures(),
        float3(0.01));

    // Instantiate a non-default, five-output configuration so this test also
    // exercises the dimension-generic API beyond the Chapter 7 alias.
    nrc::ScalarEncoding<4> alternateInput;
    nrc::NetworkValues<5> alternateGradient;
    [ForceUnroll]
    for(int lane = 0; lane < 4; ++lane)
    {
        alternateInput.setLane(lane, float(lane) * 0.1);
    }
    [ForceUnroll]
    for(int lane = 0; lane < 5; ++lane)
    {
        alternateGradient.setLane(lane, 0.01);
    }
    uint alternateMatrixOffset[2];
    uint alternateBiasOffset[2];
    alternateMatrixOffset[0] = 0u;
    alternateMatrixOffset[1] = 0u;
    alternateBiasOffset[0] = 0u;
    alternateBiasOffset[1] = 0u;
    nrc::Network<4, 8, 5, 1> alternateNetwork;
    nrc::NetworkValues<5> alternateOutput = alternateNetwork.inference(
        g_weights,
        alternateMatrixOffset,
        alternateBiasOffset,
        alternateInput);
    alternateNetwork.forward(
        g_weights,
        g_derivatives,
        alternateMatrixOffset,
        alternateBiasOffset,
        alternateInput);
    alternateNetwork.backward(
        g_weights,
        g_derivatives,
        alternateMatrixOffset,
        alternateBiasOffset,
        alternateMatrixOffset,
        alternateBiasOffset,
        alternateInput,
        alternateGradient);

    // Sink module results so they are not dead-code-eliminated.
    g_image.Store<float3>(
        tid.x * 12,
        radiance + forwardOutput * 0.001 + spread * 0.0001
            + frequency.getLane(0) * 0.00001
            + oneBlob.getLane(0) * 0.000001
            + alternateOutput.getLane(4) * 0.0000001);
}
