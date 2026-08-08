// Inference-only subset of the NRC MLP facade. Used by the cache-query
// compute shader.

#ifndef NRC_MLP_RTXNS_INFERENCE_SLANG
#define NRC_MLP_RTXNS_INFERENCE_SLANG

#include "nrc_common.h.slang"
#include "nrc_features.slang"

import MLP;
import Activation;
import CooperativeVectorFunctions;

using namespace rtxns::mlp;

// B4: input is now 64-dim encoded features (post nrcEncode()).
#define NRC_INPUT_NEURONS    64
#define NRC_HIDDEN_NEURONS   64
#define NRC_OUTPUT_NEURONS    3
#define NRC_HIDDEN_LAYERS     5

// These are required to be constants the template accepts; #define them.
#define NRC_LAYOUT          CoopVecMatrixLayout::TrainingOptimal
#define NRC_COMPONENT_TYPE  CoopVecComponentType::Float16

// Legacy names still used by the host-side MlpLayout struct.
static const int kNrcInputNeurons  = NRC_INPUT_NEURONS;
static const int kNrcHiddenNeurons = NRC_HIDDEN_NEURONS;
static const int kNrcOutputNeurons = NRC_OUTPUT_NEURONS;
static const int kNrcHiddenLayers  = NRC_HIDDEN_LAYERS;

CoopVec<float16_t, kNrcInputNeurons> nrcFeaturesToCoopVec(EncodedMlpFeatures f) {
    var v = CoopVec<float16_t, kNrcInputNeurons>(float16_t(0));
    [ForceUnroll]
    for (int i = 0; i < kNrcInputNeurons; ++i)
        v[i] = float16_t(f.f[i]);
    return v;
}

float3 nrcCoopVecToFloat3(CoopVec<float16_t, kNrcOutputNeurons> v) {
    return float3(float(v[0]), float(v[1]), float(v[2]));
}

// Activation factories: isolating each template instantiation in its own
// small function avoids the "two consecutive rtxns::mlp::X<...>()" parse
// failure seen on Slang 2026.5 and 2025.23.1.
ReLUAct<float16_t, NRC_HIDDEN_NEURONS> makeReluActNrc() {
    ReLUAct<float16_t, NRC_HIDDEN_NEURONS> a;
    return a;
}
LinearAct<float16_t, NRC_OUTPUT_NEURONS> makeLinearActNrc() {
    LinearAct<float16_t, NRC_OUTPUT_NEURONS> a = LinearAct<float16_t, NRC_OUTPUT_NEURONS>(float16_t(1.0));
    return a;
}

float3 nrcInference(
    ByteAddressBuffer weights,
    uint matrixOffset[NRC_HIDDEN_LAYERS + 1],
    uint biasOffset[NRC_HIDDEN_LAYERS + 1],
    MlpFeatures features)
{
    EncodedMlpFeatures encoded = nrcEncode(features);
    var input = nrcFeaturesToCoopVec(encoded);
    var mlp = InferenceMLP<float16_t, NRC_HIDDEN_LAYERS,
        NRC_INPUT_NEURONS, NRC_HIDDEN_NEURONS, NRC_OUTPUT_NEURONS,
        NRC_LAYOUT, NRC_COMPONENT_TYPE>(weights, matrixOffset, biasOffset);
    var relu = makeReluActNrc();
    var linear = makeLinearActNrc();
    var output = mlp.forward(input, relu, linear);
    return max(nrcCoopVecToFloat3(output), float3(0));
}

#endif // NRC_MLP_RTXNS_INFERENCE_SLANG
