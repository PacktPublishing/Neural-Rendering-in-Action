// NRC MLP facade, Part 2 backend (uses RTXNS Slang modules).
//
// This file is the single point of contact between our NRC compute shaders
// and RTXNS's TrainingMLP. Part 3 replaces this file's implementation with
// our own MLP, leaving the public API (NrcMlp* types, nrcInference, nrcTrainForward,
// nrcTrainBackward) identical so the compute shaders don't change.
//
// Network architecture (NRC paper defaults):
//   - 14-scalar input (pre-encoding — encoding happens in nrc_features.slang)
//   - 5 hidden layers × 64 neurons
//   - 3-channel output (RGB radiance)
//   - ReLU in hidden layers, linear output, nonnegative clamp for inference.
//
// Precision: fp16 everywhere. The 16-bit matmul path is the fast one on
// Ada/Blackwell and sufficient for NRC's dynamic range (we train in
// relative-L2 which is scale-invariant).

#ifndef NRC_MLP_RTXNS_SLANG
#define NRC_MLP_RTXNS_SLANG

#include "nrc_common.h.slang"
#include "nrc_features.slang"

// RTXNS modules live on the Slang include path (set by CMake in Task 8 /
// src/nrc/CMakeLists.txt:NRC_SHADER_INCLUDE_DIRS).
import MLP;
import Activation;
import CooperativeVectorFunctions;
import CooperativeVectorAutoDiff;

// ---------------------------------------------------------------------------
// Network dimensions.
// input is 64-dim encoded features (from nrcEncode() in nrc_features.slang).
// ---------------------------------------------------------------------------
static const int kNrcInputNeurons   = 64;
static const int kNrcHiddenNeurons  = 64;
static const int kNrcOutputNeurons  = 3;    // RGB
static const int kNrcHiddenLayers   = 5;

// Matrix layout: TrainingOptimal. Both the RTXNS DisneyMLP sample and the
// ShaderTraining sample use the same layout for inference and training to
// avoid a per-frame layout-conversion pass. We follow that convention —
// Task 10 uploads weights in TrainingOptimal once and never needs to
// repack them.
static const CoopVecMatrixLayout kNrcLayout = CoopVecMatrixLayout::TrainingOptimal;
static const CoopVecComponentType kNrcComponentType = CoopVecComponentType::Float16;

// ---------------------------------------------------------------------------
// Inference type — used by the cache-query pass and by the training-record
// completion pass (which queries EMA weights for the suffix tail).
// ---------------------------------------------------------------------------
typealias NrcInferenceMlp = rtxns::mlp::InferenceMLP<
    float16_t,
    kNrcHiddenLayers,
    kNrcInputNeurons,
    kNrcHiddenNeurons,
    kNrcOutputNeurons,
    kNrcLayout,
    kNrcComponentType
>;

// ---------------------------------------------------------------------------
// Training type — same layout, but carries a derivative buffer for backward.
// ---------------------------------------------------------------------------
typealias NrcTrainingMlp = rtxns::mlp::TrainingMLP<
    float16_t,
    kNrcHiddenLayers,
    kNrcInputNeurons,
    kNrcHiddenNeurons,
    kNrcOutputNeurons,
    kNrcLayout,
    kNrcComponentType
>;

// ---------------------------------------------------------------------------
// Convert the 14-float MlpFeatures into a CoopVec<float16,14> input.
// ---------------------------------------------------------------------------
CoopVec<float16_t, kNrcInputNeurons> nrcFeaturesToCoopVec(EncodedMlpFeatures f) {
    var v = CoopVec<float16_t, kNrcInputNeurons>(float16_t(0));
    [ForceUnroll]
    for (int i = 0; i < kNrcInputNeurons; ++i)
        v[i] = float16_t(f.f[i]);
    return v;
}

// ---------------------------------------------------------------------------
// Convert a CoopVec<float16,3> to a float3 (no activation — the activation is
// chosen inside the forward call below).
// ---------------------------------------------------------------------------
float3 nrcCoopVecToFloat3(CoopVec<float16_t, kNrcOutputNeurons> v) {
    return float3(float(v[0]), float(v[1]), float(v[2]));
}

// ---------------------------------------------------------------------------
// Inference entry point: ReLU hidden activations, linear output
// with nonnegative clamp during inference
// ---------------------------------------------------------------------------
float3 nrcInference(
    ByteAddressBuffer weights,
    uint matrixOffset[kNrcHiddenLayers + 1],
    uint biasOffset[kNrcHiddenLayers + 1],
    MlpFeatures features)
{
    EncodedMlpFeatures encoded = nrcEncode(features);
    // Note: construct the MLP by fully-qualifying the rtxns type (Slang 2026.5
    // stricter than 2025.x: typealias works as a type but not as a constructor).
    var mlp = rtxns::mlp::InferenceMLP<
        float16_t, kNrcHiddenLayers, kNrcInputNeurons, kNrcHiddenNeurons,
        kNrcOutputNeurons, kNrcLayout, kNrcComponentType>(
        weights, matrixOffset, biasOffset);
    var input = nrcFeaturesToCoopVec(encoded);
    rtxns::mlp::ReLUAct<float16_t, kNrcHiddenNeurons> reluAct;
    var linearAct = rtxns::mlp::LinearAct<float16_t, kNrcOutputNeurons>(float16_t(1.0));
    var output = mlp.forward(input, reluAct, linearAct);
    return max(nrcCoopVecToFloat3(output), float3(0));
}

// ---------------------------------------------------------------------------
// Training forward: same activations as inference. The output is returned
// in CoopVec form so the training pass can compute loss against the target
// in the same space the backward pass expects.
// ---------------------------------------------------------------------------
CoopVec<float16_t, kNrcOutputNeurons> nrcTrainForward(
    ByteAddressBuffer weights,
    RWByteAddressBuffer derivatives,
    uint matrixOffset[kNrcHiddenLayers + 1],
    uint biasOffset[kNrcHiddenLayers + 1],
    MlpFeatures features)
{
    EncodedMlpFeatures encoded = nrcEncode(features);
    var mlp = rtxns::mlp::TrainingMLP<
        float16_t, kNrcHiddenLayers, kNrcInputNeurons, kNrcHiddenNeurons,
        kNrcOutputNeurons, kNrcLayout, kNrcComponentType>(
        weights, derivatives, matrixOffset, biasOffset);
    var input = nrcFeaturesToCoopVec(encoded);
    rtxns::mlp::ReLUAct<float16_t, kNrcHiddenNeurons> reluAct;
    var linearAct = rtxns::mlp::LinearAct<float16_t, kNrcOutputNeurons>(float16_t(1.0));
    return mlp.forward(input, reluAct, linearAct);
}

// ---------------------------------------------------------------------------
// Training backward: given the upstream loss gradient (dL/d(linear-output)),
// accumulate weight/bias derivatives into the derivatives buffer. Slang's
// autodiff lowers this to the CoopVector training intrinsics.
// ---------------------------------------------------------------------------
void nrcTrainBackward(
    ByteAddressBuffer weights,
    RWByteAddressBuffer derivatives,
    uint matrixOffset[kNrcHiddenLayers + 1],
    uint biasOffset[kNrcHiddenLayers + 1],
    MlpFeatures features,
    CoopVec<float16_t, kNrcOutputNeurons> upstreamGrad)
{
    typealias ReluT = rtxns::mlp::ReLUAct<float16_t, kNrcHiddenNeurons>;
    typealias LinearT = rtxns::mlp::LinearAct<float16_t, kNrcOutputNeurons>;
    ReluT reluAct;
    LinearT linearAct = LinearT(float16_t(1.0));
    EncodedMlpFeatures encoded = nrcEncode(features);
    var mlp = rtxns::mlp::TrainingMLP<
        float16_t, kNrcHiddenLayers, kNrcInputNeurons, kNrcHiddenNeurons,
        kNrcOutputNeurons, kNrcLayout, kNrcComponentType>(
        weights, derivatives, matrixOffset, biasOffset);
    var input = nrcFeaturesToCoopVec(encoded);
    mlp.backward(input, reluAct, linearAct, upstreamGrad);
}

#endif // NRC_MLP_RTXNS_SLANG
