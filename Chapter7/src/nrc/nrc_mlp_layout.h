// NRC MLP weight-buffer layout helper.
//
// The RTXNS MLP.slang reads weights from a ByteAddressBuffer at per-layer
// byte offsets that we compute on the host. For the TrainingOptimal matrix
// layout, the size of each weight matrix depends on the GPU — we query it
// via vkConvertCooperativeVectorMatrixNV with a null destination.
//
// This header exposes a pure function that, given the 5x64 network dimensions
// and the device, returns:
//   - per-layer weight byte offsets    (HIDDEN_LAYERS+1 entries)
//   - per-layer bias byte offsets      (HIDDEN_LAYERS+1 entries)
//   - total byte size of the buffer    (sum of weights + biases, aligned)
// plus the per-layer row-major sizes (which we also need when uploading
// Xavier-initialized weights before they're converted to TrainingOptimal).

#pragma once

#if defined(VK_NO_PROTOTYPES)
  #include <volk.h>
#else
  #include <vulkan/vulkan.h>
#endif
#include <cstdint>
#include <array>

namespace nrc {

// Must match nrc_mlp_rtxns.slang constants.
// B4: input bumped from 14 (raw) to 64 (encoded via nrc_features.slang::nrcEncode).
static constexpr int kNrcInputNeurons  = 64;
static constexpr int kNrcHiddenNeurons = 64;
static constexpr int kNrcOutputNeurons = 3;
static constexpr int kNrcHiddenLayers  = 5;
static constexpr int kNrcNumLayers     = kNrcHiddenLayers + 1;   // 6 transitions

// fp16 scalar size in bytes.
static constexpr uint32_t kNrcElementBytes = 2;

// CoopVec spec mandates 64-byte matrix alignment.
static constexpr uint32_t kMatrixAlignment = 64;

struct MlpLayout {
    // Byte offsets for weight matrices (layout-specific size).
    std::array<uint32_t, kNrcNumLayers> weightOffset{};
    // Byte offsets for bias vectors (simple: outputs * elementSize).
    std::array<uint32_t, kNrcNumLayers> biasOffset{};
    // Per-layer weight sizes (as stored in `layout`, which may or may not be
    // row-major). Used for upload staging.
    std::array<uint32_t, kNrcNumLayers> weightSize{};
    // Per-layer bias sizes (always outputs * kNrcElementBytes).
    std::array<uint32_t, kNrcNumLayers> biasSize{};
    // Rows/cols per layer (outputs, inputs).
    std::array<uint32_t, kNrcNumLayers> rows{};
    std::array<uint32_t, kNrcNumLayers> cols{};
    // Total byte size of the combined weight+bias buffer.
    uint32_t totalBytes = 0;
};

// Compute an MlpLayout for the given matrix layout. `device` is needed for
// TrainingOptimal/InferencingOptimal queries; not used for ROW_MAJOR.
// Returns VK_SUCCESS / throws on failure.
MlpLayout computeMlpLayout(VkDevice device, VkCooperativeVectorMatrixLayoutNV layout);

}  // namespace nrc
