#include "nrc/nrc_mlp_layout.h"

#include <stdexcept>

namespace nrc {

static uint32_t alignUp(uint32_t v, uint32_t a) { return (v + a - 1) & ~(a - 1); }

// Query the required size in bytes for a weight matrix in the given layout.
// Uses vkConvertCooperativeVectorMatrixNV with null dstData to ask the driver.
static size_t queryMatrixSize(VkDevice device,
                              VkCooperativeVectorMatrixLayoutNV layout,
                              uint32_t rows, uint32_t cols) {
    if (layout == VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV ||
        layout == VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_COLUMN_MAJOR_NV) {
        // Packed row-major / column-major: rows * cols * elementSize, no stride.
        return static_cast<size_t>(rows) * cols * kNrcElementBytes;
    }

    // TrainingOptimal / InferencingOptimal: ask the driver.
    size_t dstSize = 0;

    VkConvertCooperativeVectorMatrixInfoNV info{};
    info.sType            = VK_STRUCTURE_TYPE_CONVERT_COOPERATIVE_VECTOR_MATRIX_INFO_NV;
    info.srcSize          = static_cast<size_t>(rows) * cols * kNrcElementBytes;
    info.srcData.hostAddress = nullptr;
    info.pDstSize         = &dstSize;
    info.dstData.hostAddress = nullptr;   // null => query only
    info.srcComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    info.dstComponentType = VK_COMPONENT_TYPE_FLOAT16_KHR;
    info.numRows          = rows;
    info.numColumns       = cols;
    info.srcLayout        = VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_ROW_MAJOR_NV;
    info.srcStride        = cols * kNrcElementBytes;
    info.dstLayout        = layout;
    info.dstStride        = 0;   // driver-chosen for Optimal layouts

    if (vkConvertCooperativeVectorMatrixNV(device, &info) != VK_SUCCESS)
        throw std::runtime_error("nrc: vkConvertCooperativeVectorMatrixNV size query failed");
    return dstSize;
}

MlpLayout computeMlpLayout(VkDevice device, VkCooperativeVectorMatrixLayoutNV layout) {
    MlpLayout ml;

    for (int i = 0; i < kNrcNumLayers; ++i) {
        const uint32_t inputs  = (i == 0)                  ? kNrcInputNeurons  : kNrcHiddenNeurons;
        const uint32_t outputs = (i == kNrcNumLayers - 1)  ? kNrcOutputNeurons : kNrcHiddenNeurons;
        ml.rows[i] = outputs;
        ml.cols[i] = inputs;
    }

    uint32_t offset = 0;
    for (int i = 0; i < kNrcNumLayers; ++i) {
        // Weight matrix: queried from the driver for Optimal layouts.
        const size_t wBytes = queryMatrixSize(device, layout, ml.rows[i], ml.cols[i]);
        ml.weightSize[i]   = static_cast<uint32_t>(wBytes);

        offset = alignUp(offset, kMatrixAlignment);
        ml.weightOffset[i] = offset;
        offset += ml.weightSize[i];

        // Bias vector: outputs * elementSize.
        ml.biasSize[i]   = ml.rows[i] * kNrcElementBytes;
        offset           = alignUp(offset, kMatrixAlignment);
        ml.biasOffset[i] = offset;
        offset += ml.biasSize[i];
    }
    ml.totalBytes = alignUp(offset, kMatrixAlignment);
    return ml;
}

}  // namespace nrc
