// NRC compatibility adapter for the inherited path-tracing shader.
//
// This is intentionally the only NRC-local textual include. The renderer's
// material system is preprocessor-configured, whereas imported Slang modules
// have isolated preprocessor state. Keep renderer types on the caller side and
// pass semantic NRC values through this narrow boundary.

#ifndef NRC_RENDERER_ADAPTER_H_SLANG
#define NRC_RENDERER_ADAPTER_H_SLANG

import nrc;

// Descriptor set 3 is a frozen host ABI (see nrc_pass.cpp).
[[vk::binding(0, 3)]] RWStructuredBuffer<nrc::TrainingRecord> g_trainingRecords;
[[vk::binding(1, 3)]] RWStructuredBuffer<nrc::QueryRecord> g_queryRecords;
[[vk::binding(2, 3)]] RWStructuredBuffer<uint> g_trainingCount;
[[vk::binding(3, 3)]] RWStructuredBuffer<uint> g_queryCount;
[[vk::binding(4, 3)]] RWStructuredBuffer<float4> g_nrcContrib;
[[vk::binding(5, 3)]] RWStructuredBuffer<float4> g_pathPrefix;

void nrcEmitQuery(
    nrc::NrcVertexFeatures features,
    uint pixelIndex,
    float3 pathThroughput,
    uint capacity)
{
    // The atomic returns this invocation's reserved slot; inter-invocation order
    // is unspecified. The counter counts attempted appends, including overflow,
    // so consumers must bound it by capacity before reading records.
    uint index;
    InterlockedAdd(g_queryCount[0], 1u, index);
    if(index >= capacity)
    {
        return;
    }

    g_queryRecords[index] = nrc::makeQueryRecord(
        features, pixelIndex, pathThroughput);
}

void nrcEmitTrainingStack(
    nrc::PendingTrainingRecord stack[nrc::kMaxPendingTrainingRecords],
    int stackSize,
    nrc::NrcVertexFeatures suffixEndFeatures,
    bool isUnbiasedRr,
    bool suffixTerminated,
    uint capacity)
{
    // A cache endpoint is chosen before creating a record at that vertex.
    // Every stored record therefore has a strictly later endpoint, including
    // paths which exceeded the finite recording stack.
    if(stackSize <= 0)
    {
        return;
    }

    [loop]
    for(int stackIndex = 0; stackIndex < stackSize; ++stackIndex)
    {
        // Count attempted appends even when full. Consumers cap the count to
        // capacity; records from different paths have no guaranteed ordering.
        uint recordIndex;
        InterlockedAdd(g_trainingCount[0], 1u, recordIndex);
        if(recordIndex >= capacity)
        {
            continue;
        }

        g_trainingRecords[recordIndex] = nrc::completeTrainingRecord(
            stack[stackIndex],
            suffixEndFeatures,
            isUnbiasedRr,
            suffixTerminated);
    }
}

void nrcFoldContribution(
    inout nrc::PendingTrainingRecord stack[nrc::kMaxPendingTrainingRecords],
    int stackSize, float3 localRadiance)
{
    // localRadiance excludes camera-path throughput. Each pending record applies
    // its own throughput measured from that record's starting vertex.
    [loop]
    for(int i = 0; i < stackSize; ++i)
        nrc::accumulateTrainingContribution(stack[i], localRadiance);
}

void nrcAdvanceThroughput(
    inout nrc::PendingTrainingRecord stack[nrc::kMaxPendingTrainingRecords],
    int stackSize, float3 factor)
{
    // Apply each BSDF/PDF, segment attenuation, or RR-survival multiplier once
    // to every pending record, alongside the camera path's throughput update.
    // Passing the cumulative camera throughput here would double-count transport.
    [loop]
    for(int i = 0; i < stackSize; ++i)
        nrc::advanceTrainingThroughput(stack[i], factor);
}

#endif // NRC_RENDERER_ADAPTER_H_SLANG
