// NRC additions to gltf_pathtrace.slang.
// #include'd near the top of the upstream ray-gen shader. Keeps the host
// diff small — all NRC logic lives in this file.

#ifndef NRC_PATCH_PATHTRACE_SLANG
#define NRC_PATCH_PATHTRACE_SLANG

#include "nrc_common.h.slang"
#include "nrc_features.slang"   // for sphEncode()

// ---------------------------------------------------------------------------
// NRC descriptor set. Bound at set index 3 by the host (see nrc_pass.cpp).
// ---------------------------------------------------------------------------

[[vk::binding(0, 3)]] RWStructuredBuffer<TrainingRecord> g_trainingRecords;
[[vk::binding(1, 3)]] RWStructuredBuffer<QueryRecord>    g_queryRecords;
[[vk::binding(2, 3)]] RWStructuredBuffer<uint>           g_trainingCount;
[[vk::binding(3, 3)]] RWStructuredBuffer<uint>           g_queryCount;
// Cache contribution, written same-frame by the cache-query compute pass
// and summed with g_pathPrefix into the output image by the compose pass.
// NOTE: ray-gen no longer reads this buffer — the compose pass performs
// the final sum. The declaration stays so the descriptor set layout matches.
[[vk::binding(4, 3)]] RWStructuredBuffer<float4>         g_nrcContrib;
// Path-prefix buffer (§1 same-frame dispatch): when NRC is on, ray-gen
// writes the frame's direct-lighting accumulation (`radiance`) here instead
// of returning it as the final pixel value. The compose pass later sums
// g_pathPrefix + g_nrcContrib and writes the result to the output image.
[[vk::binding(5, 3)]] RWStructuredBuffer<float4>         g_pathPrefix;

// ---------------------------------------------------------------------------
// Constants. These come from NRC paper defaults.
//
// §3 (Task 3): kTrainingCapacity is no longer a compile-time constant —
// it's pushed via the ray-gen PC (pushConst.nrcTrainingCapacity) because
// trainingRecordsPerFrame changes between tasks (16384 → 65536). The
// training-ray selection mask is also host-driven now: the host's P-controller
// updates pushConst.nrcTrainingRayThreshold each frame to target ~60k emitted
// records. See nrcIsTrainingRayHashLt below and spec §6 B26.
// ---------------------------------------------------------------------------

static const float kSpreadThresholdC  = 0.01;

// ---------------------------------------------------------------------------
// Area-spread heuristic per NRC paper §5.1 (Bekaert et al. 2003 footprint).
//
// Sub-path spread along x_1..x_n (eq. 3):
//   a(x_1..x_n) = ( Σ_{i=2..n} sqrt( ||x_{i-1} - x_i||² / (p_i · |cos θ_i|) ) )²
//
// where p_i is the BSDF sampling PDF at vertex x_i, and θ_i is the angle
// between the outgoing direction and the normal at x_i.
//
// Primary-vertex footprint as seen from camera (eq. 4):
//   a_0 = ||x_0 - x_1||² / (4π · cos θ_1)
//
// Terminate render ray when a > c·a_0 with c = 0.01.
//
// Specular vertices have delta-PDF (infinite density), so their spread
// contribution (sqrt(d²/(p·cosθ))) goes to zero — specular chains do not
// grow spread, meaning the heuristic cannot fire mid-mirror. That's the
// paper's desired behavior.
// ---------------------------------------------------------------------------

// Running accumulator in the paper is the SQRT-sum (before squaring at the
// end). We track that running sum; compare its square against threshold
// when testing for termination.
float nrcSpreadContribution(float segmentLength, float pdf, float cosTheta) {
    // Delta-specular bounces have pdf effectively infinite — contribution is
    // zero (sqrt of near-zero).
    if (pdf > 1e20)
        return 0.0;
    float denom = max(pdf * max(cosTheta, 1e-3), 1e-8);
    return sqrt(segmentLength * segmentLength / denom);
}

// Returns true if current accumulated spread crosses the threshold
// relative to primary footprint a0.
bool nrcSpreadExceeds(float spreadSqrtSum, float a0) {
    float a = spreadSqrtSum * spreadSqrtSum;
    return a > kSpreadThresholdC * a0;
}

// ---------------------------------------------------------------------------
// §3 (Task 3): host-driven training-ray selection. The hash covers the full
// 32-bit range and is compared against `threshold`; fraction is threshold /
// 2^32. A host-side P-controller adjusts threshold each frame (post warmup)
// to drive emitted records toward trainingRecordsPerFrame. Initial threshold
// is calibrated from trainingPathsTarget / (W·H) · 2^32.
// ---------------------------------------------------------------------------

bool nrcIsTrainingRay(uint2 pix, uint frameIdx, uint threshold) {
    uint h = pix.x * 1973u + pix.y * 9277u + frameIdx * 26699u;
    h = (h ^ (h >> 16)) * 0x85ebca6bu;
    h = (h ^ (h >> 13)) * 0xc2b2ae35u;
    h = h ^ (h >> 16);
    return h < threshold;
}

// Overload for callers that only have a pixel index, not a 2D coord.
bool nrcIsTrainingRayI(uint pixelIdx, uint frameIdx, uint threshold) {
    uint h = pixelIdx * 2654435761u + frameIdx * 26699u;
    h = (h ^ (h >> 16)) * 0x85ebca6bu;
    h = (h ^ (h >> 13)) * 0xc2b2ae35u;
    h = h ^ (h >> 16);
    return h < threshold;
}

// ---------------------------------------------------------------------------
// §4 γ=1/16 unbiased-RR-completed training suffixes.
//
// Paper §3.2: 1-in-16 of the training paths skip the cache-tail term in their
// target — their target is just suffixDirect (full path-traced radiance from
// v_i to absorption/RR-kill). This injects unbiased signal into the otherwise
// self-referential training and prevents drift.
//
// Our architecture runs ALL training paths to natural RR end (no spread-based
// early termination), so trace-behavior is identical between biased and
// unbiased paths. Only the training-target formula differs (handled in
// record_complete). This helper decides per-path; all records emitted from
// one training path share the same flag value (stamped in flags bit 1).
//
// Hash uses DIFFERENT mixing constants from nrcIsTrainingRayI so the two
// decisions decorrelate — otherwise a training path could end up always
// (or never) unbiased.
// ---------------------------------------------------------------------------

bool nrcIsUnbiasedRRPath(uint pixelIdx, uint frameIdx) {
    uint h = pixelIdx * 0x9E3779B9u + frameIdx * 0x85EBCA6Bu;
    h = (h ^ (h >> 16)) * 0xc2b2ae35u;
    h = (h ^ (h >> 13)) * 0x27d4eb2du;
    h = h ^ (h >> 16);
    return (h & 0xFu) == 0u;    // 1-in-16
}

// ---------------------------------------------------------------------------
// Record emission.
// ---------------------------------------------------------------------------

void nrcEmitQuery(
    float3 pos,
    float3 viewDir,
    float3 normal,
    float  roughness,
    float3 diffuse,
    float3 specular,
    uint   pixelIdx,
    float3 throughput,
    uint   capacity)
{
    uint idx;
    InterlockedAdd(g_queryCount[0], 1u, idx);
    if (idx >= capacity) return;

    QueryRecord q;
    q.position       = pos;
    q.viewDirSph     = sphEncode(viewDir);   
    q.normalSph      = sphEncode(normal);   
    q.roughness      = roughness;
    q.diffuseAlbedo  = diffuse;
    q.specularAlbedo = specular;
    q.pixelIdx       = pixelIdx;
    q.pathThroughput = throughput;
    q._pad0 = 0.0; q._pad1 = 0.0;

    g_queryRecords[idx] = q;
}

// ---------------------------------------------------------------------------
// §2 Per-vertex training records.
//
// The ray-gen shader pushes one NrcPendingRecord per non-specular vertex
// visited along a training path. At path end the stack is emitted
// atomically via nrcEmitTrainingStack below.
// ---------------------------------------------------------------------------

struct NrcPendingRecord {
    float3 position;
    float2 viewDirSph;
    float2 normalSph;
    float  roughness;
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float3 suffixDirect;        // accumulates Σ_{j>i} T_{i->j} · direct(v_j)
    // T_eye after scattering from v_i into viewDirSph. 
    float3 startThroughputEye;
    uint   vertexIndex;         // v_0, v_1, ... for debug (encoded in flags)
};


// Preserve the early training vertices so the cache view trains what it shows.
static const int kNrcMaxStack = 8;

// Atomic emission. Reserve one record per stack entry so the first `capacity`
// records are densely filled even when the frame overflows the pool. Per-entry
// suffixThroughput is computed here as T_{i -> last} = lastThroughputEye /
// startThroughputEye (componentwise), both measured after their vertex scatter.
void nrcEmitTrainingStack(
    NrcPendingRecord stack[kNrcMaxStack],
    int    stackSize,
    float3 lastPos,
    float2 lastViewSph,
    float2 lastNormSph,
    float  lastRoughness,
    float3 lastDiffuse,
    float3 lastSpecular,
    float3 lastThroughputEye,
    bool   isUnbiasedRR,
    bool   suffixTerminated,
    uint   capacity)
{
    if (stackSize <= 0) return;

    [loop]
    for (int k = 0; k < stackSize; ++k) {
        uint idx;
        InterlockedAdd(g_trainingCount[0], 1u, idx);
        if (idx >= capacity) continue;

        TrainingRecord r;
        r.position       = stack[k].position;
        r.viewDirSph     = stack[k].viewDirSph;
        r.normalSph      = stack[k].normalSph;
        r.roughness      = stack[k].roughness;
        r.diffuseAlbedo  = stack[k].diffuseAlbedo;
        r.specularAlbedo = stack[k].specularAlbedo;
        r.suffixDirect   = stack[k].suffixDirect;
        // T_{i -> last}; componentwise ratio with clamped denominator.
        r.suffixThroughput = lastThroughputEye
                            / max(stack[k].startThroughputEye, float3(1e-6));
        r.suffixEndPosition   = lastPos;
        r.suffixEndViewDirSph = lastViewSph;
        r.suffixEndNormalSph  = lastNormSph;
        r.suffixEndRoughness  = lastRoughness;
        r.suffixEndDiffuse    = lastDiffuse;
        r.suffixEndSpecular   = lastSpecular;
        r.flags = 0x1u                                          // bit 0 = valid
                | (isUnbiasedRR ? 0x2u : 0x0u)                  // bit 1 = §4 flag (always 0 in Task 2)
                | ((stack[k].vertexIndex & 0x3Fu) << 2u)        // bits 2..7 = vertex index
                | (suffixTerminated ? 0x100u : 0x0u);           // bit 8 = no MLP tail beyond suffix
        g_trainingRecords[idx] = r;
    }
}

#endif // NRC_PATCH_PATHTRACE_SLANG
