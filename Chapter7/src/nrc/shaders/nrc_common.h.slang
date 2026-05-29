// Shared NRC struct layout. Consumed by:
//   - gltf_pathtrace.slang (via nrc_patch_pathtrace.slang) to emit records
//   - nrc_cache_query.comp.slang             to read query records
//   - nrc_record_complete.comp.slang         to finish training targets
//   - nrc_train.comp.slang                   to train the MLP
//
// Byte layouts here must stay in lockstep with the C++ allocator sizes in
// nrc_pass.cpp (kTrainingRecordBytes, kQueryRecordBytes). The matching
// static_asserts live in tests/test_feature_pack_sizes.cpp.

#ifndef NRC_COMMON_H_SLANG
#define NRC_COMMON_H_SLANG

// ---------------------------------------------------------------------------
// TrainingRecord — one per non-specular vertex along a training path (§2
// per-vertex emission). 140 bytes, layout-unchanged from Part 2 — the oct
// direction fields were re-interpreted as 2-float spherical (theta, phi),
// and the uint `valid` was re-interpreted as bitfield `flags` (B13).
//
//   Offsets (bytes)    Field                             Size
//   -----------------  --------------------------------  ----
//    0                 position            (float3)       12
//   12                 viewDirSph          (float2)        8   (theta,phi in [0,1])
//   20                 normalSph           (float2)        8   (theta,phi in [0,1])
//   28                 roughness           (float)         4
//   32                 diffuseAlbedo       (float3)       12
//   44                 specularAlbedo      (float3)       12
//   56                 suffixDirect        (float3)       12
//   68                 suffixThroughput    (float3)       12   T_{i -> last}
//   80                 suffixEndPosition   (float3)       12
//   92                 suffixEndViewDirSph (float2)        8
//  100                 suffixEndNormalSph  (float2)        8
//  108                 suffixEndRoughness  (float)         4
//  112                 suffixEndDiffuse    (float3)       12
//  124                 suffixEndSpecular   (float3)       12
//  136                 flags               (uint)          4
//                                                        ---
//                                                        140
//
// `flags` bit layout:
//   bit 0      = valid (1 = complete record, 0 = empty slot — skip it)
//   bit 1      = isUnbiasedRR (§4; Task 2 always emits 0)
//   bits 2..7  = vertexIndex within this path (0..63, debug-only)
//   bit 8      = suffixTerminated (suffix hit env/light/RR/absorb; no MLP tail)
// ---------------------------------------------------------------------------
struct TrainingRecord {
    float3 position;
    float2 viewDirSph;
    float2 normalSph;
    float  roughness;
    float3 diffuseAlbedo;
    float3 specularAlbedo;

    // §2: suffixDirect = Σ_{j > i} T_{i -> j} · direct(v_j)
    // already throughput-stripped relative to this record's own v_i.
    float3 suffixDirect;
    float3 suffixThroughput;   // T_{i -> last} (componentwise ratio)

    // Features at v_last (end of suffix) — used to query MLP for tail radiance.
    float3 suffixEndPosition;
    float2 suffixEndViewDirSph;
    float2 suffixEndNormalSph;
    float  suffixEndRoughness;
    float3 suffixEndDiffuse;
    float3 suffixEndSpecular;

    // Bitfield; see bit layout comment above.
    uint   flags;
};

// ---------------------------------------------------------------------------
// QueryRecord — one per render-ray termination; 80 bytes.
// Direction fields changed from octahedral to spherical (B13).
//
//   Offsets (bytes)  Field                    Size
//   ---------------  -----------------------  ----
//    0               position   (float3)       12
//   12               viewDirSph (float2)        8   (theta,phi in [0,1])
//   20               normalSph  (float2)        8   (theta,phi in [0,1])
//   28               roughness  (float)         4
//   32               diffuseAlbedo (float3)    12
//   44               specularAlbedo (float3)   12
//   56               pixelIdx   (uint)          4
//   60               pathThroughput (float3)   12   post-BSDF throughput for viewDirSph
//                                              ---
//                                               72
//   +8 bytes padding to 80 (next-multiple-of-16 alignment for std430).
// ---------------------------------------------------------------------------
struct QueryRecord {
    float3 position;
    float2 viewDirSph;
    float2 normalSph;
    float  roughness;
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    uint   pixelIdx;
    float3 pathThroughput;
    float  _pad0;          // explicit pad to 80 bytes
    float  _pad1;
};

// ---------------------------------------------------------------------------
// MlpFeatures — the 14-scalar RAW feature vector (before encoding).
// ---------------------------------------------------------------------------
struct MlpFeatures {
    float f[14];
};

// ---------------------------------------------------------------------------
// EncodedMlpFeatures — the 64-scalar ENCODED vector fed to the MLP.
// Per NRC paper §4.2:
//   - Position [0..35]: 12 sin-only frequencies × 3 axes
//   - View sph (theta,phi) [36..43]: one-blob × 4 bins × 2 scalars
//   - Normal sph (theta,phi) [44..51]: one-blob × 4 bins × 2 scalars
//   - Roughness [52..55]: one-blob × 4 bins × 1 scalar
//   - Diffuse [56..58]: raw
//   - Specular [59..61]: raw
//   - Pad [62..63]: 1.0 (implicit bias)
// ---------------------------------------------------------------------------
struct EncodedMlpFeatures {
    float f[64];
};

#endif // NRC_COMMON_H_SLANG
