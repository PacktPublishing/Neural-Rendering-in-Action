// Feature packing: turn a vertex (position, directions, material) into the
// 14-scalar input vector the MLP consumes. Same packing is used everywhere:
//   - Render-ray cache query (from QueryRecord)
//   - Training-ray query at v_i  (from TrainingRecord prefix fields)
//   - Training-ray query at v_last (from TrainingRecord suffixEnd* fields)
//
// Encoding:
//   index 0-2   : position xyz
//   index 3-4   : view direction, spherical (theta,phi) in [0,1]^2  (B13)
//   index 5-6   : normal,          spherical (theta,phi) in [0,1]^2  (B13)
//   index 7     : roughness
//   index 8-10  : diffuse albedo
//   index 11-13 : specular albedo

#ifndef NRC_FEATURES_SLANG
#define NRC_FEATURES_SLANG

#include "nrc_common.h.slang"

// ---------------------------------------------------------------------------
// Spherical-coordinate direction encoding (B13, paper Table 1).
// Maps a unit float3 to (theta, phi) normalized to [0, 1]^2.
//   theta = acos(z) / pi                 in [0, 1]
//   phi   = atan2(y, x) / (2*pi) + 0.5   in [0, 1]
// ---------------------------------------------------------------------------
float2 sphEncode(float3 v) {
    v = normalize(v);
    float theta = acos(clamp(v.z, -1.0, 1.0)) * (1.0 / 3.14159265);
    float phi   = atan2(v.y, v.x) * (1.0 / (2.0 * 3.14159265)) + 0.5;
    return float2(theta, phi);
}

MlpFeatures packFeatures(
    float3 pos,
    float2 viewSph,   // (theta, phi) in [0, 1]^2
    float2 normSph,   // (theta, phi) in [0, 1]^2
    float  roughness,
    float3 diffuse,
    float3 specular)
{
    MlpFeatures out;
    out.f[0]  = pos.x;      out.f[1]  = pos.y;      out.f[2]  = pos.z;
    out.f[3]  = viewSph.x;  out.f[4]  = viewSph.y;
    out.f[5]  = normSph.x;  out.f[6]  = normSph.y;
    // paper Table 1 — roughness is passed as (1 - exp(-r)) so that
    // unbounded physical roughness is compressed to [0, 1) before encoding.
    // For glTF materials roughness is already in [0, 1], so this is a mild
    // non-linear warp that still helps the network distinguish perfectly-
    // smooth (0 -> 0) from slightly-rough (0.1 -> 0.095) vs rough (1 -> 0.63).
    out.f[7]  = 1.0 - exp(-roughness);
    out.f[8]  = diffuse.x;  out.f[9]  = diffuse.y;  out.f[10] = diffuse.z;
    out.f[11] = specular.x; out.f[12] = specular.y; out.f[13] = specular.z;
    return out;
}

MlpFeatures packFromQuery(QueryRecord q) {
    return packFeatures(q.position, q.viewDirSph, q.normalSph, q.roughness,
                        q.diffuseAlbedo, q.specularAlbedo);
}

MlpFeatures packFromTrainingStart(TrainingRecord r) {
    return packFeatures(r.position, r.viewDirSph, r.normalSph, r.roughness,
                        r.diffuseAlbedo, r.specularAlbedo);
}

MlpFeatures packFromTrainingEnd(TrainingRecord r) {
    return packFeatures(r.suffixEndPosition, r.suffixEndViewDirSph,
                        r.suffixEndNormalSph, r.suffixEndRoughness,
                        r.suffixEndDiffuse, r.suffixEndSpecular);
}

// Octahedral unit-vector encoding. Maps a unit float3 to [-1,1]² with low
// distortion. Cited from Cigolle et al. 2014, "A Survey of Efficient
// Representations of Independent Unit Vectors."
//
// normalize the input first. Callers pass directions that are
// "almost unit" after interpolation + rounding; the divide-by-L1-norm
// assumes unit input, so a near-unit input produces a slightly-off
// encoding.
float2 octEncode(float3 v) {
    v = normalize(v);
    v /= (abs(v.x) + abs(v.y) + abs(v.z));
    float2 e = v.xy;
    if (v.z < 0.0) {
        e = (1.0 - abs(e.yx))
          * float2(v.x >= 0.0 ? 1.0 : -1.0,
                   v.y >= 0.0 ? 1.0 : -1.0);
    }
    return e;
}

// ---------------------------------------------------------------------------
// Feature encoding per NRC paper §4.2
// ---------------------------------------------------------------------------

// Integrated quartic CDF used by one-blob encoding. Each output bin receives the quartic mass overlapping
// that bin's [left,right] interval rather than a point-sampled center value.
float nrcQuarticCdf(float x, float invRadius) {
    float u  = x * invRadius;
    float u2 = u * u;
    float u4 = u2 * u2;
    return clamp(0.9375 * u * (1.0 - (2.0 / 3.0) * u2 + (1.0 / 5.0) * u4) + 0.5, 0.0, 1.0);
}

// One-blob encoding: maps a scalar s in [0, 1] to four integrated quartic
// bins: [0,.25], [.25,.5], [.5,.75], [.75,1]. Paper section 4.2 replaces
// the original Gaussian primitive with this cheaper quartic kernel.
void nrcOneBlob4(float s, out float b0, out float b1, out float b2, out float b3) {
    float x = saturate(s);
    b0 = nrcQuarticCdf(0.25 - x, 4.0) - nrcQuarticCdf(0.00 - x, 4.0);
    b1 = nrcQuarticCdf(0.50 - x, 4.0) - nrcQuarticCdf(0.25 - x, 4.0);
    b2 = nrcQuarticCdf(0.75 - x, 4.0) - nrcQuarticCdf(0.50 - x, 4.0);
    b3 = nrcQuarticCdf(1.00 - x, 4.0) - nrcQuarticCdf(0.75 - x, 4.0);
}

// Frequency encoding: L=12 sin-only frequencies. Paper uses triangle-wave
// approximation of sin for speed; we use real sin (fast enough on a 5080).
// Output: 12 values per scalar. With 3 position components -> 36 total.
// Paper §4.2: "sin(2^ℓ π x), ℓ = 0..11; cosine dropped (no accuracy loss)."
void nrcFreqEncodeScalar(float s, out float out_[12]) {
    float phase = 3.14159265 * s;
    float f = 1.0;
    [ForceUnroll]
    for (int k = 0; k < 12; ++k) {
        out_[k] = sin(f * phase);
        f *= 2.0;
    }
}

// Encode raw features (14 scalars) into the 64-dim MLP input vector.
EncodedMlpFeatures nrcEncode(MlpFeatures raw) {
    EncodedMlpFeatures enc;

    // Clear to 0 first (most bins are 0 for a given input; pad slots too).
    [ForceUnroll]
    for (int k = 0; k < 64; ++k) enc.f[k] = 0.0;

    // [0..35] Position: 12 sin-only frequencies × 3 axes
    float freqsX[12]; nrcFreqEncodeScalar(raw.f[0], freqsX);
    float freqsY[12]; nrcFreqEncodeScalar(raw.f[1], freqsY);
    float freqsZ[12]; nrcFreqEncodeScalar(raw.f[2], freqsZ);
    [ForceUnroll]
    for (int k = 0; k < 12; ++k) {
        enc.f[0 + k]  = freqsX[k];
        enc.f[12 + k] = freqsY[k];
        enc.f[24 + k] = freqsZ[k];
    }

    // [36..43] View sph (theta,phi) already in [0,1]^2 -> one-blob × 4.
    // spherical coords; no [-1,1]->[0,1] remap needed.
    nrcOneBlob4(raw.f[3], enc.f[36], enc.f[37], enc.f[38], enc.f[39]);
    nrcOneBlob4(raw.f[4], enc.f[40], enc.f[41], enc.f[42], enc.f[43]);

    // [44..51] Normal sph (theta,phi) already in [0,1]^2 -> one-blob × 4.
    nrcOneBlob4(raw.f[5], enc.f[44], enc.f[45], enc.f[46], enc.f[47]);
    nrcOneBlob4(raw.f[6], enc.f[48], enc.f[49], enc.f[50], enc.f[51]);

    // [52..55] Roughness (already mapped to 1-exp(-r) in packFeatures) -> one-blob × 4
    nrcOneBlob4(raw.f[7], enc.f[52], enc.f[53], enc.f[54], enc.f[55]);

    // [56..58] Diffuse raw
    enc.f[56] = raw.f[8];
    enc.f[57] = raw.f[9];
    enc.f[58] = raw.f[10];

    // [59..61] Specular raw
    enc.f[59] = raw.f[11];
    enc.f[60] = raw.f[12];
    enc.f[61] = raw.f[13];

    // [62..63] Pad with 1 (paper's "implicit bias" trick)
    enc.f[62] = 1.0;
    enc.f[63] = 1.0;

    return enc;
}

#endif // NRC_FEATURES_SLANG
