// Differentiable rendering compute shaders.
//
// These entry points are compiled SEPARATELY from MainEntryPoints.slang because they are
// compute shaders ([shader("compute")]) and cannot coexist with ray tracing entry points
// in a lib_6_3 shader library. They are compiled with profile cs_6_6 instead.
//
// Compiled only when ENABLE_DIFFERENTIABLE_RENDERING is defined.
//
// Entry points:
//   GradientAccumShader — backward pass: reads gPathRecords, calls bwd_diff(evaluateMaterial),
//                         writes material gradients to gMaterialGrads.
//   TestDiffAD          — compile-time AD chain verification (never dispatched at runtime, used only for compilation works).

#include "Options.slang"
#include "PathTracingCommon.slang"
#include "ShadeFunctions.slang"
#include "DefaultMaterial.slang"
#include "GlobalBufferAccessors.slang"
#include "InstancePipelineState.slang"
#include "EvaluateMaterial.slang"

#include "DiffRenderingConstants.slang"

// ============================================================
// Shared resource declarations (must match MainEntryPoints.slang bindings)
// ============================================================

[[vk::binding(1)]] RWTexture2D<float4> gResult : register(u0);

RWByteAddressBuffer         gPathRecords   : register(u10);
RWByteAddressBuffer         gMaterialGrads : register(u11);
Texture2D<float4>           gTargetImage   : register(t0, space2);

cbuffer gDiffRenderConstants : register(b4)
{
    uint  gScreenWidth;
    uint  gScreenHeight;
    uint  gNumMaterials;
    float _diffPad;
}

// ============================================================
// GradientAccumShader — Backward Pass (multi-bounce)
//
// Runs after RayGenShader. For each pixel:
//   1. Reads the per-pixel header to get numValidBounces
//   2. Computes loss gradient: dL/d_radiance = 2*(rendered - target)  [L2 loss]
//   3. For each recorded bounce, calls bwd_diff(evaluateMaterial) and
//      bwd_diff(shadeEmission) weighted by that bounce's throughput
//   4. Sums gradients across all bounces into gMaterialGrads (per-pixel slot)
// ============================================================
[shader("compute")]
[numthreads(8, 8, 1)]
void GradientAccumShader(uint3 id : SV_DispatchThreadID)
{
    uint2 screenCoords = id.xy;
    if (screenCoords.x >= gScreenWidth || screenCoords.y >= gScreenHeight)
        return;

    uint pixelIndex = screenCoords.y * gScreenWidth + screenCoords.x;

    // Read per-pixel header: numValidBounces.
    uint pixelByteBase = pixelIndex * DIFF_PIXEL_TOTAL_STRIDE * 16;
    uint numBounces = asuint(gPathRecords.Load(pixelByteBase));
    if (numBounces == 0u)
        return;

    // Clamp to the compile-time maximum.
    numBounces = min(numBounces, (uint)MAX_PATH_BOUNCES);

    // ---- Compute loss gradient (L2) ----
    float3 rendered = gResult[screenCoords].rgb;
    float3 target   = gTargetImage.Load(int3(screenCoords, 0)).rgb;
    float3 lossGrad = 2.0f * (rendered - target);

    // Accumulated gradient across all bounces for this pixel.
    Material.Differential totalGrad = Material.dzero();

    for (uint b = 0; b < numBounces; b++)
    {
        uint byteBase = pixelByteBase + (DIFF_PIXEL_HEADER_SIZE + b * DIFF_RECORD_STRIDE) * 16;

        float4 r3 = asfloat(gPathRecords.Load4(byteBase + 3*16));
        if (r3.w == 0.0f)
            continue;

        float4 r0  = asfloat(gPathRecords.Load4(byteBase +  0*16));
        float4 r1  = asfloat(gPathRecords.Load4(byteBase +  1*16));
        float4 r2  = asfloat(gPathRecords.Load4(byteBase +  2*16));
        float4 r4  = asfloat(gPathRecords.Load4(byteBase +  4*16));
        float4 r5  = asfloat(gPathRecords.Load4(byteBase +  5*16));
        float4 r6  = asfloat(gPathRecords.Load4(byteBase +  6*16));
        float4 r7  = asfloat(gPathRecords.Load4(byteBase +  7*16));
        float4 r8  = asfloat(gPathRecords.Load4(byteBase +  8*16));
        float4 r10 = asfloat(gPathRecords.Load4(byteBase + 10*16));

        float3 bounceGrad = lossGrad * r10.xyz;

        Material mat = defaultMaterial();
        mat.baseColor          = r0.xyz;
        mat.specularRoughness  = r0.w;
        mat.emissionColor      = r1.xyz;
        mat.emission           = r1.w;
        mat.specularColor      = r2.xyz;
        mat.specular           = r2.w;
        mat.specularIOR        = r3.x;
        mat.specularAnisotropy = r3.y;
        mat.metalness          = r3.z;

        ShadingData shading;
        shading.normal    = r4.xyz;
        shading.tangent   = r5.xyz;
        shading.bitangent = r6.xyz;

        // Backward pass via Slang AD — each bounce is independent (detached sampling).
        // IMPORTANT: use separate DifferentialPairs for each bwd_diff call because
        // bwd_diff overwrites .d rather than accumulating. Sharing a single pair
        // would cause the second call to zero out gradients from the first.
        var dpBrdf = diffPair(mat, Material.dzero());
        bwd_diff(evaluateMaterial)(dpBrdf, shading, r7.xyz, r8.xyz, bounceGrad);

        var dpEmit = diffPair(mat, Material.dzero());
        bwd_diff(shadeEmission)(dpEmit, bounceGrad);

        // Accumulate this bounce's gradients (sum of BRDF + emission contributions).
        totalGrad.baseColor          += dpBrdf.d.baseColor          + dpEmit.d.baseColor;
        totalGrad.specularRoughness  += dpBrdf.d.specularRoughness  + dpEmit.d.specularRoughness;
        totalGrad.metalness          += dpBrdf.d.metalness          + dpEmit.d.metalness;
        totalGrad.emissionColor      += dpBrdf.d.emissionColor      + dpEmit.d.emissionColor;
        totalGrad.emission           += dpBrdf.d.emission           + dpEmit.d.emission;
        totalGrad.specular           += dpBrdf.d.specular           + dpEmit.d.specular;
        totalGrad.specularColor      += dpBrdf.d.specularColor      + dpEmit.d.specularColor;
        totalGrad.specularIOR        += dpBrdf.d.specularIOR        + dpEmit.d.specularIOR;
        totalGrad.specularAnisotropy += dpBrdf.d.specularAnisotropy + dpEmit.d.specularAnisotropy;
    }

    // ---- Write per-pixel gradients to gMaterialGrads ----
    uint gradByteBase = pixelIndex * MATERIAL_GRAD_STRIDE * 4;
    gMaterialGrads.Store(gradByteBase +  0*4, asuint(totalGrad.baseColor.x));
    gMaterialGrads.Store(gradByteBase +  1*4, asuint(totalGrad.baseColor.y));
    gMaterialGrads.Store(gradByteBase +  2*4, asuint(totalGrad.baseColor.z));
    gMaterialGrads.Store(gradByteBase +  3*4, asuint(totalGrad.specularRoughness));
    gMaterialGrads.Store(gradByteBase +  4*4, asuint(totalGrad.metalness));
    gMaterialGrads.Store(gradByteBase +  5*4, asuint(totalGrad.emissionColor.x));
    gMaterialGrads.Store(gradByteBase +  6*4, asuint(totalGrad.emissionColor.y));
    gMaterialGrads.Store(gradByteBase +  7*4, asuint(totalGrad.emissionColor.z));
    gMaterialGrads.Store(gradByteBase +  8*4, asuint(totalGrad.emission));
    gMaterialGrads.Store(gradByteBase +  9*4, asuint(totalGrad.specular));
    gMaterialGrads.Store(gradByteBase + 10*4, asuint(totalGrad.specularColor.x));
    gMaterialGrads.Store(gradByteBase + 11*4, asuint(totalGrad.specularColor.y));
    gMaterialGrads.Store(gradByteBase + 12*4, asuint(totalGrad.specularColor.z));
    gMaterialGrads.Store(gradByteBase + 13*4, asuint(totalGrad.specularIOR));
    gMaterialGrads.Store(gradByteBase + 14*4, asuint(totalGrad.specularAnisotropy));
}

// ============================================================
// TestDiffAD — AD chain compile-time verification
//
// Never dispatched at runtime. Exists only to verify that
// bwd_diff(evaluateMaterial) and bwd_diff(shadeEmission) compile
// successfully with the full Aurora BSDF chain.
// ============================================================
[shader("compute")]
[numthreads(1, 1, 1)]
void TestDiffAD(uint3 id : SV_DispatchThreadID)
{
    Material mat          = defaultMaterial();
    mat.baseColor         = float3(0.8f, 0.2f, 0.1f);
    mat.specularRoughness = 0.3f;
    mat.specular          = 1.0f;

    ShadingData shading;
    shading.normal         = float3(0.0f, 1.0f, 0.0f);
    shading.tangent        = float3(1.0f, 0.0f, 0.0f);
    shading.bitangent      = float3(0.0f, 0.0f, 1.0f);
    shading.position       = float3(0.0f, 0.0f, 0.0f);
    shading.geomPosition   = float3(0.0f, 0.0f, 0.0f);
    shading.texCoord       = float2(0.0f, 0.0f);
    shading.bitangentWorld = shading.bitangent;
    shading.objectNormal   = shading.normal;
    shading.objectPosition = float3(0.0f, 0.0f, 0.0f);
    shading.objToWorld     = float3x4(1,0,0,0, 0,1,0,0, 0,0,1,0);
    shading.indices        = uint3(0, 0, 0);
    shading.barycentrics   = float3(0.33f, 0.33f, 0.34f);

    float3 Vw = normalize(float3(0.0f, 1.0f, 0.5f));
    float3 Lw = normalize(float3(0.5f, 1.0f, 0.0f));

    var dpMat = diffPair(mat, Material.dzero());
    bwd_diff(evaluateMaterial)(dpMat, shading, Vw, Lw, float3(1.0f, 1.0f, 1.0f));

    Material matEmit      = defaultMaterial();
    matEmit.emission      = 2.0f;
    matEmit.emissionColor = float3(1.0f, 0.5f, 0.0f);

    var dpEmit = diffPair(matEmit, Material.dzero());
    bwd_diff(shadeEmission)(dpEmit, float3(1.0f, 1.0f, 1.0f));

    if (id.x == 0xFFFFFFFF)
    {
        gResult[id.xy] = float4(dpMat.d.baseColor, dpEmit.d.emission);
    }
}
