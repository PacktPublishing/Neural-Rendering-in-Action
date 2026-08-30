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
    {
        // No surface hit: write an all-zero record (coverage flag 0 in slot [15]).
        // This keeps the whole buffer deterministic on every dispatch — no stale
        // data from a previous frame
        uint clearBase = pixelIndex * MATERIAL_GRAD_STRIDE * 4;
        gMaterialGrads.Store4(clearBase + 0*16, uint4(0u, 0u, 0u, 0u));
        gMaterialGrads.Store4(clearBase + 1*16, uint4(0u, 0u, 0u, 0u));
        gMaterialGrads.Store4(clearBase + 2*16, uint4(0u, 0u, 0u, 0u));
        gMaterialGrads.Store4(clearBase + 3*16, uint4(0u, 0u, 0u, 0u));
        return;
    }

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
        float4 r9  = asfloat(gPathRecords.Load4(byteBase +  9*16));
        float4 r10 = asfloat(gPathRecords.Load4(byteBase + 10*16));
        float4 r11 = asfloat(gPathRecords.Load4(byteBase + 11*16));
        float4 r12 = asfloat(gPathRecords.Load4(byteBase + 12*16));

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
        // Only differentiate the BRDF when the forward pass actually emitted a MIS
        // material shadow ray (r8.w). Otherwise r8.xyz is a placeholder direction that
        // contributed nothing to the image, and differentiating there would fabricate a
        // gradient for a path segment that was never shaded.
        //
        // The seed is scaled by r9 (visibility * misWeight * envRadiance / materialPDF),
        // the non-BSDF factors of the estimator. Without it the BSDF derivative would be
        // reconstructing throughput * bsdf rather than the contribution the forward pass
        // actually made, which is throughput * lightScale * bsdf.
        var dpBrdf = diffPair(mat, Material.dzero());
        if (r8.w != 0.0f)
            bwd_diff(evaluateMaterial)(dpBrdf, shading, r7.xyz, r8.xyz, bounceGrad * r9.xyz);

        // The directional light contributes "throughput * visibility * radiance * bsdf(Ldir)"
        // in the forward pass, independently of the environment MIS term above, so it needs
        // its own backward call seeded with its own scale (r12) at its own direction (r11).
        var dpDirLight = diffPair(mat, Material.dzero());
        if (r11.w != 0.0f)
            bwd_diff(evaluateMaterial)(dpDirLight, shading, r7.xyz, r11.xyz, bounceGrad * r12.xyz);

        var dpEmit = diffPair(mat, Material.dzero());
        bwd_diff(shadeEmission)(dpEmit, bounceGrad);

        // Accumulate this bounce's gradients (sum of BRDF + emission contributions).
        totalGrad.baseColor          += dpBrdf.d.baseColor          + dpDirLight.d.baseColor + dpEmit.d.baseColor;
        totalGrad.specularRoughness  += dpBrdf.d.specularRoughness  + dpDirLight.d.specularRoughness + dpEmit.d.specularRoughness;
        totalGrad.metalness          += dpBrdf.d.metalness          + dpDirLight.d.metalness + dpEmit.d.metalness;
        totalGrad.emissionColor      += dpBrdf.d.emissionColor      + dpDirLight.d.emissionColor + dpEmit.d.emissionColor;
        totalGrad.emission           += dpBrdf.d.emission           + dpDirLight.d.emission + dpEmit.d.emission;
        totalGrad.specular           += dpBrdf.d.specular           + dpDirLight.d.specular + dpEmit.d.specular;
        totalGrad.specularColor      += dpBrdf.d.specularColor      + dpDirLight.d.specularColor + dpEmit.d.specularColor;
        totalGrad.specularIOR        += dpBrdf.d.specularIOR        + dpDirLight.d.specularIOR + dpEmit.d.specularIOR;
        totalGrad.specularAnisotropy += dpBrdf.d.specularAnisotropy + dpDirLight.d.specularAnisotropy + dpEmit.d.specularAnisotropy;
    }

    // ---- Write per-pixel gradients to gMaterialGrads ----
    // MATERIAL_GRAD_STRIDE is padded to 16 floats, so gradByteBase is always
    // 16-byte aligned and the record can be written as four Store4 calls.
    // Slot [15] is the coverage flag: 1 for pixels with a recorded surface hit
    // (even if every gradient is zero, e.g. fully shadowed), 0 otherwise. The
    // CPU reduction uses it as the valid-pixel mask so the AD mean and the
    // tests' alpha-masked FD mean share the same denominator.
    uint gradByteBase = pixelIndex * MATERIAL_GRAD_STRIDE * 4;
    gMaterialGrads.Store4(gradByteBase + 0*16, asuint(float4(
        totalGrad.baseColor, totalGrad.specularRoughness)));
    gMaterialGrads.Store4(gradByteBase + 1*16, asuint(float4(
        totalGrad.metalness, totalGrad.emissionColor)));
    gMaterialGrads.Store4(gradByteBase + 2*16, asuint(float4(
        totalGrad.emission, totalGrad.specular, totalGrad.specularColor.xy)));
    gMaterialGrads.Store4(gradByteBase + 3*16, asuint(float4(
        totalGrad.specularColor.z, totalGrad.specularIOR,
        totalGrad.specularAnisotropy, 1.0f)));
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
