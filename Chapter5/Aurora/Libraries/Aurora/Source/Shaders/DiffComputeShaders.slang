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

// ============================================================
// Shared resource declarations (must match MainEntryPoints.slang bindings)
// ============================================================

[[vk::binding(1)]] RWTexture2D<float4> gResult : register(u0);

// Path records written by RayGenShader (forward pass).
// Flat float4 buffer — 11 float4s per pixel (DIFF_RECORD_STRIDE = 11).
// Layout matches the writes in MainEntryPoints.slang:
//   [0]  baseColor.xyz, roughness
//   [1]  emissionColor.xyz, emission
//   [2]  specularColor.xyz, specular
//   [3]  specularIOR, specularAnisotropy, metalness, isValid (1=valid)
//   [4]  normal.xyz, 0
//   [5]  tangent.xyz, 0
//   [6]  bitangent.xyz, 0
//   [7]  viewDir.xyz, 0
//   [8]  lightDir.xyz, 0
//   [9]  lightRadiance.xyz, 0
//   [10] throughput.xyz, 0
#define DIFF_RECORD_STRIDE 11
#define MAX_PATH_BOUNCES 1

RWByteAddressBuffer         gPathRecords   : register(u10);
// Gradient buffer: MATERIAL_GRAD_STRIDE floats per pixel (not per material).
// Each pixel thread writes its own gradient contribution without atomics.
// The CPU reduces across pixels after readback.
// Layout: pixel 0: [grad0..grad14], pixel 1: [grad0..grad14], ...
RWByteAddressBuffer         gMaterialGrads : register(u11);
Texture2D<float4>           gTargetImage   : register(t0, space2);

cbuffer gDiffRenderConstants : register(b4)
{
    uint  gScreenWidth;
    uint  gScreenHeight;
    uint  gNumMaterials;
    float _diffPad;
}

#define MATERIAL_GRAD_STRIDE 15

// ============================================================
// GradientAccumShader — Backward Pass
//
// Runs after RayGenShader. For each pixel:
//   1. Reads the PixelPathRecord written by RayGenShader
//   2. Computes loss gradient: dL/d_radiance = 2*(rendered - target)  [L2 loss]
//   3. For each bounce, calls bwd_diff(evaluateMaterial) to get material gradients
//   4. Accumulates gradients into gMaterialGrads via atomic adds
// ============================================================
[shader("compute")]
[numthreads(8, 8, 1)]
void GradientAccumShader(uint3 id : SV_DispatchThreadID)
{
    uint2 screenCoords = id.xy;
    if (screenCoords.x >= gScreenWidth || screenCoords.y >= gScreenHeight)
        return;

    uint pixelIndex = screenCoords.y * gScreenWidth + screenCoords.x;

    // ---- Compute loss gradient (L2) ----
    // dL/d_rendered[xy] = 2 * (rendered[xy] - target[xy])
    float3 rendered = gResult[screenCoords].rgb;
    float3 target   = gTargetImage.Load(int3(screenCoords, 0)).rgb;
    float3 lossGrad = 2.0f * (rendered - target);

    // ---- Read path record from flat byte-address buffer ----
    uint byteBase = pixelIndex * DIFF_RECORD_STRIDE * 16; // 16 bytes per float4
    float4 r3 = asfloat(gPathRecords.Load4(byteBase + 3*16)); // specularIOR, specularAnisotropy, metalness, isValid

    // Skip pixels where no bounce was recorded (isValid == 0).
    if (r3.w == 0.0f) return;

    float4 r0  = asfloat(gPathRecords.Load4(byteBase +  0*16)); // baseColor.xyz, roughness
    float4 r1  = asfloat(gPathRecords.Load4(byteBase +  1*16)); // emissionColor.xyz, emission
    float4 r2  = asfloat(gPathRecords.Load4(byteBase +  2*16)); // specularColor.xyz, specular
    float4 r4  = asfloat(gPathRecords.Load4(byteBase +  4*16)); // normal.xyz
    float4 r5  = asfloat(gPathRecords.Load4(byteBase +  5*16)); // tangent.xyz
    float4 r6  = asfloat(gPathRecords.Load4(byteBase +  6*16)); // bitangent.xyz
    float4 r7  = asfloat(gPathRecords.Load4(byteBase +  7*16)); // viewDir.xyz
    float4 r8  = asfloat(gPathRecords.Load4(byteBase +  8*16)); // lightDir.xyz
    float4 r10 = asfloat(gPathRecords.Load4(byteBase + 10*16)); // throughput.xyz

    // Weight the gradient by the path throughput.
    float3 bounceGrad = lossGrad * r10.xyz;

    // ---- Reconstruct Material from recorded float4 fields ----
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

    // ---- Reconstruct ShadingData (only fields used by evaluateMaterial) ----
    ShadingData shading;
    shading.normal    = r4.xyz;
    shading.tangent   = r5.xyz;
    shading.bitangent = r6.xyz;

    // ================================================================
    // Run the backward pass via Slang AD
    //
    // bwd_diff(evaluateMaterial) is AUTO-GENERATED by Slang from the
    // [Differentiable] annotation on evaluateMaterial() in StandardSurfaceBSDF.slang.
    //
    // After this call:
    //   dpMat.d.baseColor   = dL/d_baseColor for this bounce
    //   dpMat.d.roughness   = dL/d_roughness for this bounce
    //   etc.
    // ================================================================
    var dpMat = diffPair(mat, Material.dzero());
    bwd_diff(evaluateMaterial)(
        dpMat,       // inout DifferentialPair<Material>
        shading,     // no_diff ShadingData (geometry)
        r7.xyz,      // no_diff float3 Vw (viewDir)
        r8.xyz,      // no_diff float3 Lw (lightDir)
        bounceGrad   // float3: dL/d_radiance (incoming gradient)
    );

    // Emission is computed by shadeEmission() separately from evaluateMaterial().
    // We must also backprop through it so emissionColor/emission gradients are populated.
    // Slang accumulates into dpMat.d, so calling bwd_diff(shadeEmission) adds to the
    // existing gradients from evaluateMaterial above.
    bwd_diff(shadeEmission)(dpMat, bounceGrad);

    // ---- Write per-pixel gradients to gMaterialGrads ----
    // No atomics needed: each pixel writes to its own slot.
    // CPU reduces across pixels after readback.
    // Slot layout: pixelIndex * MATERIAL_GRAD_STRIDE * 4 bytes + field_offset * 4
    uint gradByteBase = pixelIndex * MATERIAL_GRAD_STRIDE * 4;
    gMaterialGrads.Store(gradByteBase +  0*4, asuint(dpMat.d.baseColor.x));
    gMaterialGrads.Store(gradByteBase +  1*4, asuint(dpMat.d.baseColor.y));
    gMaterialGrads.Store(gradByteBase +  2*4, asuint(dpMat.d.baseColor.z));
    gMaterialGrads.Store(gradByteBase +  3*4, asuint(dpMat.d.specularRoughness));
    gMaterialGrads.Store(gradByteBase +  4*4, asuint(dpMat.d.metalness));
    gMaterialGrads.Store(gradByteBase +  5*4, asuint(dpMat.d.emissionColor.x));
    gMaterialGrads.Store(gradByteBase +  6*4, asuint(dpMat.d.emissionColor.y));
    gMaterialGrads.Store(gradByteBase +  7*4, asuint(dpMat.d.emissionColor.z));
    gMaterialGrads.Store(gradByteBase +  8*4, asuint(dpMat.d.emission));
    gMaterialGrads.Store(gradByteBase +  9*4, asuint(dpMat.d.specular));
    gMaterialGrads.Store(gradByteBase + 10*4, asuint(dpMat.d.specularColor.x));
    gMaterialGrads.Store(gradByteBase + 11*4, asuint(dpMat.d.specularColor.y));
    gMaterialGrads.Store(gradByteBase + 12*4, asuint(dpMat.d.specularColor.z));
    gMaterialGrads.Store(gradByteBase + 13*4, asuint(dpMat.d.specularIOR));
    gMaterialGrads.Store(gradByteBase + 14*4, asuint(dpMat.d.specularAnisotropy));
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
