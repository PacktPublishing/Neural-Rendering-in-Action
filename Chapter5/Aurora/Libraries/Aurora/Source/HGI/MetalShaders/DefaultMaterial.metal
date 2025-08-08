// Copyright 2025 Autodesk, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __DEFAULT_MATERIAL_H__
#define __DEFAULT_MATERIAL_H__

#include "Geometry.metal"
#include "Material.metal"
#include "GlobalBufferAccessors.metal"

struct MaterialConstants
{
    float base; // Offset:0 Property:base
    packed_float3 baseColor; // Offset:4 Property:base_color
    float diffuseRoughness; // Offset:16 Property:diffuse_roughness
    float metalness; // Offset:20 Property:metalness
    float specular; // Offset:24 Property:specular
    int _padding0;
    packed_float3 specularColor; // Offset:32 Property:specular_color
    float specularRoughness; // Offset:44 Property:specular_roughness
    float specularIOR; // Offset:48 Property:specular_IOR
    float specularAnisotropy; // Offset:52 Property:specular_anisotropy
    float specularRotation; // Offset:56 Property:specular_rotation
    float transmission; // Offset:60 Property:transmission
    packed_float3 transmissionColor; // Offset:64 Property:transmission_color
    float subsurface; // Offset:76 Property:subsurface
    packed_float3 subsurfaceColor; // Offset:80 Property:subsurface_color
    int _padding1;
    packed_float3 subsurfaceRadius; // Offset:96 Property:subsurface_radius
    float subsurfaceScale; // Offset:108 Property:subsurface_scale
    float subsurfaceAnisotropy; // Offset:112 Property:subsurface_anisotropy
    float sheen; // Offset:116 Property:sheen
    int _padding2;
    int _padding3;
    packed_float3 sheenColor; // Offset:128 Property:sheen_color
    float sheenRoughness; // Offset:140 Property:sheen_roughness
    float coat; // Offset:144 Property:coat
    packed_float3 coatColor; // Offset:148 Property:coat_color
    float coatRoughness; // Offset:160 Property:coat_roughness
    float coatAnisotropy; // Offset:164 Property:coat_anisotropy
    float coatRotation; // Offset:168 Property:coat_rotation
    float coatIOR; // Offset:172 Property:coat_IOR
    float coatAffectColor; // Offset:176 Property:coat_affect_color
    float coatAffectRoughness; // Offset:180 Property:coat_affect_roughness
    float emission; // Offset:184 Property:emission
    int _padding4;
    packed_float3 emissionColor; // Offset:192 Property:emission_color
    int _padding5;
    packed_float3 opacity; // Offset:208 Property:opacity
    int thinWalled; // Offset:220 Property:thin_walled
    int hasBaseColorTex; // Offset:224 Property:has_base_color_image
    packed_float2 baseColorTexOffset; // Offset:228 Property:base_color_image_offset
    int _padding6;
    packed_float2 baseColorTexScale; // Offset:240 Property:base_color_image_scale
    float2 baseColorTexPivot; // Offset:248 Property:base_color_image_pivot
    float baseColorTexRotation; // Offset:256 Property:base_color_image_rotation
    int hasSpecularRoughnessTex; // Offset:260 Property:has_specular_roughness_image
    packed_float2 specularRoughnessTexOffset; // Offset:264 Property:specular_roughness_image_offset
    packed_float2 specularRoughnessTexScale; // Offset:272 Property:specular_roughness_image_scale
    packed_float2 specularRoughnessTexPivot; // Offset:280 Property:specular_roughness_image_pivot
    float specularRoughnessTexRotation; // Offset:288 Property:specular_roughness_image_rotation
    int hasEmissionColorTex; // Offset:292 Property:has_emission_color_image
    packed_float2 emissionColorTexOffset; // Offset:296 Property:emission_color_image_offset
    packed_float2 emissionColorTexScale; // Offset:304 Property:emission_color_image_scale
    packed_float2 emissionColorTexPivot; // Offset:312 Property:emission_color_image_pivot
    float emissionColorTexRotation; // Offset:320 Property:emission_color_image_rotation
    int hasOpacityTex; // Offset:324 Property:has_opacity_image
    packed_float2 opacityTexOffset; // Offset:328 Property:opacity_image_offset
    packed_float2 opacityTexScale; // Offset:336 Property:opacity_image_scale
    packed_float2 opacityTexPivot; // Offset:344 Property:opacity_image_pivot
    float opacityTexRotation; // Offset:352 Property:opacity_image_rotation
    int hasNormalTex; // Offset:356 Property:has_normal_image
    packed_float2 normalTexOffset; // Offset:360 Property:normal_image_offset
    packed_float2 normalTexScale; // Offset:368 Property:normal_image_scale
    packed_float2 normalTexPivot; // Offset:376 Property:normal_image_pivot
    float normalTexRotation; // Offset:384 Property:normal_image_rotation
    int _padding7;
    int _padding8;
    int _padding9;
}
;

float Material_base(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.base;
}

packed_float3 Material_baseColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.baseColor;
}

float Material_diffuseRoughness(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.diffuseRoughness;
}

float Material_metalness(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.metalness;
}

float Material_specular(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specular;
}

packed_float3 Material_specularColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularColor;
}

float Material_specularRoughness(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularRoughness;
}

float Material_specularIOR(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularIOR;
}

float Material_specularAnisotropy(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularAnisotropy;
}

float Material_specularRotation(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularRotation;
}

float Material_transmission(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.transmission;
}

packed_float3 Material_transmissionColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.transmissionColor;
}

float Material_subsurface(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.subsurface;
}

packed_float3 Material_subsurfaceColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.subsurfaceColor;
}

packed_float3 Material_subsurfaceRadius(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.subsurfaceRadius;
}

float Material_subsurfaceScale(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.subsurfaceScale;
}

float Material_subsurfaceAnisotropy(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.subsurfaceAnisotropy;
}

float Material_sheen(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.sheen;
}

packed_float3 Material_sheenColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.sheenColor;
}

float Material_sheenRoughness(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.sheenRoughness;
}

float Material_coat(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coat;
}

packed_float3 Material_coatColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coatColor;
}

float Material_coatRoughness(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coatRoughness;
}

float Material_coatAnisotropy(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coatAnisotropy;
}

float Material_coatRotation(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coatRotation;
}

float Material_coatIOR(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coatIOR;
}

float Material_coatAffectColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coatAffectColor;
}

float Material_coatAffectRoughness(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.coatAffectRoughness;
}

float Material_emission(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.emission;
}

packed_float3 Material_emissionColor(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.emissionColor;
}

packed_float3 Material_opacity(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.opacity;
}

int Material_thinWalled(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.thinWalled;
}

int Material_hasBaseColorTex(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.hasBaseColorTex;
}

packed_float2 Material_baseColorTexOffset(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.baseColorTexOffset;
}

packed_float2 Material_baseColorTexScale(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.baseColorTexScale;
}

packed_float2 Material_baseColorTexPivot(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.baseColorTexPivot;
}

float Material_baseColorTexRotation(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.baseColorTexRotation;
}

int Material_hasSpecularRoughnessTex(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.hasSpecularRoughnessTex;
}

packed_float2 Material_specularRoughnessTexOffset(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularRoughnessTexOffset;
}

packed_float2 Material_specularRoughnessTexScale(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularRoughnessTexScale;
}

packed_float2 Material_specularRoughnessTexPivot(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularRoughnessTexPivot;
}

float Material_specularRoughnessTexRotation(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.specularRoughnessTexRotation;
}

int Material_hasEmissionColorTex(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.hasEmissionColorTex;
}

packed_float2 Material_emissionColorTexOffset(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.emissionColorTexOffset;
}

packed_float2 Material_emissionColorTexScale(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.emissionColorTexScale;
}

packed_float2 Material_emissionColorTexPivot(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.emissionColorTexPivot;
}

float Material_emissionColorTexRotation(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.emissionColorTexRotation;
}

int Material_hasOpacityTex(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.hasOpacityTex;
}

packed_float2 Material_opacityTexOffset(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.opacityTexOffset;
}

packed_float2 Material_opacityTexScale(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.opacityTexScale;
}

packed_float2 Material_opacityTexPivot(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.opacityTexPivot;
}

float Material_opacityTexRotation(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.opacityTexRotation;
}

int Material_hasNormalTex(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.hasNormalTex;
}

packed_float2 Material_normalTexOffset(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.normalTexOffset;
}

packed_float2 Material_normalTexScale(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.normalTexScale;
}

packed_float2 Material_normalTexPivot(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.normalTexPivot;
}

float Material_normalTexRotation(MaterialConstants mtl, int /*materialOffset*/) {
    return mtl.normalTexRotation;
}


#define TANGENT_SPACE 0
#define OBJECT_SPACE 1

Material evaluateMaterialFromConstants(const constant InstanceRecord& instanceRecord, int shaderIndex, ShadingData shading, int mtlOffset, thread float3& materialNormal, thread bool& isGeneratedNormal)
{
    constant MetalContext::MaterialConstants* mtlPtr = reinterpret_cast<constant MetalContext::MaterialConstants*> (instanceRecord.material);
    constant MetalContext::MaterialConstants& mtl = *mtlPtr;

    Material material = defaultMaterial();
    material.base                 = mtl.base          ;
    material.baseColor            = mtl.baseColor     ;
    material.diffuseRoughness     = mtl.diffuseRoughness;
    material.metalness            = mtl.metalness     ;
    material.specular             = mtl.specular      ;
    material.specularColor        = mtl.specularColor ;
    material.specularRoughness    = mtl.specularRoughness;
    material.specularIOR          = mtl.specularIOR   ;
    material.specularAnisotropy   = mtl.specularAnisotropy;
    material.specularRotation     = mtl.specularRotation;
    material.transmission         = mtl.transmission  ;
    material.transmissionColor    = mtl.transmissionColor;
    material.subsurface           = mtl.subsurface    ;
    material.subsurfaceColor      = mtl.subsurfaceColor;
    material.subsurfaceRadius     = mtl.subsurfaceRadius;
    material.subsurfaceScale      = mtl.subsurfaceScale;
    material.subsurfaceAnisotropy = mtl.subsurfaceAnisotropy;
    material.sheen                = mtl.sheen         ;
    material.sheenColor           = mtl.sheenColor    ;
    material.sheenRoughness       = mtl.sheenRoughness;
    material.coat                 = mtl.coat          ;
    material.coatColor            = mtl.coatColor     ;
    material.coatRoughness        = mtl.coatRoughness  ;
    material.coatAnisotropy       = mtl.coatAnisotropy    ;
    material.coatRotation         = mtl.coatRotation      ;
    material.coatIOR              = mtl.coatIOR           ;
    material.coatAffectColor      = mtl.coatAffectColor;
    material.coatAffectRoughness  = mtl.coatAffectRoughness;
    material.emission             = mtl.emission      ;
    material.emissionColor        = mtl.emissionColor ;
    material.opacity              = mtl.opacity       ;
    material.thinWalled           = mtl.thinWalled   ;

    // Sample base color from a texture if necessary.
    float4 texCoord = float4(shading.texCoord, 0.0f, 1.0f);
    if (instanceRecord.baseColorTextureIndex >= 0)
    {
        float2 uv =
        applyUVTransform(shading.texCoord, mtl.baseColorTexPivot,
                         mtl.baseColorTexScale,
                         mtl.baseColorTexRotation,
                         mtl.baseColorTexOffset);
        material.baseColor = sampleBaseColorTexture(instanceRecord.baseColorTextureIndex, uv, 0.0f).rgb;
    }

    // Sample specular roughness from a texture if necessary.
    if (instanceRecord.specularRoughnessTextureIndex >= 0)
    {
        float2 uv = applyUVTransform(shading.texCoord,
                                     mtl.specularRoughnessTexPivot,
                                     mtl.specularRoughnessTexScale,
                                     mtl.specularRoughnessTexRotation,
                                     mtl.specularRoughnessTexOffset);

        material.specularRoughness = sampleSpecularRoughnessTexture(instanceRecord.specularRoughnessTextureIndex, uv, 0.0f).r;
    }

    // Sample emission color from a texture if necessary.
    if (instanceRecord.emissionTextureIndex >= 0)
    {
        float2 uv = applyUVTransform(shading.texCoord,
                                     mtl.emissionColorTexPivot,
                                     mtl.emissionColorTexScale,
                                     mtl.emissionColorTexRotation,
                                     mtl.emissionColorTexOffset);
        material.emissionColor = sampleEmissionColorTexture(instanceRecord.emissionTextureIndex, uv, 0.0f).rgb;
    }

    // Sample opacity from a texture if necessary.
    if (instanceRecord.opacityTextureIndex >= 0)
    {
        float2 uv = applyUVTransform(shading.texCoord, mtl.opacityTexPivot,
                                     mtl.opacityTexScale,
                                     mtl.opacityTexRotation,
                                     mtl.opacityTexOffset);
        material.opacity = sampleOpacityTexture(instanceRecord.opacityTextureIndex, uv, 0.0f).rgb;
    }

    // Sample a normal from the normal texture, convert it to an object-space normal, transform to
    // world space, and store it in the output value.
    isGeneratedNormal = false;
    if (instanceRecord.normalTextureIndex >= 0)
    {
        float2 uv = applyUVTransform(shading.texCoord, mtl.normalTexPivot,
                                     mtl.normalTexScale,
                                     mtl.normalTexRotation,
                                     mtl.normalTexOffset);
        float3 normalTexel = sampleNormalTexture(instanceRecord.normalTextureIndex, uv, 0.0f).rgb;
        float3 objectSpaceNormal = calculateNormalFromMap(
                                                          normalTexel, TANGENT_SPACE, 1.0, shading.normal, shading.tangent);
        materialNormal = normalize((shading.objToWorld * objectSpaceNormal).xyz);

        isGeneratedNormal = true;
    }

    material.metalColor = material.baseColor;
    return material;
}
#if DIRECTX

// Indices of default material textures.  Must match order of StandardSurfaceTextures array in MaterialBase.cpp.
#define BASE_COLOR_TEXTURE_INDEX          0
#define SPECULAR_ROUGHNESS_TEXTURE_INDEX  1
#define OPACITY_TEXTURE_INDEX             2
#define NORMAL_TEXTURE_INDEX              3
#define EMISSION_TEXTURE_INDEX            4

float4 sampleBaseColorTexture(int mtlOffset, float2 uv, float level)
{
    return sampleTexture(mtlOffset, BASE_COLOR_TEXTURE_INDEX, uv, level);
}

float4 sampleSpecularRoughnessTexture(int mtlOffset, float2 uv, float level)
{
    return sampleTexture(mtlOffset, SPECULAR_ROUGHNESS_TEXTURE_INDEX, uv, level);
}

float4 sampleNormalTexture(int mtlOffset, float2 uv, float level)
{
    return sampleTexture(mtlOffset, NORMAL_TEXTURE_INDEX, uv, level);
}

float4 sampleEmissionColorTexture(int mtlOffset, float2 uv, float level)
{
    return sampleTexture(mtlOffset, EMISSION_TEXTURE_INDEX, uv, level);
}

float4 sampleOpacityTexture(int mtlOffset, float2 uv, float level)
{
    return sampleTexture(mtlOffset, OPACITY_TEXTURE_INDEX, uv, level);
}

#else
// Vulkan GLSL versions are forward declared and implemented in raw GLSL suffix file.
MaterialConstants getMaterial();
float4 sampleBaseColorTexture(int mtlOffset, float2 uv, float level) {
    return (*gGlobalMaterialTextures)[mtlOffset].sample((*gSamplerArray)[mtlOffset], uv, 0.0f);
}
float4 sampleSpecularRoughnessTexture(int mtlOffset, float2 uv, float level) {
    return (*gGlobalMaterialTextures)[mtlOffset].sample((*gSamplerArray)[mtlOffset], uv, 0.0f);
}
float4 sampleEmissionColorTexture(int mtlOffset, float2 uv, float level) {
    return (*gGlobalMaterialTextures)[mtlOffset].sample((*gSamplerArray)[mtlOffset], uv, 0.0f);
}
float4 sampleOpacityTexture(int mtlOffset, float2 uv, float level) {
    return (*gGlobalMaterialTextures)[mtlOffset].sample((*gSamplerArray)[mtlOffset], uv, 0.0f);
}
float4 sampleNormalTexture(int mtlOffset, float2 uv, float level) {
    return (*gGlobalMaterialTextures)[mtlOffset].sample(gDefaultSampler, uv, 0.0f);
}
#endif

// Normal map spaces definitions.
#define TANGENT_SPACE 0
#define OBJECT_SPACE 1

// Rotate a 2D vector by given number of degrees.
// Based on MaterialX mx_rotate_vector2 functionn.
float2 rotateUV(float2 uv, float amountDegrees)
{
    float rotationRadians = amountDegrees * M_PI / 180.0f;
    float sa = sin(rotationRadians);
    float ca = cos(rotationRadians);
    return float2(ca * uv.x + sa * uv.y, -sa * uv.x + ca * uv.y);
}

// Rotate a 2D vector by given number of degrees.
// Based on MaterialX NG_place2d_vector2 function.
// NOTE: Applies transform in order Scale-Rotation-Translation
float2 applyUVTransform(float2 uv, float2 pivot, float2 scale, float rotate, float2 offset)
{
    float2 subpivot = uv - pivot;
    float2 scaled = subpivot / scale;
    float2 rotated = rotateUV(scaled, rotate);
    float2 translated = rotated - offset;
    float2 addpivot = translated + pivot;
    return addpivot;
}

// Calculate a per-pixel normal from a normal map texel value.
// NOTE: This is based on the MaterialX function mx_normalmap().
float3 calculateNormalFromMap(float3 texelValue, int space, float scale, float3 N, float3 T)
{
    // Remap texel components from [0.0, 1.0] to [-1.0, 1.0].
    float3 v = texelValue * 2.0 - 1.0;

    // If the texel normal is in tangent space, transform it to the coordinate system defined by N
    // and T.
    if (space == TANGENT_SPACE)
    {
        float3 B = normalize(cross(N, T));
        return normalize(T * v.x * scale + B * v.y * scale + N * v.z);
    }

    // Otherwise the texel normal is in object space, and is simply normalized.
    else
    {
        return normalize(v);
    }
}

// Initializes the full set of property values for a material, for the specified shading data.
Material evaluateDefaultMaterial(
                                        ShadingData shading, int headerOffset, thread float3& materialNormal, thread bool& isGeneratedNormal)
{
#if !DIRECTX
    MaterialConstants gGlobalMaterialConstants = getMaterial();
#endif
    int offset = headerOffset + kMaterialHeaderSize;

    // Copy the constant values to the material from the constant buffer.
    Material material;
    material.base = Material_base(gGlobalMaterialConstants, offset);
    material.baseColor = Material_baseColor(gGlobalMaterialConstants, offset);
    material.diffuseRoughness = Material_diffuseRoughness(gGlobalMaterialConstants, offset);
    material.metalness = Material_metalness(gGlobalMaterialConstants, offset);
    material.specular = Material_specular(gGlobalMaterialConstants, offset);
    material.specularColor = Material_specularColor(gGlobalMaterialConstants, offset);
    material.specularRoughness = Material_specularRoughness(gGlobalMaterialConstants, offset);
    material.specularIOR = Material_specularIOR(gGlobalMaterialConstants, offset);
    material.specularAnisotropy = Material_specularAnisotropy(gGlobalMaterialConstants, offset);
    material.specularRotation = Material_specularRotation(gGlobalMaterialConstants, offset);
    material.transmission = Material_transmission(gGlobalMaterialConstants, offset);
    material.transmissionColor = Material_transmissionColor(gGlobalMaterialConstants, offset);
    material.subsurface = Material_subsurface(gGlobalMaterialConstants, offset);
    material.subsurfaceColor = Material_subsurfaceColor(gGlobalMaterialConstants, offset);
    material.subsurfaceRadius = Material_subsurfaceRadius(gGlobalMaterialConstants, offset);
    material.subsurfaceScale = Material_subsurfaceScale(gGlobalMaterialConstants, offset);
    material.subsurfaceAnisotropy = Material_subsurfaceAnisotropy(gGlobalMaterialConstants, offset);
    material.sheen = Material_sheen(gGlobalMaterialConstants, offset);
    material.sheenColor = Material_sheenColor(gGlobalMaterialConstants, offset);
    material.sheenRoughness = Material_sheenRoughness(gGlobalMaterialConstants, offset);
    material.coat = Material_coat(gGlobalMaterialConstants, offset);
    material.coatColor = Material_coatColor(gGlobalMaterialConstants, offset);
    material.coatRoughness = Material_coatRoughness(gGlobalMaterialConstants, offset);
    material.coatAnisotropy = Material_coatAnisotropy(gGlobalMaterialConstants, offset);
    material.coatRotation = Material_coatRotation(gGlobalMaterialConstants, offset);
    material.coatIOR = Material_coatIOR(gGlobalMaterialConstants, offset);
    material.coatAffectColor = Material_coatAffectColor(gGlobalMaterialConstants, offset);
    material.coatAffectRoughness = Material_coatAffectRoughness(gGlobalMaterialConstants, offset);
    material.emission = Material_emission(gGlobalMaterialConstants, offset);
    material.emissionColor = Material_emissionColor(gGlobalMaterialConstants, offset);
    material.opacity = Material_opacity(gGlobalMaterialConstants, offset);
    material.thinWalled = Material_thinWalled(gGlobalMaterialConstants, offset);

    // Sample base color from a texture if necessary.
    float4 texCoord = float4(shading.texCoord, 0.0f, 1.0f);
    if (Material_hasBaseColorTex(gGlobalMaterialConstants, offset))
    {
        float2 uv =
        applyUVTransform(shading.texCoord, Material_baseColorTexPivot(gGlobalMaterialConstants, offset),
                         Material_baseColorTexScale(gGlobalMaterialConstants, offset),
                         Material_baseColorTexRotation(gGlobalMaterialConstants, offset),
                         Material_baseColorTexOffset(gGlobalMaterialConstants, offset));
        material.baseColor = sampleBaseColorTexture(headerOffset, uv, 0.0f).rgb;
    }

    // Sample specular roughness from a texture if necessary.
    if (Material_hasSpecularRoughnessTex(gGlobalMaterialConstants, offset))
    {
        float2 uv = applyUVTransform(shading.texCoord,
                                     Material_specularRoughnessTexPivot(gGlobalMaterialConstants, offset),
                                     Material_specularRoughnessTexScale(gGlobalMaterialConstants, offset),
                                     Material_specularRoughnessTexRotation(gGlobalMaterialConstants, offset),
                                     Material_specularRoughnessTexOffset(gGlobalMaterialConstants, offset));

        material.specularRoughness = sampleSpecularRoughnessTexture(headerOffset, uv, 0.0f).r;
    }

    // Sample emission color from a texture if necessary.
    if (Material_hasEmissionColorTex(gGlobalMaterialConstants, offset))
    {
        float2 uv = applyUVTransform(shading.texCoord,
                                     Material_emissionColorTexPivot(gGlobalMaterialConstants, offset),
                                     Material_emissionColorTexScale(gGlobalMaterialConstants, offset),
                                     Material_emissionColorTexRotation(gGlobalMaterialConstants, offset),
                                     Material_emissionColorTexOffset(gGlobalMaterialConstants, offset));
        material.emissionColor = sampleEmissionColorTexture(headerOffset, uv, 0.0f).rgb;
    }

    // Sample opacity from a texture if necessary.
    if (Material_hasOpacityTex(gGlobalMaterialConstants, offset))
    {
        float2 uv = applyUVTransform(shading.texCoord, Material_opacityTexPivot(gGlobalMaterialConstants, offset),
                                     Material_opacityTexScale(gGlobalMaterialConstants, offset),
                                     Material_opacityTexRotation(gGlobalMaterialConstants, offset),
                                     Material_opacityTexOffset(gGlobalMaterialConstants, offset));
        material.opacity = sampleOpacityTexture(headerOffset, uv, 0.0f).rgb;
    }

    // Sample a normal from the normal texture, convert it to an object-space normal, transform to
    // world space, and store it in the output value.
    isGeneratedNormal = false;
    if (Material_hasNormalTex(gGlobalMaterialConstants, offset))
    {
        float2 uv = applyUVTransform(shading.texCoord, Material_normalTexPivot(gGlobalMaterialConstants, offset),
                                     Material_normalTexScale(gGlobalMaterialConstants, offset),
                                     Material_normalTexRotation(gGlobalMaterialConstants, offset),
                                     Material_normalTexOffset(gGlobalMaterialConstants, offset));
        float3 normalTexel = sampleNormalTexture(headerOffset, uv, 0.0f).rgb;
        float3 objectSpaceNormal = calculateNormalFromMap(
                                                          normalTexel, TANGENT_SPACE, 1.0, shading.normal, shading.tangent);
        materialNormal = normalize((shading.objToWorld * shading.objectPosition).xyz);

        isGeneratedNormal = true;
    }

    // Copy the base color to the (internal) metal color.
    material.metalColor = material.baseColor;

    return material;
}

#endif // __DEFAULTMATERIAL_H__

