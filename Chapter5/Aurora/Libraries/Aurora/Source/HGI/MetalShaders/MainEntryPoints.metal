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

#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;
using namespace raytracing;

// Defines to enable Debug modes
//#define DEBUG_NORMALS (1)

// Slang to Metal
#define lerp mix
#define Texture2D texture2d
#define SamplerState sampler
#define RaytracingAccelerationStructure instance_acceleration_structure
#define RayDesc ray

float3 hsv2rgb(float3 c) {
    float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    float3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
struct RayPayload {
    int rayId;
};
using HandlerFuncSig = int(ray, thread RayPayload&);


// Per instance data.
struct InstanceRecord
{
    // Geometry data.
    uint64_t indexBufferDeviceAddress;
    uint64_t vertexBufferDeviceAddress;
    uint64_t normalBufferDeviceAddress;
    uint64_t tangentBufferDeviceAddress;
    uint64_t texCoordBufferDeviceAddress;

    // ADDRESS into material array.
    uint64_t material; // TODO: add wrapper, in Vulkan it's integer, in Metal it's HGIBuffer

    // Index into texture sampler array for material's textures.
    int baseColorTextureIndex = -1;

    int specularRoughnessTextureIndex = -1;
    int normalTextureIndex            = -1;
    int opacityTextureIndex           = -1;
    int emissionTextureIndex          = -1;
//    // Geometry flags.
    unsigned int hasNormals   = true;
    unsigned int hasTangents  = false;
    unsigned int hasTexCoords = true;
};

#define DISABLE_HGI_SHADER_STAGE_MISS        1
#define DISABLE_HGI_SHADER_STAGE_CLOSEST_HIT 1
#define DISABLE_HGI_SHADER_STAGE_ANY_HIT     1
//#define DISABLE_HGI_SHADER_STAGE_RAY_GEN   1

class MetalContext {
public:

#include "Options.metal"
#include "PathTracingCommon.metal"
#include "ShadeFunctions.metal"
#include "DefaultMaterial.metal"
#include "GlobalBufferAccessors.metal"
#include "InstancePipelineState.metal"
#include "EvaluateMaterial.metal"

    MetalContext(RaytracingAccelerationStructure accelerationStructure, const constant EnvironmentConstants& envConstants, const constant SampleData& sampleData, const constant FrameData& frameData)
    : gEnvironmentConstants(envConstants), gScene(accelerationStructure), gNullScene(accelerationStructure), gSampleData(sampleData), gFrameData(frameData) {};
}; // MetalContext !


// Math constants.
#define M_PI 3.1415926535897932384626433832795f
#define M_PI_INV 1.0 / 3.1415926535897932384626433832795
#define M_FLOAT_EPS 0.000001f
#define M_RAY_TMIN 0.01f

// Results of the path tracing sample.
struct Output
{
    // Depth in Normalized Device Coordinates, for entire path.
    float depthNDC;
    // Normal of first collision in path.
    float3 normal;
    // Material metalness of first collision in path.
    float metalness;
    // Material roughness of first collision in path.
    float roughness;
    // Material base color of first collision in path.
    float3 baseColor;
    // Depth in Normalized Device Coordinates, for entire path.
    float depthView;
    // Direct lighting contribution for path.
    float3 direct;
    // Indirect lighting contribution for path (split into diffuse and glossy.)
    MetalContext::IndirectOutput indirect;
    // Alpha for path (0.0 only if all ray segments were transparent)
    float alpha;

    void clear()
    {
        indirect.clear();
        depthNDC = 1.0f;
        normal = -1.0f;
        metalness = 0.1f;
        roughness = 0.1f;
        baseColor = BLACK;
        depthView = INFINITY;
        direct = 0.0f;
        alpha = 0.0;
    }
};

struct Uniforms {
    unsigned int width, height, frameIndex;
};
struct Textures {
    texture2d<float, access::sample> tex_0;
    texture2d<float, access::read_write> directTex;
    texture2d<float, access::sample> tex_2;
    array<texture2d<float, access::sample>, TEXTURE_ARRAY_SIZE> texturesArray;
    texture2d<float, access::read_write> tex_4;
    texture2d<float, access::read_write> tex_5;
    texture2d<float, access::read_write> defaultSampler;
    texture2d<float, access::sample> backgroundTex;
    texture2d<float, access::sample> lightTex;
    texture2d<float, access::read_write> accumulationTex; // 9
    texture2d<float, access::read_write> depthTexture;
    texture2d<float, access::read_write> motionTexture;
    texture2d<float, access::read_write> diffuseAlbedoTexture;
    texture2d<float, access::read_write> specularAlbedoTexture;
    texture2d<float, access::read_write> normalTexture;
    texture2d<float, access::read_write> roughnessTexture;
};

struct Samplers {
    sampler smp_0;
    sampler smp_1;
    sampler smp_2;
    array<sampler, TEXTURE_ARRAY_SIZE> smpArray;
};

struct Buffers {
    device void* buf_0;
    device void* buf_1;
    constant MetalContext::FrameData* frameData;
    device void* buf_3;
    constant MetalContext::SampleData* sampleData;
    constant MetalContext::EnvironmentConstants* environmentData;
    device void* buf_6;
    device void* buf_7;
    device void* buf_8;
    device void* buf_9;
    constant InstanceRecord* instanceData;
    constant MetalContext::AliasEntry* aliasMapData;
};

using HandlerFuncSig = int(ray, thread RayPayload&);

// The ray generation shader, iteratively shades the entire ray path.
kernel void RayGenShader(
     uint2                                                  tid                       [[thread_position_in_grid]],
     instance_acceleration_structure                        accelerationStructure     [[buffer(0)]],
     constant Uniforms&                                     uniformBuf                [[buffer(27)]],
     constant Samplers&                                     samplerBuf                [[buffer(28)]],
     constant Textures&                                     textureBuf                [[buffer(29)]],
     constant Buffers&                                      bufferBuf                 [[buffer(30)]],
     intersection_function_table<triangle_data, instancing> intersectionFunctionTable [[buffer(5)]],
     visible_function_table<HandlerFuncSig>                 hitTable                  [[buffer(6)]],
     visible_function_table<HandlerFuncSig>                 missTable                 [[buffer(7)]]
     )
{
    // Get the max trace depth from frame settings.
    int maxTraceDepth = bufferBuf.frameData->traceDepth;
    constant MetalContext::FrameData& gFrameData = *bufferBuf.frameData;
    texture2d<float, access::read_write> dstTex = textureBuf.directTex;
    texture2d<float, access::read_write> depthTexture = textureBuf.depthTexture;
    texture2d<float, access::read_write> motionTexture = textureBuf.motionTexture;
    texture2d<float, access::read_write> diffuseAlbedoTexture = textureBuf.diffuseAlbedoTexture;
    texture2d<float, access::read_write> specularAlbedoTexture = textureBuf.specularAlbedoTexture;
    texture2d<float, access::read_write> normalTexture = textureBuf.normalTexture;
    texture2d<float, access::read_write> roughnessTexture = textureBuf.roughnessTexture;

    // Is opaque shadow hits only enabled in frame settings.
    bool onlyOpaqueShadowHits = bufferBuf.frameData->isForceOpaqueShadowsEnabled;

    // Create Metal Context class object.
    constexpr sampler s(coord::normalized, address::repeat, filter::linear, mip_filter::none);
    constexpr sampler envs(coord::normalized, filter::linear, mip_filter::none);
    thread MetalContext context(accelerationStructure, *bufferBuf.environmentData, *bufferBuf.sampleData, *bufferBuf.frameData);
    context.gInstanceBuffer = bufferBuf.instanceData;
    context.gEnvironmentAliasMap = bufferBuf.aliasMapData;
    context.gRaysIndex = tid;
    context.gRaysDimensions = uint2(dstTex.get_width(), dstTex.get_height());
    context.gInstanceBuffer = bufferBuf.instanceData;
    context.gEnvironmentLightTexture = textureBuf.lightTex;
    context.gEnvironmentBackgroundTexture = textureBuf.backgroundTex;
    context.gGlobalMaterialTextures = &textureBuf.texturesArray;
    context.gSamplerArray = &samplerBuf.smpArray;
    context.gDefaultSampler = s;
    context.gDefaultEnvSampler = envs;
    
    // Get the dispatch dimensions (screen size), dispatch index (screen coordinates), and sample
    // index.
    uint2 screenSize = uint2(dstTex.get_width(), dstTex.get_height());
    uint2 screenCoords = tid;
    uint sampleIndex = bufferBuf.sampleData->sampleIndex;

    // Initialize a random number generator, so that each sample and pixel gets a unique seed.
    MetalContext::Random rng = context.initRand(sampleIndex, screenSize, screenCoords);

    // Compute a camera ray (origin and direction) for the current screen coordinates. This applies
    // a random offset to support antialiasing and depth of field.
    float3 rayOrigin;
    float3 rayDirection;
    int rayType = CAMERA_RAY;
    context.computeCameraRay((float2)screenCoords, (float2)screenSize, gFrameData.cameraInvView, gFrameData.viewSize,
        gFrameData.isOrthoProjection, gFrameData.focalDistance, gFrameData.lensRadius, rng, rayOrigin,
        rayDirection);

    // The results of the tracing the entire ray path through the scene.
    Output output;
    output.clear();

    // The ray t (distance along path.)
    float t = 0;

    // Prepare the environment data structure.
    MetalContext::Environment environment = context.prepareEnvironmentValues();

    // The ray contribution is how much the current ray segment contributes to the overall path.
    // It starts at 1.0 and then is scaled downwards at each subsequent segment.
    float3 rayContribution = 1.0f;

    // Is the current ray segment a primary ray?
    bool isPrimaryRay = true;

    // Has the path been terminated?
    bool pathEnded = false;
    MetalContext::InstanceRayPayload rayPayload;

    // Iterate through every segment in the ray path, until maximum depth is reached or path is ended.
    for (int rayDepth = 0; rayDepth <= maxTraceDepth && !pathEnded; rayDepth++)
    {
        // The minimum T for this ray segment is zero for the initial camera ray otherwise use
        // a small epsilon to avoid self-intersection.
        float minT = rayDepth == 0 ? 0.0f : M_RAY_TMIN;

        // If this ray collides with nothing, should the background or lighting environment be shaded?
        bool shadeBackgroundOnMiss = isPrimaryRay || rayType == TRANSMISSION_LOBE || rayType == GROUND_PLANE_REFLECTION_RAY;

        // Direct light contribution for this ray segment.
        float3 direct = 0;

        // Indirect light contribution for this ray segment.
        MetalContext::IndirectOutput indirect;
        indirect.clear();

        // How much the contribution will be scaled for the subsequent ray segments (based on sampling PDF of this surface's material.)
        float3 nextRayContribution = 1.0;

        // The ray type of the next ray (based on sampling lobe produced from sampling this surface's material.)
        int nextRayType;

        // Have we exceeded ray depth for this path?
        if (rayDepth == maxTraceDepth)
        {
            // If we have reached the end of this path, evaluate the environment and terminate the path.
            direct = context.evaluateEnvironment(environment, rayDirection, shadeBackgroundOnMiss);
            pathEnded = true;
        }
        else
        {
            bool absorbedByGroundPlane = false;

            // Trace an instance ray that will return the instance shading result from the closest hit, if any.
            context.traceInstanceRay(context.gScene, rayOrigin, rayDirection, minT, rayPayload);

//            // Add the contribution of the ground plane, if enabled.
//            float3 groundPlanePosition = 0;
//            float groundPlaneDistance = INFINITY;
//            if (isPrimaryRay && context.intersectGroundPlane(gGroundPlane, rayOrigin, rayDirection, groundPlanePosition, groundPlaneDistance) && groundPlaneDistance < rayPayload.distance)
//            {
//                float3 groundPlaneRayDirection;
//                float3 groundPlaneRayOrigin;
//                float3 groundPlaneRayColor;
//
//                // Shade the ground plane.
//                int groundPlaneRayType = setupGroundPlaneRay(groundPlanePosition, gGroundPlane, gScene, environment, rayOrigin, rayDirection,
//                onlyOpaqueShadowHits, maxTraceDepth, rng,
//                groundPlaneRayDirection, groundPlaneRayOrigin, groundPlaneRayColor);
//
//                if (groundPlaneRayType == GROUND_PLANE_REFLECTION_RAY)
//                {
//                    // The ground plane reflection path was selected, continue the path as a reflection ray.
//                    nextRayType = GROUND_PLANE_REFLECTION_RAY;
//
//                    // Setup the next ray direction, origin and contribution from ground plane.
//                    nextRayContribution = groundPlaneRayColor;
//                    rayDirection = groundPlaneRayDirection;
//                    rayOrigin = groundPlaneRayOrigin;
//
//                    // The ray was absorbed.
//                    absorbedByGroundPlane = true;
//
//                    // This ray is no longer a primary ray.
//                    isPrimaryRay = false;
//                }
//                else
//                {
//                    // If the ray was not absorbed scale the contribution based on ground plane color.
//                    rayContribution *= groundPlaneRayColor;
//                }
//            }

            if (!absorbedByGroundPlane)
            {
                // Did the ray segment hit any geometry?
                if (rayPayload.hit())
                {

                    // Increment t by the distance of the ray hit.
                    t += rayPayload.distance;

                    // View direction is inverse of ray direction.
                    float3 V = -rayDirection;

                    // Shading data and normal are taken from ray hit results.
                    MetalContext::ShadingData shading = context.shadingDataFromInstanceResult(rayPayload);



                    // Were any layers hit?  If this remains false, ray was not absorbed by any of the layer, including base layer..
                    bool hitLayer = false;

                    // Get the UV for the base layer (as the UVs in the shading struct will be over-written in layer loop.)
                    float2 baseLayerUV = shading.texCoord;

// Only include code for layer loop if ENABLE_LAYERS is set to 1 (otherwise just set layerIndex to zero.)
#if ENABLE_LAYERS
                    // Get layer count from the instance header.
                    int layerCount = context.instanceLayerCount(rayPayload.instanceBufferOffset);

                    // Iterate through all the layers in reverse order, ending with base layer.
                    for (int li = layerCount; li >= 0; li--)
                    {
#else
                    int li = 0;
                    {
#endif

                        // Compute material buffer offset, either for layer material or base layer.
                        int materialBufferOffset;
                        if (li == 0)
                        {
                            materialBufferOffset = context.instanceMaterialBufferOffset(rayPayload.instanceBufferOffset);
                            shading.texCoord = baseLayerUV;
                        }
                        else
                        {
                            int layerIndex = li - 1;
                            materialBufferOffset = context.instanceLayerMaterialBufferOffset(rayPayload.instanceBufferOffset, layerIndex);
                            shading.texCoord = context.computeLayerUV(rayPayload.instanceBufferOffset, layerIndex, shading);
                        }

                        // Retrieve the shader index for the layer material.
                        // This is stored in the material header in the global material byte buffer,
                        // at a position indicated by instance's gMaterialBufferOffset.
                        int shaderIndex = context.materialShaderIndex(materialBufferOffset);

                        // Was a normal generated by the material?
                        bool isGeneratedNormal = false;

                        // Material normal (overriden by material setup, if material has normal map.)
                        float3 materialNormal = shading.normal;

                        // Initialize the material using the shader index with the externally defined evaluate function.
                        // This function will be generated at runtime and linked with the rest of the code.
                        MetalContext::Material material = context.evaluateMaterialFromConstants(bufferBuf.instanceData[rayPayload.instanceIndex],
                                                                                                shaderIndex, shading,
                                                                                                  materialBufferOffset, materialNormal, isGeneratedNormal);
                        // Is this a primary ray?
                        if (isPrimaryRay)
                        {
                            // Set the output normal from the first hit.
                            output.normal = materialNormal;
#if defined(DEBUG_NORMALS)
                            dstTex.write(float4((output.normal + 1.0f) * 0.5f, 1.0f), screenCoords);
                            return;
#endif
                            // Clamp the output roughness for the ray path to a minimum, because the denoiser
                            // handles materials with low (near-zero) roughness poorly, leading to a noisy result. This addresses
                            // two separate but related issues: low-roughness metallic materials reflecting
                            // noisy surroundings, and low- roughness dielectric materials where the glossy
                            // lobe is sparsely sampled. NOTE: The roughness in the ray payload is used for
                            // denoising only. The material specular roughness (used in shading) is not
                            // modified here.
                            const float kMinRoughness = 0.05f;
                            float clampedRoughness = max(material.specularRoughness, kMinRoughness);

                            // Set the output materal properties for the first hit.
                            output.metalness = material.metalness;
                            output.roughness = clampedRoughness;
                            output.baseColor = material.baseColor;

                            // Set the output view-space depth from ray t.
                            output.depthView = t;

                            // Set the output Normalized Device Coordinate (NDC) depth from hit position.
                            float4 positionClip = transpose(context.gFrameData.cameraViewProj) * float4(shading.geomPosition, 1.0f);
                            output.depthNDC = (positionClip.z / positionClip.w + 1.0f) / 2.0f;
                            
                            /*
                             MetalFX outputs:
                             * colorTexture is now a jittered, low-res and noisy image. The game can directly render its RT effects in this buffer.
                             -> dstTex
                             * depthTexture
                             -> output.depthNDC
                             * motionTexture
                             -> zeroTexture
                             * diffuseAlbedoTexture is a new input texture that contains the base color of the scene. The buffer is expected to be taken from either the first bounce or the G-Buffer and is noise-free.
                             -> output.baseColor
                             * specularAlbedoTexture is a new input texture that contains the pre-integrated specular radiance of the specular BRDF. The buffer is expected to be calculated using an approximation of the specular BRDF assuming L == N, the buffer is expected to be noise free.
                             -> spec_albedo = specAlbedo(baseColorToSpecularF0(output.baseColor, output.metalness), output.roughness, NdotV);
                             * normalTexture is a new input texture that contains the world space normals of the scene. The buffer is expected to be taken from either the first bounce or the G-Buffer and is noise-free.
                             -> output.normal
                             * roughnessTexture is a new input texture that contains the roughness of the material. The buffer is expected to be taken from either the first bounce or the G-Buffer and is noise-free.
                             -> output.roughness
                             * specularHitDistanceTexture is a new input texture that contains the distance of the first hit of the specular rays from the first bounce. This buffer is expected to be noisy.
                             -> output.depthView ?
                             * worldToViewMatrix the scene's worldToView matrix (TODO: transpose?)
                             * viewToClipMatrix the scene's viewToClip matrix (TODO: transpose?)
                             * jitter ?
                             * exposure ?
                             */
#define DEBUG_MTLFX
#ifdef DEBUG_MTLFX
#define MIN_DIELECTRICS_F0 0.04f
                            float3 specularF0 = lerp(float3(MIN_DIELECTRICS_F0, MIN_DIELECTRICS_F0, MIN_DIELECTRICS_F0), output.baseColor, output.metalness);
                            float alpha = output.roughness;
                            float NdotV = max(0.0f, dot(V, shading.normal));
                            const float2x2 A = float2x2(
                                0.99044f, -1.28514f,
                                1.29678f, -0.755907f
                            );

                            const float3x3 B = float3x3(
                                1.0f, 2.92338f, 59.4188f,
                                20.3225f, -27.0302f, 222.592f,
                                121.563f, 626.13f, 316.627f
                            );

                            const float2x2 C = float2x2(
                                0.0365463f, 3.32707f,
                                9.0632f, -9.04756f
                            );

                            const float3x3 D = float3x3(
                                1.0f, 3.59685f, -1.36772f,
                                9.04401f, -16.3174f, 9.22949f,
                                5.56589f, 19.7886f, -20.2123f
                            );

                            const float alpha2 = alpha * alpha;
                            const float alpha3 = alpha * alpha2;
                            const float NdotV2 = NdotV * NdotV;
                            const float NdotV3 = NdotV * NdotV2;

                            const float E = dot((A * float2(1.0f, NdotV)), float2(1.0f, alpha));
                            const float F = dot((B * float3(1.0f, NdotV, NdotV3)), float3(1.0f, alpha, alpha3));

                            const float G = dot((C * float2(1.0f, NdotV)), float2(1.0f, alpha));
                            const float H = dot((D * float3(1.0f, NdotV2, NdotV3)), float3(1.0f, alpha, alpha3));

                            const float biasModifier = saturate(dot(specularF0, float3(0.333333f, 0.333333f, 0.333333f)) * 50.0f);

                            const float bias = max(0.0f, (E / F)) * biasModifier;
                            const float scale = max(0.0f, (G / H));

                            float3 specAlbedo = float3(bias, bias, bias) + float3(scale, scale, scale) * specularF0;
                            
                            depthTexture.write(float4(output.depthNDC, 0.0f, 0.0f, 1.0f), screenCoords);
                            motionTexture.write(float4(0.0f, 0.0f, 0.0f, 1.0f), screenCoords);
                            diffuseAlbedoTexture.write(float4(output.baseColor, 1.0f), screenCoords);
                            specularAlbedoTexture.write(float4(specAlbedo, 1.0f), screenCoords);
                            normalTexture.write(float4(output.normal, 1.0f), screenCoords);
                            roughnessTexture.write(float4(output.roughness, 0.0f, 0.0f, 1.0f), screenCoords);
#endif
                        }

                        // Compute transparency from inverse of opacity.
                        float3 transparency = 1.0f - material.opacity;

                        // Randomly choose if next ray segment hit any opaque geometry for this layer based on the transparency of the surface.
                        float P = context.luminance(transparency);
                        hitLayer = context.random2D(rng).x >= P;
                        float3 directionalShadowRayDirection = 0;
                        float3 directionalLightColor = 0;
                        bool hasDirectionalLight = false;
                        if (hitLayer)
                        {
                            // This ray struck opaque geometry, shade the collision as the next ray segment in the path.

                            // If a new normal has been generated, transform it to world space and build a
                            // corresponding basis from it.
                            if (isGeneratedNormal)
                            {
                                shading.normal = materialNormal;
                                context.buildBasis(shading.normal, shading.tangent, shading.bitangent);
                            }

                            // Add the emissive light for this collision to the direct light for this ray segement.
                            direct += context.shadeEmission(material);

                            // Shade with a randomly selected global distant light. Skip this if there are
                            // no distant lights.
                            if (context.gFrameData.lights.distantLightCount > 0)
                            {
                                // Choose a random distant light for this sample, by computing an index in
                                // range the 0 to (distantLightCount-1).
                                int lightIdx =
                                    int(context.random2D(rng).x * float(context.gFrameData.lights.distantLightCount));

                                // Skip if the light has zero intensity.
                                if (context.gFrameData.lights.distantLights[lightIdx].colorAndIntensity.a > 0.0f)
                                {
                                    // Shade the light, computing a color and a shadow ray direction to be emitted later.
                                    directionalLightColor = context.shadeDirectionalLight(
                                        context.gFrameData.lights.distantLights[lightIdx].direction,
                                        context.gFrameData.lights.distantLights[lightIdx].colorAndIntensity,
                                        context.gFrameData.lights.distantLights[lightIdx].cosRadius,
                                        material, shading, V, rng, directionalShadowRayDirection);
                                    hasDirectionalLight = true;
                                }
                            }

                            // The multiple importance sampling (MIS) environment shade function will return these values.
                            // They will be used to emit shadow rays for MIS material and light at the end of the loop.
                            int misLobeID;
                            bool misEmitsMaterialShadowRay;
                            float3 misMaterialResult = 0;
                            float3 misMaterialShadowRayDirection = 0;
                            bool misEmitsLightShadowRay;
                            float3 misLightResult = 0;
                            float3 misLightShadowRayDirection = 0;

                            // Shade with the environment light with multiple importance sampling.
                            context.shadeEnvironmentLightMIS(environment, material, shading, V, rng, misLobeID,
                                misEmitsMaterialShadowRay, misMaterialResult, misMaterialShadowRayDirection,
                                misEmitsLightShadowRay, misLightResult, misLightShadowRayDirection
                            );

                            // Scale the ray contribution (for this ray segment and subsequent ones) by the inverse of
                            // the transparency, and normalized by the probability of this segment being considered opaque.
                            rayContribution *= material.opacity / (1.0f - P);

                            // If the output alpha is zero and this was a transmission lobe ray, pass along an
                            // alpha of zero. If all the path segments beyond this hit have transparent or transmissive
                            // the final result will be zero, otherwise the alpha is will be one (opaque).
                            output.alpha = (output.alpha == 0.0f && rayType == TRANSMISSION_LOBE) ? 0.0f : 1.0f;

                            // Sample the PDF for the current surface material, and compute the next ray direction and contribution.
                            nextRayType = context.setupIndirectLightRay(material, shading, V, rayDepth, maxTraceDepth, rng,
                                rayDirection, rayOrigin, nextRayContribution);

                            // Emit all the shadows rays (for the directional light and MIS environment).
                            // Emitting them here at the end of loop avoids storing all the material state in registers, and is a huge performance optimisation.
                            // All the shadow rays use the same origin, calculated from geometric position or intersection position (depending on whether back-face hit.)
                            float3 shadowRayOrigin = dot(V, shading.normal) < 0.0f ? shading.geomPosition : shading.position;

                            // Trace shadow ray for directional light, if we have one.
                            float3 lightVisibility;
                            if (hasDirectionalLight)
                            {
                                // Add directional result to the direct light.
                                lightVisibility = context.traceShadowRay(accelerationStructure, shadowRayOrigin, directionalShadowRayDirection, M_RAY_TMIN, onlyOpaqueShadowHits, rayDepth, maxTraceDepth);
                                direct += lightVisibility * directionalLightColor;
                            }

                            // Trace shadow rays for material and light component of MIS environment (if we have them.)
                            if (misEmitsLightShadowRay || misEmitsMaterialShadowRay)
                            {
                                float3 misResult = 0;
                                if (misEmitsMaterialShadowRay)
                                {
                                    lightVisibility = context.traceShadowRay(accelerationStructure, shadowRayOrigin, misMaterialShadowRayDirection, M_RAY_TMIN, onlyOpaqueShadowHits, rayDepth, maxTraceDepth);
                                    misResult += lightVisibility * misMaterialResult;
                                }
                                if (misEmitsLightShadowRay)
                                {
                                    lightVisibility = context.traceShadowRay(accelerationStructure, shadowRayOrigin, misLightShadowRayDirection, M_RAY_TMIN, onlyOpaqueShadowHits, rayDepth, maxTraceDepth);
                                    misResult += lightVisibility * misLightResult;
                                }

                                // If the material sampling for MIS environment returned a diffuse lobe, add the MIS result to the diffuse indirect result, otherwise add to glossy result.
                                // Set the hit distance to infinity (as the environment assumed to be at infinite distance.)
                                if (misLobeID == DIFFUSE_LOBE)
                                {
                                    indirect.diffuse += misResult;
                                    indirect.diffuseHitDist = INFINITY;
                                }
                                else
                                {
                                    indirect.glossy += misResult;
                                    indirect.glossyHitDist = INFINITY;
                                }

                            }

                            // If next ray not needed (as contribution is too small) terminate the path.
                            if (nextRayType == NO_SECONDARY_RAY)
                                pathEnded = true;

                            // This is no longer a primary ray.
                            isPrimaryRay = false;

                            // Break out of loop when ray is absorbed by a layer (if we are in a loop.)
#if ENABLE_LAYERS
                            break;
#endif
                        }
                        // Scale the ray contribution (for this ray segment and subsequent ones) by transparency, and normalized
                        // by the probability of this layer being considered transparent.
                        else
                            rayContribution *= transparency / P;
                    }

                    // Handle case were no layers were hit, including base layer.
                    if (!hitLayer)
                    {
                        // No opaque geometry was hit in any layer, continue the ray path as if no collision occurred.
                        rayDirection = -V;
                        rayOrigin = shading.geomPosition;

                        // All other ray variables remain the same.

                        // Continue to the next path segment without applying contribution (as if this
                        // hit didn't happen.)
                        continue;
                    }
                }
                else
                {
                    // No geometry was hit, shade the ray using the background color.
                    // If needed otherwise do nothing, as environment lighting was calculated during MIS environment shading.
                    if (shadeBackgroundOnMiss)
                        direct = context.evaluateEnvironment(environment, rayDirection, true);

                    // Terminate the path.
                    pathEnded = true;
                }
            }
        }

        // Scale all the lighting components for the current ray segment, indirect and direct, by the contribution.
        direct *= rayContribution;
        indirect.diffuse *= rayContribution;
        indirect.glossy *= rayContribution;

        // How the result is accumulated depends on whether this ray is a primary ray.
        if (isPrimaryRay)
        {
            // If it is a primary ray, accumulate direct and indirect components in to equivalent component in the output data.
            output.direct = direct;
            output.indirect = indirect;
        }
        else
        {
            // If this secondary ray compute the total radiance from all the lighting components.
            float3 radiance = direct + indirect.diffuse + indirect.glossy;

            // Accumulate the total radiance into diffuse or glossy component of output indirect
            // component, based on lobe this ray segment originated from.
            if (rayType == DIFFUSE_LOBE)
            {
                output.indirect.diffuse += radiance;
                output.indirect.diffuseHitDist = t;
            }
            else
            {
                output.indirect.glossy += radiance;
                output.indirect.glossyHitDist = t;
            }
        }

        // Scale the contribution to be used in subsequent segments.
        rayContribution *= nextRayContribution;

        // Set the ray type for the next segment.
        rayType = nextRayType;
    }

    thread float3 finalRadiance = output.direct + output.indirect.diffuse + output.indirect.glossy;
    // Adjust the radiance of the sample, e.g. to perform corrections.
    context.adjustRadiance(gFrameData.maxLuminance, gFrameData.isDisplayErrorsEnabled, finalRadiance);

    // Store the result in the "result" output texture. If denoising is enabled, only the "direct"
    // (non-denoised) shading is included in the result texture, as the rest of shading is stored in
    // the denoising AOVs below.
    float4 result = float4(finalRadiance, output.alpha);
    result.rgb = gFrameData.isDenoisingEnabled ? output.direct : result.rgb;
    dstTex.write(result, screenCoords);
}
