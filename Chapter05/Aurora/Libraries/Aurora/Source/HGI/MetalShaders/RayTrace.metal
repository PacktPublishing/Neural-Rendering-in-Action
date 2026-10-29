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

#ifndef __RAY_TRACE_H__
#define __RAY_TRACE_H__

#include "Environment.metal"
#include "Globals.metal"
#include "Random.metal"
#include "Sampling.metal"

// A structure for the results of indirect lighting and environment light sampling, which is used
// for denoising.
struct IndirectOutput
{
    float3 diffuse;
    float diffuseHitDist;
    float3 glossy;
    float glossyHitDist;

    // Clears the indirect output to values expected for a miss.
    void clear()
    {
        diffuse        = BLACK;
        diffuseHitDist = INFINITY;
        glossy         = BLACK;
        glossyHitDist  = INFINITY;
    }
};

// Instance ray payload.  Used by all closest hit shaders.
struct InstanceRayPayload {
    float3 geomPosition; // the geometric position, interpolated from adjancent vertices
    float2 texCoord; // the shading texture coordinates, interpolated from adjancent vertices
    float3 objectNormal; // the shading position in object space (not world space). Used only in
    // materialX generated code.
    float3 objectPosition; // the shading normal in object space (not world space). Used only in
    // materialX generated code.
    // Offset withing instance data buffer for instance.
    int instanceBufferOffset;
    // Triangle indices for hit.
    uint3 indices;
    // Barycentric coordinates for hit.
    float2 bary2;
    // Distance from ray origin collision occurred.
    float distance = INFINITY;
    // BLAS index of instance that was hit (or -1 if not hit)
    int instanceIndex;
    float3x4 objToWorld;

    // Clears the indirect output to values expected for a miss.
    void clear()
    {
        instanceIndex = -1;
        distance = INFINITY;
        geomPosition = 0.0;
        texCoord = 0.0;
        objectNormal = 0.0;
        objectPosition = 0.0;
    }
    bool hit() {
        return instanceIndex>=0;
    }

};
float3x4 transformMatrixForBLAS(int blasIndex)
{
    return float3x4(1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0);
}
ShadingData shadingDataFromInstanceResult(InstanceRayPayload result)
{
    // Fill in the shading data from hit result.
    ShadingData shading;
    shading.geomPosition = result.geomPosition;
    shading.texCoord = result.texCoord;
    shading.objectNormal = result.objectNormal;
    shading.objectPosition = result.objectPosition;
    shading.barycentrics = computeBarycentrics(result.bary2);
    shading.indices = result.indices;

    // Get transform matrix from instance buffer.
    shading.objToWorld = result.objToWorld;

    // Transform object space attributes.
    shading.position = (shading.objToWorld * float4(shading.objectPosition, 1.0f).xyz).xyz;
    shading.normal = normalize((float3x3(shading.objToWorld[0].xyz, shading.objToWorld[1].xyz, shading.objToWorld[2].xyz) * shading.objectNormal));

    // As we don't currently support user-supplied tangents, we always use arbitrary tangent computed from normal.
    float3 up =
    abs(shading.normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    shading.tangent = normalize(cross(shading.normal, up));

    // Generate bitangent from normal and tangent.
    shading.bitangent = computeBitangent(shading.normal, shading.tangent);


    return shading;
}

// Computes the origin and direction of a ray at the specified screen coordinates, using the
// specified view parameters (orientation, position, and FOV).
void computeCameraRay(float2 screenCoords, float2 screenSize, float4x4 invView, float2 viewSize,
                      bool isOrtho, float focalDistance, float lensRadius, thread Random& rng, thread float3& origin,
                      thread float3& direction)
{
    // Apply a random offset to the screen coordinates, for antialiasing. Convert the screen
    // coordinates to normalized device coordinates (NDC), i.e. the range [-1, 1] in X and Y. Also
    // flip the Y component, so that +Y is up.
    screenCoords += random2D(rng);
    float2 ndc = (screenCoords / screenSize) * 2.0f - 1.0f;
    ndc.y      = -ndc.y;
    
    // Get the world-space orientation vectors from the inverse view matrix.
    invView = transpose(invView);
    float3 right = invView[0].xyz;  // right: row 0
    float3 up    = invView[1].xyz;  // up: row 1
    float3 front = -invView[2].xyz; // front: row 2, negated for RH coordinates

    // Build a world-space offset on the view plane, based on the view size and the right and up
    // vectors.
    float2 size            = viewSize * 0.5f;
    float3 offsetViewPlane = size.x * ndc.x * right + size.y * ndc.y * up;

    // Compute the ray origin and direction:
    // - Direction: For orthographic projection, this is just the front direction (i.e. all rays are
    //   parallel). For perspective, it is the normalized combination of the front direction and the
    //   view plane offset.
    // - Origin: For orthographic projection, this is the eye position (row 3 of the view matrix),
    //   translated by the view plane offset. For perspective, it is just the eye position.
    //
    // NOTE: It is common to "unproject" a NDC point using the view-projection matrix, and subtract
    // that from the eye position to get a direction. However, this is numerically unstable when the
    // eye position has very large coordinates and the projection matrix has small (nearby) clipping
    // distances. Clipping is not relevant for ray tracing anyway.
    if (isOrtho)
    {
        direction = front;
        origin    = invView[3].xyz + offsetViewPlane;
    }
    else
    {
        direction = normalize(front + offsetViewPlane);
        origin    = invView[3].xyz;
    }

    // Adjust the ray origin and direction if depth of field is enabled. The ray must pass through
    // the focal point (along the original direction, at the focal distance), with an origin that
    // is offset on the lens, represented as a disk.
    if (lensRadius > 0.0f)
    {
        float3 focalPoint   = origin + direction * focalDistance;
        float2 originOffset = sampleDisk(random2D(rng), lensRadius);
        origin              = origin + originOffset.x * right + originOffset.y * up;
        direction           = normalize(focalPoint - origin);
    }
}

template<typename T>
inline T interpolateVertexAttribute(thread T* attributes, float2 uv) {
    const T T0 = attributes[0];
    const T T1 = attributes[1];
    const T T2 = attributes[2];
    return (1.f - uv.x - uv.y) * T0 + uv.x * T1 + uv.y * T2;
}

// Traces a ray in the specified direction, returning the radiance from that direction.
bool traceInstanceRay(RaytracingAccelerationStructure scene,
                      float3 origin, float3 dir, float tMin, thread InstanceRayPayload& rayPayload)
{

    // Set the force opaque ray flag to treat all objects as opaque, so that the any hit shader is
    // not called.

    // Prepare the ray.
    RayDesc ray;
    ray.origin    = origin;
    ray.direction = dir;
    ray.min_distance = tMin;
    ray.max_distance = INFINITY;

    rayPayload.clear();

    // Trace the ray.

    intersector<triangle_data, instancing, world_space_data, max_levels<2>> i;
    i.accept_any_intersection(false);
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(forced_opacity::opaque);
    typename intersector<triangle_data, instancing, world_space_data, max_levels<2>>::result_type intersection;
    intersection = i.intersect(ray, scene, 2);

    if(intersection.type == intersection_type::none) {
        rayPayload.clear();
    }
    else {
        float2 barycentricCoords = intersection.triangle_barycentric_coord;
        rayPayload.bary2 = barycentricCoords;
        rayPayload.distance = intersection.distance;
        rayPayload.instanceIndex = intersection.instance_id[0]; // should be max_levels - 1
        uint triangleIndex = intersection.primitive_id;
        InstanceRecord record = gInstanceBuffer[intersection.instance_id[0]];
        Geometry gGeometry(record);
        float3x4 objToWorld = transpose(intersection.object_to_world_transform);
        rayPayload.objToWorld = objToWorld;
        ShadingData shading = computeShadingData(gGeometry, triangleIndex, barycentricCoords, objToWorld);
        rayPayload.geomPosition = shading.geomPosition;
        rayPayload.texCoord = shading.texCoord;
        rayPayload.objectNormal = shading.objectNormal;
        rayPayload.objectPosition = shading.objectPosition;
        rayPayload.indices = shading.indices;
        rayPayload.instanceBufferOffset = record.material;
        return true;
    }
    return rayPayload.hit();
}


// The payload for shadow rays.
struct ShadowRayPayload
{
    float3 visibility;
};

// Traces a shadow ray for the specified sample position and light direction, returning the
// visibiliity of the light in that direction.
float3 traceShadowRay(RaytracingAccelerationStructure scene, float3 origin, float3 L, float tMin,
                      bool onlyOpaqueHits, int depth, int maxDepth)
{
    // If the maximum trace recursion depth has been reached, treat the light as visible. This will
    // mean there is more light than expected, but that works better than blocking the light, for
    // typical scenes. Depth is not tracked in the shadow ray payload because there should not be
    // any further ray tracing from shadow rays. Shadow rays are traced one level deeper than
    // radiance rays, to avoid distracting shadow-less results.
    if (depth == maxDepth + 1)
    {
        return WHITE;
    }

    // Build a ray payload, starting with either:
    // - Full visibility for normal shadow rays. This assumes a miss since the any hit shader
    //   will reduce visibility from there.
    // - Zero visibility for shadow rays that treat all objects as opaque. if the miss shader is
    //   evaluated, it will set full visibility.
    ShadowRayPayload rayPayload;
    rayPayload.visibility = onlyOpaqueHits ? BLACK : WHITE;

    // Prepare the shadow ray, with a small offset from the origin to avoid self-intersection.
    RayDesc ray;
    ray.origin    = origin;
    ray.direction = L;
    ray.min_distance = tMin;
    ray.max_distance = INFINITY;

    // Trace the shadow ray. This is different from standard tracing for performance and behavior,
    // as follows:
    // - Skip the closest hit shader, as none is needed for shadow rays.
    // - For normal shadow rays, which support non-opaque objects, the any hit shader is used. So
    //   don't use a miss shader, as none is needed when the any hit shader is used.
    // - For shadow rays that treat all objects as opaque (i.e. forced opaque):
    //   - Stop searching as soon as the first hit is found, which is not necessarily the closest.
    //   - Treat all intersections as opaque, so that the shadow any hit shader is never called.
    //   - Use the shadow miss shader, which simply sets full visibility on ray payload.

#if DIRECTX
    uint missShaderIndex = onlyOpaqueHits ? kMissShadow : kMissNull;
#else
    uint missShaderIndex = kMissShadow;
#endif
    intersector<triangle_data, instancing, world_space_data, max_levels<2>> i;
    i.accept_any_intersection(onlyOpaqueHits);
    i.assume_geometry_type(geometry_type::triangle);
    i.force_opacity(onlyOpaqueHits ? forced_opacity::opaque : forced_opacity::none);
    typename intersector<triangle_data, instancing, world_space_data, max_levels<2>>::result_type intersection;
    intersection = i.intersect(ray, scene, 2);

    if(intersection.type == intersection_type::none) {
        rayPayload.visibility = WHITE;
    }
    else {
        float2 barycentricCoords = intersection.triangle_barycentric_coord;
        Geometry gGeometry(gInstanceBuffer[intersection.instance_id[0]]);

        // Get the interpolated vertex data for the hit triangle, at the hit barycentric coordinates.
        uint triangleIndex = intersection.primitive_id;
        ShadingData shading =
            computeShadingData(gGeometry, triangleIndex, barycentricCoords, transpose(intersection.object_to_world_transform));

        // Get material buffer offset from instance data header.
        int materialBufferOffset = instanceMaterialBufferOffset(gGeometry.getBufferOffset());

        // Retrieve the shader index for intersected surface material.
        // This is stored in the material header in the global material byte buffer,
        // at a position indicated by instance's gMaterialBufferOffset.
        int shaderIndex = materialShaderIndex(materialBufferOffset);

        // Normal parameters generated by material's evaluate function (not used in shadow any hit shader.)
        float3 materialNormal = shading.normal;
        bool isGeneratedNormal = false;

        Material material = evaluateMaterialFromConstants(gInstanceBuffer[intersection.instance_id[0]], shaderIndex, shading,
                                                                         materialBufferOffset, materialNormal, isGeneratedNormal);

        // Compute opacity from material opacity and transmission.
        float3 opacity =
            material.opacity * (WHITE - material.transmission * material.transmissionColor);
        rayPayload.visibility *= 1.0f - opacity;
    }
    // Return the shadow ray visibility.
    return rayPayload.visibility;
}

#endif // __RAY_TRACE_H__
