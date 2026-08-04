// Shared constants for differentiable rendering.
// Included by both MainEntryPoints.slang (forward pass) and
// DiffComputeShaders.slang (backward pass) to keep layouts in sync.
#ifndef __DIFF_RENDERING_CONSTANTS_H__
#define __DIFF_RENDERING_CONSTANTS_H__

// Maximum number of bounces recorded per pixel for the backward pass.
// Each bounce stores DIFF_RECORD_STRIDE float4s of material + geometry data.
#define MAX_PATH_BOUNCES 4

// Number of float4s per bounce record in gPathRecords.
// Layout per bounce:
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

// Number of gradient floats per pixel in gMaterialGrads.
// [0-2] baseColor, [3] roughness, [4] metalness,
// [5-7] emissionColor, [8] emission, [9] specular,
// [10-12] specularColor, [13] specularIOR, [14] specularAnisotropy
#define MATERIAL_GRAD_STRIDE 15

// Total float4s per pixel in gPathRecords = header (1) + bounces * stride.
// The first float4 per pixel is a header storing the valid bounce count.
//   header[0].x = numValidBounces (uint, stored via asuint)
//   header[0].yzw = reserved
#define DIFF_PIXEL_HEADER_SIZE 1
#define DIFF_PIXEL_TOTAL_STRIDE (DIFF_PIXEL_HEADER_SIZE + MAX_PATH_BOUNCES * DIFF_RECORD_STRIDE)

#endif // __DIFF_RENDERING_CONSTANTS_H__
