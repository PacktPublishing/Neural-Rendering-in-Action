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

// //////// Global Macros ////////
#define ARCH_OS_MACOS
#define METAL_FEATURESET_MACOS_GPUFAMILY1_v3
#define METAL_FEATURESET_MACOS_GPUFAMILY1_v4
#define METAL_FEATURESET_MACOS_GPUFAMILY2_v1
#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_pack>
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wsign-compare"
using namespace metal;
using namespace raytracing;
#define double float
#define vec2 float2
#define vec3 float3
#define vec4 float4
#define mat2 float2x2
#define mat3 float3x3
#define mat4 float4x4
#define ivec2 int2
#define ivec3 int3
#define ivec4 int4
#define uvec2 uint2
#define uvec3 uint3
#define uvec4 uint4
#define bvec2 bool2
#define bvec3 bool3
#define bvec4 bool4
#define dvec2 float2
#define dvec3 float3
#define dvec4 float4
#define dmat2 float2x2
#define dmat3 float3x3
#define dmat4 float4x4
#define usampler1DArray texture1d_array<uint16_t>
#define sampler2DArray texture2d_array<float>
#define sampler2DShadow depth2d<float>
#define MAT4 mat4
#define REF(space,type) space type &
#define FORWARD_DECL(...)
#define ATOMIC_LOAD(a) atomic_load_explicit(&a, memory_order_relaxed)
#define ATOMIC_STORE(a, v) atomic_store_explicit(&a, v, memory_order_relaxed)
#define ATOMIC_ADD(a, v) atomic_fetch_add_explicit(&a, v, memory_order_relaxed)
#define ATOMIC_EXCHANGE(a, desired) atomic_exchange_explicit(&a, desired, memory_order_relaxed)
static int atomicCompSwap(device atomic_int *a, int expected, int desired) {
    int found = expected;
    while(!atomic_compare_exchange_weak_explicit(a, &found, desired,
        memory_order_relaxed, memory_order_relaxed)) {
        if (found != expected) { return found; }
        else { found = expected; }
    } return expected; }
static uint atomicCompSwap(device atomic_uint *a, uint expected, uint desired) {
    uint found = expected;
    while(!atomic_compare_exchange_weak_explicit(a, &found, desired,
        memory_order_relaxed, memory_order_relaxed)) {
        if (found != expected) { return found; }
        else { found = expected; }
    } return expected; }
#define ATOMIC_COMP_SWAP(a, expected, desired) atomicCompSwap(&a, expected, desired)

struct hgi_ivec3 { int    x, y, z;
  hgi_ivec3(int _x, int _y, int _z): x(_x), y(_y), z(_z) {}
};
struct hgi_vec3  { float  x, y, z;
  hgi_vec3(float _x, float _y, float _z): x(_x), y(_y), z(_z) {}
};
struct hgi_dvec3 { double x, y, z;
  hgi_dvec3(double _x, double _y, double _z): x(_x), y(_y), z(_z) {}
};
struct hgi_mat3 { float m00, m01, m02,
                        m10, m11, m12,
                        m20, m21, m22;
  hgi_mat3(float _00, float _01, float _02, \
           float _10, float _11, float _12, \
           float _20, float _21, float _22) \
             : m00(_00), m01(_01), m02(_02) \
             , m10(_10), m11(_11), m12(_12) \
             , m20(_20), m21(_21), m22(_22) {}
};
struct hgi_dmat3 { double m00, m01, m02,
                          m10, m11, m12,
                          m20, m21, m22;
  hgi_dmat3(double _00, double _01, double _02, \
            double _10, double _11, double _12, \
            double _20, double _21, double _22) \
              : m00(_00), m01(_01), m02(_02) \
              , m10(_10), m11(_11), m12(_12) \
              , m20(_20), m21(_21), m22(_22) {}
};

static mat4 inverse_fast(float4x4 a) { return transpose(a); }
static mat4 inverse(float4x4 a) {
    float b00 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    float b01 = a[0][0] * a[1][2] - a[0][2] * a[1][0];
    float b02 = a[0][0] * a[1][3] - a[0][3] * a[1][0];
    float b03 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
    float b04 = a[0][1] * a[1][3] - a[0][3] * a[1][1];
    float b05 = a[0][2] * a[1][3] - a[0][3] * a[1][2];
    float b06 = a[2][0] * a[3][1] - a[2][1] * a[3][0];
    float b07 = a[2][0] * a[3][2] - a[2][2] * a[3][0];
    float b08 = a[2][0] * a[3][3] - a[2][3] * a[3][0];
    float b09 = a[2][1] * a[3][2] - a[2][2] * a[3][1];
    float b10 = a[2][1] * a[3][3] - a[2][3] * a[3][1];
    float b11 = a[2][2] * a[3][3] - a[2][3] * a[3][2];
    float invdet = 1.0 / (b00 * b11 - b01 * b10 + b02 * b09 +
                          b03 * b08 - b04 * b07 + b05 * b06);
    return mat4(a[1][1] * b11 - a[1][2] * b10 + a[1][3] * b09,
                a[0][2] * b10 - a[0][1] * b11 - a[0][3] * b09,
                a[3][1] * b05 - a[3][2] * b04 + a[3][3] * b03,
                a[2][2] * b04 - a[2][1] * b05 - a[2][3] * b03,
                a[1][2] * b08 - a[1][0] * b11 - a[1][3] * b07,
                a[0][0] * b11 - a[0][2] * b08 + a[0][3] * b07,
                a[3][2] * b02 - a[3][0] * b05 - a[3][3] * b01,
                a[2][0] * b05 - a[2][2] * b02 + a[2][3] * b01,
                a[1][0] * b10 - a[1][1] * b08 + a[1][3] * b06,
                a[0][1] * b08 - a[0][0] * b10 - a[0][3] * b06,
                a[3][0] * b04 - a[3][1] * b02 + a[3][3] * b00,
                a[2][1] * b02 - a[2][0] * b04 - a[2][3] * b00,
                a[1][1] * b07 - a[1][0] * b09 - a[1][2] * b06,
                a[0][0] * b09 - a[0][1] * b07 + a[0][2] * b06,
                a[3][1] * b01 - a[3][0] * b03 - a[3][2] * b00,
                a[2][0] * b03 - a[2][1] * b01 + a[2][2] * b00) * invdet;
}

#define in /*in*/
#define radians(d) (d * 0.01745329252)
#define noperspective /*center_no_perspective MTL_FIXME*/
#define dFdx    dfdx
#define dFdy    dfdy
#define lessThan(a, b) ((a) < (b))
#define lessThanEqual(a, b) ((a) <= (b))
#define greaterThan(a, b) ((a) > (b))
#define greaterThanEqual(a, b) ((a) >= (b))
#define equal(a, b) ((a) == (b))
#define notEqual(a, b) ((a) != (b))
union HgiPackedf16 { uint i; half2 h; };
static vec2 unpackHalf2x16(uint val)
{
    HgiPackedf16 v;
    v.i = val;
    return vec2(v.h.x, v.h.y);
}
static uint packHalf2x16(vec2 val)
{
    HgiPackedf16 v;
    v.h = half2(val.x, val.y);
    return v.i;
}
template <typename T>
T mod(T y, T x) { return fmod(y, x); }

template <typename T>
T atan(T y, T x) { return atan2(y, x); }

template <typename T>
T bitfieldReverse(T x) { return reverse_bits(x); }

template <typename T>
T bitfieldExtract(T value, int offset, int bits) {
  return extract_bits(value, offset, bits); }

template <typename T>
int imageSize1d(T texture) {
    return int(texture.get_width());
}
template <typename T>
ivec2 imageSize2d(T texture) {
    return ivec2(texture.get_width(), texture.get_height());
}
template <typename T>
ivec3 imageSize3d(T texture) {
    return ivec3(texture.get_width(),
        texture.get_height(), texture.get_depth());
}
template <typename T>
static ivec2 textureSize(T texture, uint lod = 0) {
    return ivec2(texture.get_width(lod), texture.get_height(lod));
}
static ivec2 textureSize(texture1d_array<uint16_t> texture, uint lod = 0) {
    return ivec2(texture.get_width(),
        texture.get_array_size());
}
static ivec3 textureSize(texture2d_array<float> texture, uint lod = 0) {
    return ivec3(texture.get_width(lod),
        texture.get_height(lod), texture.get_array_size());
}
template <typename T>
static int textureSize1d(T texture, uint lod = 0) {
    return int(texture.get_width());
}
template <typename T>
static ivec2 textureSize2d(T texture, uint lod = 0) {
    return ivec2(texture.get_width(lod), texture.get_height(lod));
}
template <typename T>
static ivec3 textureSize3d(T texture, uint lod = 0) {
    return ivec3(texture.get_width(lod),
        texture.get_height(lod), texture.get_depth(lod));
}

template<typename T, typename Tc>
float4 texelFetch(T texture, Tc coords, uint lod = 0) {
    return texture.read(uint2(coords), lod);
}
template<typename Tc>
uint4 texelFetch(texture1d_array<uint16_t> texture, Tc coords, uint lod = 0) {
    return uint4(texture.read((uint)coords.x, (uint)coords.y, 0));
}
template<typename Tc>
vec4 texelFetch(texture2d_array<float> texture, Tc coords, uint lod = 0) {
    return texture.read(uint2(coords.xy), (uint)coords.z, 0);
}
#define textureQueryLevels(texture) texture.get_num_mip_levels()
template <typename T, typename Tv>
void imageStore(T texture, short2 coords, Tv color) {
    return texture.write(color, ushort2(coords.x, coords.y));
}
template <typename T, typename Tv>
void imageStore(T texture, int2 coords, Tv color) {
    return texture.write(color, uint2(coords.x, coords.y));
}

constexpr sampler texelSampler(address::clamp_to_edge,
                               filter::linear);
template<typename T, typename Tc>
float4 texture(T texture, Tc coords) {
    return texture.sample(texelSampler, coords);
}
template<typename Tc>
vec4 texture(texture2d_array<float> texture, Tc coords) {
    return texture.sample(texelSampler, coords.xy, coords.z);
}
#define discard discard_fragment(); discarded_fragment = true;

// //////// Global Member Declarations ////////
struct MSLCsUniforms {
vec3 gSettings_brightness[[]];
int gSettings_debugMode[[]];
vec2 gSettings_range[[]];
bool gSettings_isDenoisingEnabled[[]];
bool gSettings_isToneMappingEnabled[[]];
bool gSettings_isGammaCorrectionEnabled[[]];
bool gSettings_isAlphaEnabled[[]];
};

struct PostProcessingData {
    packed_float3 brightness;
    int debugMode;
    packed_float2 range;
    int isDenoisingEnabled;
    int isToneMappingEnabled;
    int isGammaCorrectionEnabled;
    int isAlphaEnabled;
};

struct MSLBufferBindings {
    device void* buf_0;
    device void* buf_1;
    device void* buf_2;
    constant PostProcessingData* postProcessingData;
    device void* buf_4;
    device void* buf_5;
    device void* buf_6;
};

struct MSLSamplerBindings {
sampler samplerBind_outTexture[[id(0)]];
sampler samplerBind_accumulationTexture[[id(1)]];
};

struct MSLTextureBindings {
texture2d<half, access::write> textureBind_outTexture[[id(0)]];
texture2d<float> textureBind_accumulationTexture[[id(1)]];
};

struct ProgramScope_ComputePostProcess {

// //////// Scope Structs ////////

// //////// Scope Member Declarations ////////
uvec3 hd_GlobalInvocationID;
vec3 gSettings_brightness;
int gSettings_debugMode;
vec2 gSettings_range;
bool gSettings_isDenoisingEnabled;
bool gSettings_isToneMappingEnabled;
bool gSettings_isGammaCorrectionEnabled;
bool gSettings_isAlphaEnabled;
sampler samplerBind_outTexture;
texture2d<half, access::write> textureBind_outTexture;
sampler samplerBind_accumulationTexture;
texture2d<float> textureBind_accumulationTexture;

// //////// Scope Function Definitions ////////
#define HgiGetSampler_outTexture() textureBind_outTexture
void HgiSet_outTexture(ivec2 uv, vec4 data) {
    imageStore(textureBind_outTexture, uv, half4(data));
}
ivec2 HgiGetSize_outTexture() {
    return imageSize2d(textureBind_outTexture);
}
#define HgiGetSampler_accumulationTexture() textureBind_accumulationTexture
vec4 HgiGet_accumulationTexture(vec2 coord) {
    vec4 result = is_null_texture(textureBind_accumulationTexture) ? 0 : vec4(textureBind_accumulationTexture.sample(samplerBind_accumulationTexture, coord));
    return result;
}
ivec2 HgiGetSize_accumulationTexture() {
    return textureSize2d(textureBind_accumulationTexture, 0);
}
vec4 HgiTexelFetch_accumulationTexture(ivec2 coord) {
    vec4 result = vec4(textureBind_accumulationTexture.read(ushort2(coord.x, coord.y)));
    return result;
}
vec4 HgiTextureLod_accumulationTexture(vec2 coord, float lod) {
    vec4 result = vec4(textureBind_accumulationTexture.sample(samplerBind_accumulationTexture, coord, level(lod)));
    return result;
}
    ProgramScope_ComputePostProcess(

// //////// Scope Constructor Declarations ////////
sampler _samplerBind_outTexture,
texture2d<half, access::write> _textureBind_outTexture,
sampler _samplerBind_accumulationTexture,
texture2d<float> _textureBind_accumulationTexture):

// //////// Scope Constructor Initialization ////////
samplerBind_outTexture(_samplerBind_outTexture),
textureBind_outTexture(_textureBind_outTexture),
samplerBind_accumulationTexture(_samplerBind_accumulationTexture),
textureBind_accumulationTexture(_textureBind_accumulationTexture){};

#line 1 "/Volumes/work/Autodesk/Aurora_Latest/Aurora/Libraries/Aurora/Source/HGI/Shaders/PostProcessing.glsl"














vec2 GetTexCoords(ivec2 outCoords)
{
 vec2 outDims = vec2(HgiGetSize_outTexture());
 
 vec2 texCoords = (vec2(outCoords) + vec2(0.5, 0.5)) / outDims;
 return texCoords;
}



vec3 toneMapACES(vec3 color)
{
 float a = 2.51f;
 float b = 0.03f;
 float c = 2.43f;
 float d = 0.59f;
 float e = 0.14f;

 return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}



vec3 linearTosRGB(vec3 color)
{
 vec3 sq1 = sqrt(color);
 vec3 sq2 = sqrt(sq1);
 vec3 sq3 = sqrt(sq2);

 return 0.662002687 * sq1 + 0.684122060 * sq2 - 0.323583601 * sq3 - 0.0225411470 * color;
}

void main(void) {
    
    ivec2 outCoords = ivec2(hd_GlobalInvocationID.xy);
    vec2 texCoords = GetTexCoords(outCoords);
    vec3 color = HgiTextureLod_accumulationTexture(texCoords, 0.0).rgb;

    color *= gSettings_brightness;

    if (gSettings_isToneMappingEnabled) {
        color = toneMapACES(color);
    }

    if (gSettings_isGammaCorrectionEnabled) {
        color = linearTosRGB(clamp(color,0.0,1.0));
    }

    HgiSet_outTexture(outCoords, vec4(color,1.0));
}

};

kernel void computeEntryPointPostProcessing(

// //////// Entry Point Parameter Declarations ////////
uvec3 hd_GlobalInvocationID[[thread_position_in_grid]],
const device MSLCsUniforms *csUniforms[[buffer(27)]],
const device MSLBufferBindings *bufferBindings[[buffer(30)]],
const device MSLSamplerBindings *samplerBindings[[buffer(28)]],
const device MSLTextureBindings *textureBindings[[buffer(29)]]){

constant PostProcessingData& gPostProcessingData = *bufferBindings->postProcessingData;

    ProgramScope_ComputePostProcess scope(

// //////// Scope Constructor Instantiation ////////
samplerBindings->samplerBind_outTexture,
textureBindings->textureBind_outTexture,
samplerBindings->samplerBind_accumulationTexture,
textureBindings->textureBind_accumulationTexture);

// //////// Entry Point Function Executions ////////
scope.hd_GlobalInvocationID = hd_GlobalInvocationID;

scope.gSettings_brightness = gPostProcessingData.brightness; // csUniforms->gSettings_brightness;
scope.gSettings_debugMode = gPostProcessingData.debugMode; //csUniforms->gSettings_debugMode;
scope.gSettings_range = gPostProcessingData.range; //csUniforms->gSettings_range;
scope.gSettings_isDenoisingEnabled = gPostProcessingData.isDenoisingEnabled; // csUniforms->gSettings_isDenoisingEnabled;
scope.gSettings_isToneMappingEnabled = gPostProcessingData.isToneMappingEnabled; // csUniforms->gSettings_isToneMappingEnabled;
scope.gSettings_isGammaCorrectionEnabled = gPostProcessingData.isGammaCorrectionEnabled; // csUniforms->gSettings_isGammaCorrectionEnabled;
scope.gSettings_isAlphaEnabled = gPostProcessingData.isAlphaEnabled; // csUniforms->gSettings_isAlphaEnabled;
scope.main();
}
