// Section 2: Sphere Tracing with Differentiable Distance Functions
// SDF primitives, CSG operations, and differentiable sphere tracing.

__exported import common;
__exported import shading;
import slangpy;

// ---- SDF Primitives ----

[Differentiable]
float sdSphere(float3 p, float3 center, float radius)
{
    return length(p - center) - radius;
}

[Differentiable]
float sdBox(float3 p, float3 center, float3 halfExtents)
{
    float3 d = abs(p - center) - halfExtents;
    return length(max(d, 0.0f)) + min(max(d.x, max(d.y, d.z)), 0.0f);
}

[Differentiable]
float sdTorus(float3 p, float3 center, float2 radii)
{
    float3 q = p - center;
    float2 xz = float2(q.x, q.z);
    float2 t = float2(length(xz) - radii.x, q.y);
    return length(t) - radii.y;
}

[Differentiable]
float sdPlane(float3 p, float3 normal, float offset)
{
    return dot(p, normalize(normal)) - offset;
}

// ---- CSG Operations ----

[Differentiable]
float opSmoothUnion(float d1, float d2, float k)
{
    float h = max(k - abs(d1 - d2), 0.0f) / k;
    return min(d1, d2) - h * h * k * 0.25f;
}

[Differentiable]
float opSmoothIntersection(float d1, float d2, float k)
{
    float h = max(k - abs(d1 - d2), 0.0f) / k;
    return max(d1, d2) + h * h * k * 0.25f;
}

[Differentiable]
float opSmoothSubtraction(float d1, float d2, float k)
{
    float h = max(k - abs(-d1 - d2), 0.0f) / k;
    return max(-d1, d2) + h * h * k * 0.25f;
}

// ---- SDF Scene Parameters ----

struct SDFSceneParams : IDifferentiable
{
    float3 sphereCenter;
    float  sphereRadius;
    float3 boxCenter;
    float3 boxHalfExtents;
    float3 torusCenter;
    float2 torusRadii;
    float  smoothK;
};

// Evaluate the combined SDF scene.
[Differentiable]
float evalScene(float3 p, SDFSceneParams params)
{
    float dSphere = sdSphere(p, params.sphereCenter, params.sphereRadius);
    float dBox = sdBox(p, params.boxCenter, params.boxHalfExtents);
    float dTorus = sdTorus(p, params.torusCenter, params.torusRadii);

    float d = opSmoothUnion(dSphere, dBox, params.smoothK);
    d = opSmoothUnion(d, dTorus, params.smoothK);
    return d;
}

// Compute SDF normal via central differences.
[Differentiable]
float3 sdfNormal(float3 p, SDFSceneParams params)
{
    const float eps = 0.001f;
    float3 n = float3(
        evalScene(p + float3(eps, 0, 0), params) - evalScene(p - float3(eps, 0, 0), params),
        evalScene(p + float3(0, eps, 0), params) - evalScene(p - float3(0, eps, 0), params),
        evalScene(p + float3(0, 0, eps), params) - evalScene(p - float3(0, 0, eps), params)
    );
    return normalize(n);
}

// Differentiable sphere tracing with fixed iteration count.
// Convergence checks are non-differentiable; position updates are differentiable.
static const int TRACE_MAX_STEPS = 96;
static const float TRACE_MAX_DIST = 50.0f;
static const float TRACE_HIT_EPS = 0.001f;

[Differentiable]
float sphereTrace(Ray ray, SDFSceneParams params)
{
    float t = 0.0f;

    [MaxIters(96)]
    for (int i = 0; i < TRACE_MAX_STEPS; i++)
    {
        float3 p = ray.origin + ray.direction * t;
        float d = evalScene(p, params);

        if (detach(d) < TRACE_HIT_EPS)
            return t;

        if (detach(t) > TRACE_MAX_DIST)
            return -1.0f;

        t = t + d;
    }
    return -1.0f;
}

// ---- SDF Rendering ----

// Non-differentiable display entry point.
float4 renderSDFScene(
    Camera camera,
    int2 pixelCoord,
    float2 resolution,
    SDFSceneParams sdfParams,
    Material material,
    PointLight lights[2],
    int numLights)
{
    Ray ray = camera.generateRay(float2(pixelCoord), resolution);
    float t = sphereTrace(ray, sdfParams);

    if (t < 0.0f)
    {
        float skyT = ray.direction.y * 0.5f + 0.5f;
        float3 bg = lerp(float3(0.8f, 0.85f, 0.95f), float3(0.4f, 0.6f, 0.9f), skyT);
        return float4(bg, 1.0f);
    }

    float3 hitPos = ray.origin + ray.direction * t;
    float3 normal = sdfNormal(hitPos, sdfParams);
    float3 viewDir = normalize(camera.position - hitPos);

    float3 color = float3(0.0f);
    color += material.albedo * 0.03f;

    for (int i = 0; i < numLights; i++)
    {
        color += blinnPhongShade(hitPos, normal, material, lights[i], viewDir);
    }
    return float4(toneMap(color), 1.0f);
}

// ---- SDF Space Loss and Gradient Computation ----
// Optimizes SDF parameters by comparing distance field values at sample points,
// avoiding sphere tracing discontinuities entirely.  evalScene is smoothly
// differentiable everywhere, so gradients are reliable.

[Differentiable]
float sdfSampleLoss(
    no_diff float3 samplePoint,
    SDFSceneParams sdfParams,
    no_diff SDFSceneParams targetParams)
{
    float d_current = evalScene(samplePoint, sdfParams);
    float d_target  = evalScene(samplePoint, targetParams);
    float diff = d_current - d_target;
    return diff * diff;
}

// Pack SDFSceneParams differential into 4 x float4 for atomic accumulation.
// Layout: [0]={sphereCenter, sphereRadius}  [1]={boxCenter, boxHalfExtents.x}
//         [2]={boxHalfExtents.yz, torusCenter.xy}  [3]={torusCenter.z, torusRadii, smoothK}
void packSDFGradients(SDFSceneParams.Differential d, AtomicTensor<float4, 1> grad)
{
    grad.add({0}, float4(d.sphereCenter, d.sphereRadius));
    grad.add({1}, float4(d.boxCenter, d.boxHalfExtents.x));
    grad.add({2}, float4(d.boxHalfExtents.y, d.boxHalfExtents.z, d.torusCenter.x, d.torusCenter.y));
    grad.add({3}, float4(d.torusCenter.z, d.torusRadii.x, d.torusRadii.y, d.smoothK));
}

// Each sample point accumulates its contribution to the SDF parameter gradients.
void computeSDFGradients(
    float3 samplePoint,
    SDFSceneParams sdfParams,
    SDFSceneParams targetParams,
    AtomicTensor<float4, 1> paramGrad)
{
    var dParams = diffPair(sdfParams);

    bwd_diff(sdfSampleLoss)(
        samplePoint,
        dParams, targetParams,
        1.0f);

    packSDFGradients(dParams.d, paramGrad);
}

// Side-by-side display from two pre-rendered textures.
float4 renderSDFSideBySide(
    int2 pixelCoord,
    float2 resolution,
    Tensor<float4, 2> leftImage,
    Tensor<float4, 2> rightImage,
    float2 imageRes)
{
    int halfW = int(resolution.x) / 2;
    float u, v_coord;
    if (pixelCoord.x < halfW)
    {
        u = float(pixelCoord.x) / float(halfW);
        v_coord = float(pixelCoord.y) / resolution.y;
        int2 tc = int2(
            clamp(int(u * imageRes.x), 0, int(imageRes.x) - 1),
            clamp(int(v_coord * imageRes.y), 0, int(imageRes.y) - 1));
        return leftImage.load(tc);
    }
    else
    {
        u = float(pixelCoord.x - halfW) / float(halfW);
        v_coord = float(pixelCoord.y) / resolution.y;
        int2 tc = int2(
            clamp(int(u * imageRes.x), 0, int(imageRes.x) - 1),
            clamp(int(v_coord * imageRes.y), 0, int(imageRes.y) - 1));
        return rightImage.load(tc);
    }
}
