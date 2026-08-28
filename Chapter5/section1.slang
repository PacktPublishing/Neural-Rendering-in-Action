//   - Forward rendering via ray casting + Blinn-Phong shading
//   - Backward-mode autodiff through the full render pipeline
//   - Gradient accumulation for material optimization

__exported import common;
__exported import intersection;
__exported import shading;
import slangpy;

// ============================================================================
// Differentiable per-pixel loss
// ============================================================================

// Shade a single triangle hit (differentiable w.r.t. vertices and material).
[Differentiable]
float3 shadeTriangleHit(
    float3 v0, float3 v1, float3 v2,
    no_diff Ray ray,
    Material material,
    PointLight lights[2],
    int numLights,
    no_diff float3 camPos)
{
    float t;
    float2 bary;
    bool hit = rayTriangleIntersect(ray, v0, v1, v2, t, bary);

    if (!hit)
        return float3(0.0f);

    float3 hitPos = ray.origin + ray.direction * t;
    float3 normal = triangleNormal(v0, v1, v2);
    float3 viewDir = normalize(camPos - hitPos);

    HitInfo hitInfo;
    hitInfo.t = t;
    hitInfo.normal = normal;
    hitInfo.position = hitPos;
    hitInfo.bary = bary;
    hitInfo.materialIdx = 0;
    hitInfo.hit = true;

    return shadeHit(hitInfo, material, lights, numLights, viewDir);
}

// Per-pixel MSE loss computing the target on-the-fly from a target material.
// Both rendered and target go through the SAME compiled shadeTriangleHit,
// guaranteeing zero loss and zero gradient when materials match exactly
[Differentiable]
float pixelLossMat(
    float3 v0, float3 v1, float3 v2,
    no_diff Ray ray,
    Material material,
    no_diff Material targetMaterial,
    PointLight lights[2],
    int numLights,
    no_diff float3 camPos)
{
    float3 rendered = shadeTriangleHit(v0, v1, v2, ray, material, lights, numLights, camPos);
    float3 target = detach(shadeTriangleHit(v0, v1, v2, ray, targetMaterial, lights, numLights, camPos));
    float3 diff = rendered - target;
    return dot(diff, diff);
}

// ============================================================================
// Forward rendering
// ============================================================================

// Display rendering (with tone mapping for visual quality).
float4 renderScene(
    Camera camera,
    int2 pixelCoord,
    float2 resolution,
    Tensor<float3, 1> vertices,
    int numTris,
    Material material,
    PointLight lights[2],
    int numLights)
{
    Ray ray = camera.generateRay(float2(pixelCoord), resolution);
    int hitTriIdx;
    HitInfo hit = traceMeshClosest(ray, vertices, numTris, hitTriIdx);

    if (!hit.hit)
    {
        float skyT = ray.direction.y * 0.5f + 0.5f;
        float3 bg = lerp(float3(0.8f, 0.85f, 0.95f), float3(0.4f, 0.6f, 0.9f), skyT);
        return float4(bg, 1.0f);
    }

    float3 viewDir = normalize(camera.position - hit.position);
    float3 color = shadeHit(hit, material, lights, numLights, viewDir);
    return float4(toneMap(color), 1.0f);
}

// ============================================================================
// Backward pass: compute dLoss/dAlbedo via autodiff
// ============================================================================

// Each pixel accumulates its contribution to the albedo gradient atomically.
// Uses pixelLossMat so both rendered and target colors are computed inside the
// same auto diff code path
void computeAlbedoGradients(
    Camera camera,
    int2 pixelCoord,
    float2 resolution,
    Tensor<float3, 1> vertices,
    int numTris,
    Material material,
    Material targetMaterial,
    PointLight lights[2],
    int numLights,
    AtomicTensor<float3, 1> albedoGrad)
{
    Ray ray = camera.generateRay(float2(pixelCoord), resolution);
    int hitTriIdx;
    HitInfo hit = traceMeshClosest(ray, vertices, numTris, hitTriIdx);

    if (!hit.hit)
        return;

    float3 v0 = vertices[hitTriIdx * 3 + 0];
    float3 v1 = vertices[hitTriIdx * 3 + 1];
    float3 v2 = vertices[hitTriIdx * 3 + 2];

    var dv0 = diffPair(v0);
    var dv1 = diffPair(v1);
    var dv2 = diffPair(v2);
    var dMat = diffPair(material);
    PointLight lightsArr[2] = lights;
    var dLights = diffPair(lightsArr);

    // Backward pass with seed = 1.0 (dLoss/dLoss = 1)
    bwd_diff(pixelLossMat)(
        dv0, dv1, dv2,
        ray, dMat, targetMaterial, dLights, numLights,
        camera.position,
        1.0f);

    albedoGrad.add({0}, dMat.d.albedo);
}

// ============================================================================
// Display helpers
// ============================================================================

// Side-by-side: reference image (left) and current rendering (right).
float4 renderSideBySide(
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

