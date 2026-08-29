//   - Forward rendering via ray casting + Blinn-Phong shading
//   - Backward-mode autodiff through the shading of the hit point
//   - Gradient accumulation for material optimization
//
// The albedo is the only unknown here. Geometry and lighting are fixed, so
// they are marked no_diff and the closest-hit search runs outside the
// differentiated region, its result reaching the loss as a constant.

__exported import common;
__exported import intersection;
__exported import shading;
import slangpy;

// ============================================================================
// Differentiable per-pixel loss
// ============================================================================

// Shade a surface point that has already been located. The hit data is no_diff
// because it comes from the closest-hit search, which is not differentiated;
// the material is the only parameter carrying a derivative.
[Differentiable]
float3 shadeSurface(
    no_diff HitInfo hit,
    Material material,
    no_diff PointLight lights[2],
    int numLights,
    no_diff float3 camPos)
{
    float3 viewDir = normalize(camPos - hit.position);
    return shadeHit(hit, material, lights, numLights, viewDir);
}

// Per-pixel squared error between the current material and the target material
// at the same surface point. Both evaluate through the same compiled
// shadeSurface, so the loss and its gradient are exactly zero when the two
// materials match.
[Differentiable]
float pixelLossMat(
    no_diff HitInfo hit,
    Material material,
    no_diff Material targetMaterial,
    no_diff PointLight lights[2],
    int numLights,
    no_diff float3 camPos)
{
    float3 rendered = shadeSurface(hit, material, lights, numLights, camPos);
    float3 target = shadeSurface(hit, targetMaterial, lights, numLights, camPos);
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
    HitInfo hit = traceMeshClosest(ray, vertices, numTris);

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

    // The closest-hit search picks one triangle out of many. That choice is
    // discrete, so it stays outside the differentiated region.
    HitInfo hit = traceMeshClosest(ray, vertices, numTris);

    if (!hit.hit)
        return;

    // The material is the only unknown, so it is the only differential pair.
    var dMat = diffPair(material);

    // Backward pass with seed = 1.0 (dLoss/dLoss = 1)
    bwd_diff(pixelLossMat)(
        hit, dMat, targetMaterial, lights, numLights,
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
