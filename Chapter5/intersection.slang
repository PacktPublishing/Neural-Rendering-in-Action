import common;
import slangpy;

// Moller-Trumbore ray-triangle intersection.
// Returns true if the ray hits the triangle, writing t and barycentric coords.
[Differentiable]
bool rayTriangleIntersect(
    Ray ray,
    float3 v0, float3 v1, float3 v2,
    out float t, out float2 bary)
{
    t = 0.0f;
    bary = float2(0.0f);

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h = cross(ray.direction, e2);
    float a = dot(e1, h);

    const float EPSILON = 1e-7f;
    if (abs(a) < EPSILON)
        return false;

    float f = 1.0f / a;
    float3 s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f)
        return false;

    float3 q = cross(s, e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f)
        return false;

    float tHit = f * dot(e2, q);
    if (tHit < EPSILON)
        return false;

    t = tHit;
    bary = float2(u, v);
    return true;
}

// Compute face normal from triangle vertices (differentiable w.r.t. vertices).
[Differentiable]
float3 triangleNormal(float3 v0, float3 v1, float3 v2)
{
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 n = cross(e1, e2);
    float len = length(n);
    if (len < 1e-10f)
        return float3(0, 1, 0);
    return n / len;
}

// Find closest hit triangle in a mesh (non-differentiable; discrete selection).
// Returns HitInfo and the index of the hit triangle (-1 if none).
HitInfo traceMeshClosest(
    Ray ray,
    Tensor<float3, 1> vertices,
    int numTris,
    out int hitTriIdx)
{
    HitInfo closest = noHit();
    hitTriIdx = -1;

    for (int i = 0; i < numTris; i++)
    {
        float3 v0 = vertices[i * 3 + 0];
        float3 v1 = vertices[i * 3 + 1];
        float3 v2 = vertices[i * 3 + 2];

        float t;
        float2 bary;
        bool hit = rayTriangleIntersect(ray, v0, v1, v2, t, bary);

        if (hit && t < closest.t)
        {
            closest.hit = true;
            closest.t = t;
            closest.bary = bary;
            closest.normal = triangleNormal(v0, v1, v2);
            closest.position = ray.origin + ray.direction * t;
            closest.materialIdx = 0;
            hitTriIdx = i;
        }
    }
    return closest;
}
