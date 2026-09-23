import "slangpy";

struct Ray : IDifferentiable
{
    float3 origin;
    float3 direction;
};

struct Camera
{
    float3 position;
    float3 forward;
    float3 up;
    float3 right;
    float fov;      // vertical field-of-view in radians
    float aspect;   // width / height

    [Differentiable]
    [NoDiffThis]
    Ray generateRay(float2 pixel, float2 resolution)
    {
        float2 ndc = (pixel + 0.5f) / resolution * 2.0f - 1.0f;
        ndc.y = -ndc.y;

        float tanHalfFov = tan(fov * 0.5f);
        float3 dir = normalize(
            forward + right * (ndc.x * tanHalfFov * aspect)
                    + up    * (ndc.y * tanHalfFov)
        );

        Ray ray;
        ray.origin = position;
        ray.direction = dir;
        return ray;
    }
};

struct Material : IDifferentiable
{
    float3 albedo;
    float roughness;
    float metallic;
};


struct PointLight : IDifferentiable
{
    float3 position;
    float3 color;
    float intensity;
};

struct HitInfo : IDifferentiable
{
    float t;
    float3 normal;
    float3 position;
    float2 bary;
    int materialIdx;
    bool hit;
};

static HitInfo noHit()
{
    HitInfo h;
    h.t = 1e30f;
    h.normal = float3(0, 1, 0);
    h.position = float3(0);
    h.bary = float2(0);
    h.materialIdx = -1;
    h.hit = false;
    return h;
}
