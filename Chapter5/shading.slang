import common;

// Safe clamped dot product that avoids hard zero gradients at the terminator.
[Differentiable]
float safeDot(float3 a, float3 b)
{
    float d = dot(a, b);
    return max(d, 0.001f);
}

// Blinn-Phong shading model (fully differentiable).
[Differentiable]
float3 blinnPhongShade(
    float3 hitPos,
    float3 normal,
    Material mat,
    PointLight light,
    float3 viewDir)
{
    float3 L = light.position - hitPos;
    float dist2 = dot(L, L);
    L = normalize(L);

    float NdotL = safeDot(normal, L);

    // Diffuse
    float3 diffuse = mat.albedo * NdotL;

    // Specular (Blinn-Phong)
    float3 H = normalize(L + viewDir);
    float NdotH = safeDot(normal, H);
    float shininess = 2.0f / (mat.roughness * mat.roughness + 1e-4f);
    float spec = pow(NdotH, shininess);
    float3 specColor = lerp(float3(0.04f), mat.albedo, mat.metallic);
    float3 specular = specColor * spec;

    float attenuation = light.intensity / (dist2 + 1e-4f);
    return (diffuse + specular) * light.color * attenuation;
}

// Shade a hit point with multiple lights (up to 4).
[Differentiable]
float3 shadeHit(
    HitInfo hit,
    Material mat,
    PointLight lights[2],
    no_diff int numLights,
    float3 viewDir)
{
    float3 color = float3(0.0f);
    float3 ambient = mat.albedo * 0.03f;
    color += ambient;

    [MaxIters(2)]
    for (int i = 0; i < numLights; i++)
    {
        color += blinnPhongShade(hit.position, hit.normal, mat, lights[i], viewDir);
    }

    return color;
}

// Reinhard tone mapping: compress HDR linear colors to [0,1] for display.
float3 toneMap(float3 color)
{
    return color / (color + 1.0f);
}
