// SPDX-License-Identifier: Apache-2.0
//
// Sphere-tracing renderer, entirely on the GPU: one thread per
// pixel does the whole march-then-shade pipeline, no NumPy anywhere in it.
// Shading normals come from CENTRAL DIFFERENCES of the SDF taken inside this
// same kernel -- six extra forward evaluations, no autodiff -- rather than
// the analytic input-gradient the book derives in 5.3. For a renderer this
// is the standard, much simpler choice (production sphere tracers do the
// same): it is visually indistinguishable from the analytic gradient for a
// smooth field, needs no [Differentiable] machinery at all, and so carries
// none of this toolchain's autodiff compile-time risk.
//
// `Sdf` is a Slang interface so this file stays generic over WHAT it
// traces: the exact analytic torus (`AnalyticSdf`, used for the reference
// render and for CSG in 5.6) or a trained `neural_sdf.Network`
// (`NeuralSdf`, defined in neural_sdf.slang, which implements this
// interface). Both go through the identical march/shade/composite code.

import slangpy;
__exported import common_shapes;

public interface Sdf
{
    float distance(float3 p);
}

public struct AnalyticSdf : Sdf
{
    public float distance(float3 p) { return torus_sdf(p); }
}

// -------------------------------------------------------------------------
// The march [5.4]: step by the SDF value, a guaranteed-safe distance to the
// nearest surface. step_scale < 1 adds a safety margin for fields that are
// only approximately unit-gradient (a trained network's is not exact).
// -------------------------------------------------------------------------
public static const int SPHERE_TRACE_MAX_STEPS = 96;

public bool sphere_trace<S : Sdf>(S sdf, float3 origin, float3 dir, out float t_hit,
                                  float t_max = 5.0f, float eps = 1e-3f,
                                  float step_scale = 0.9f)
{
    float t = 0.0f;
    [MaxIters(SPHERE_TRACE_MAX_STEPS)]
    for (int i = 0; i < SPHERE_TRACE_MAX_STEPS; ++i)
    {
        float3 p = origin + t * dir;
        float d = sdf.distance(p);
        if (d < eps) { t_hit = t; return true; }
        t += step_scale * max(d, 0.5f * eps);
        if (t > t_max) break;
    }
    t_hit = t;
    return false;
}

// Surface normal by central differences of the field itself -- cheap (six
// extra `distance` calls, all ordinary forward evaluations) and needs no
// differentiation of the field at all.
public float3 normal_fd<S : Sdf>(S sdf, float3 p, float h = 2e-3f)
{
    float3 g = float3(
        sdf.distance(p + float3(h, 0, 0)) - sdf.distance(p - float3(h, 0, 0)),
        sdf.distance(p + float3(0, h, 0)) - sdf.distance(p - float3(0, h, 0)),
        sdf.distance(p + float3(0, 0, h)) - sdf.distance(p - float3(0, 0, h)));
    float n = length(g);
    return n > 1e-12f ? g / n : float3(0, 0, 1);
}

// -------------------------------------------------------------------------
// One pixel, start to finish: camera ray -> march -> Lambert shade ->
// composite over a background colour. `eye`/`right`/`up`/`fwd` are the
// camera basis (built once on the Python side from eye/target/up -- three
// cross products, not worth a kernel of its own); `half_h`/`aspect` turn a
// pixel's (u, v) in [-1,1]^2 into a ray direction.
// -------------------------------------------------------------------------
public float3 render_pixel<S : Sdf>(S sdf, int px, int py, int width, int height,
                                    float3 eye, float3 fwd, float3 right, float3 up,
                                    float half_h, float3 light, float3 base_color,
                                    float3 bg_color)
{
    float aspect = float(width) / float(height);
    float u = (2.0f * (float(px) + 0.5f) / float(width) - 1.0f) * half_h * aspect;
    float v = (1.0f - 2.0f * (float(py) + 0.5f) / float(height)) * half_h;
    float3 dir = normalize(fwd + u * right + v * up);

    float t;
    if (!sphere_trace(sdf, eye, dir, t)) return bg_color;

    float3 p = eye + t * dir;
    float3 n = normal_fd(sdf, p);
    float3 l = normalize(light);
    float lam = clamp(dot(n, l), 0.0f, 1.0f);
    return base_color * (0.25f + 0.75f * lam);
}

// One thread per pixel, image laid out row-major (py * width + px).
public float3 render_analytic(int idx, int width, int height,
                              float3 eye, float3 fwd, float3 right, float3 up,
                              float half_h, float3 light, float3 base_color,
                              float3 bg_color)
{
    int px = idx % width, py = idx / width;
    AnalyticSdf sdf;
    return render_pixel(sdf, px, py, width, height, eye, fwd, right, up,
                        half_h, light, base_color, bg_color);
}
