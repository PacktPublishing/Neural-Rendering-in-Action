// SPDX-License-Identifier: Apache-2.0
//
// GPU training AND rendering for the appearance section's appearance networks (6.2
// albedo c(x), 6.3 radiance c(x, v)). HAS_VIEW selects which of the two
// networks this compiled module trains; the Python driver loads two
// separately-specialized modules, one per network -- same pattern as
// The occupancy section/5.
//
// Ground truth (`texture_albedo`, `radiance_gt`), sampling (GPU RNG, via
// `common_rng.slang`), and the loss are all in this file: nothing samples
// or labels a training batch on the CPU. Rendering composes this file's
// network with `neural_sdf`'s FROZEN geometry network (the SDF section's trained
// SDF): sphere-trace with the geometry field, then colour each hit with
// this appearance field -- geometry and appearance are two independently
// trained fields over the same coordinates, exactly as the book frames it.
//
// The FeatureGrid/HybridField race (Figure 6.7) is NOT here: see
// hybrid_race_train.slang.
//
// See neural_occupancy_fields/README.md and neural_signed_distance/README.md
// for the compile-time workarounds this reuses (L as a #define, a kernel
// warm-up before training, and persistent GPU buffers). The ReLU loops are
// [ForceUnroll]: under [MaxIters],
// Slang's backward saves the whole activation array on every iteration,
// which grows as width squared (see common_network.slang).

import slangpy;
__exported import common_network;
__exported import common_shapes;
__exported import common_rng;
__exported import neural_sdf;      // the FROZEN geometry network, for rendering
__exported import render;

#ifndef APP_L
#define APP_L 6
#endif
#ifndef HAS_VIEW
#define HAS_VIEW 0
#endif
#ifndef APP_HIDDEN_DEFINE
#define APP_HIDDEN_DEFINE 64
#endif
// Prefixed APP_ throughout: this module `__exported import`s neural_sdf
// (for rendering), which declares its own file-scope L/NUM_ENCODED/HIDDEN --
// plain names would collide. That's exactly why the width #define here is
// named APP_HIDDEN_DEFINE rather than APP_HIDDEN: the local const below
// keeps the name APP_HIDDEN (already distinct from neural_sdf's own
// HIDDEN), and a #define can't share its target const's identifier
// (self-referential substitution), so the macro needs its own, third name.
static const int APP_ENC_L = APP_L;
static const int APP_NUM_ENCODED = 3 + 6 * APP_ENC_L + (HAS_VIEW != 0 ? 3 : 0);
static const int APP_HIDDEN = APP_HIDDEN_DEFINE;

// Named distinctly from neural_sdf.Network (the frozen geometry field) --
// this chapter's convention is one Network struct per *_train.slang file,
// and rendering composes two of them here.
struct AppearanceNetwork
{
    NetworkParameters<APP_NUM_ENCODED, APP_HIDDEN> layer0;
    NetworkParameters<APP_HIDDEN, APP_HIDDEN> layer1;
    NetworkParameters<APP_HIDDEN, 3> layer2;

    [Differentiable]
    float3 eval(no_diff float3 p, no_diff float3 v)
    {
        float[APP_NUM_ENCODED] enc;
        enc[0] = p.x; enc[1] = p.y; enc[2] = p.z;
        [ForceUnroll]
        for (int k = 0; k < APP_ENC_L; ++k)
        {
            float freq = (1 << k) * 3.14159265f;
            enc[3 + 6 * k + 0] = sin(freq * p.x);
            enc[3 + 6 * k + 1] = sin(freq * p.y);
            enc[3 + 6 * k + 2] = sin(freq * p.z);
            enc[3 + 6 * k + 3] = cos(freq * p.x);
            enc[3 + 6 * k + 4] = cos(freq * p.y);
            enc[3 + 6 * k + 5] = cos(freq * p.z);
        }
#if HAS_VIEW
        enc[3 + 6 * APP_ENC_L + 0] = v.x;
        enc[3 + 6 * APP_ENC_L + 1] = v.y;
        enc[3 + 6 * APP_ENC_L + 2] = v.z;
#endif
        float[APP_HIDDEN] h0 = layer0.forward(enc);
        [ForceUnroll]
        for (int i = 0; i < APP_HIDDEN; ++i) h0[i] = max(h0[i], 0.0f);
        float[APP_HIDDEN] h1 = layer1.forward(h0);
        [ForceUnroll]
        for (int i = 0; i < APP_HIDDEN; ++i) h1[i] = max(h1[i], 0.0f);
        float[3] out = layer2.forward(h1);
        return float3(out[0], out[1], out[2]);
    }
}

float3 predict(float3 p, float3 v, AppearanceNetwork network) { return network.eval(p, v); }

float3 predict_batch(int idx, Tensor<float3, 1> pts, Tensor<float3, 1> views, AppearanceNetwork network)
{
    return predict(pts.load(idx), views.load(idx), network);
}

// -------------------------------------------------------------------------
// Ground truth [6.1] -- a procedural texture in the torus's own surface
// coordinates, plus a Blinn-Phong highlight for the view-dependent target.
// -------------------------------------------------------------------------

static const float3 LIGHT = float3(0.5f, -0.3f, 0.85f) / length(float3(0.5f, -0.3f, 0.85f));

float3 texture_albedo(float3 p)
{
    float2 uv = torus_uv(p);
    float s = 0.5f + 0.5f * sin(8.0f * uv.x) * sin(6.0f * uv.y);
    float3 terracotta = float3(0.85f, 0.45f, 0.25f);
    float3 blue = float3(0.20f, 0.45f, 0.75f);
    return terracotta * s + blue * (1.0f - s);
}

// `view`: surface -> eye, unit, pointing AWAY from the surface.
float3 radiance_gt(float3 p, float3 view)
{
    float3 n = torus_normal(p);
    float diff = clamp(dot(n, LIGHT), 0.0f, 1.0f);
    float3 h = normalize(LIGHT + view);
    float spec = pow(clamp(dot(n, h), 0.0f, 1.0f), 60.0f);
    float3 col = texture_albedo(p) * (0.20f + 0.80f * diff) + 0.9f * spec;
    return clamp(col, 0.0f, 1.0f);
}

// -------------------------------------------------------------------------
// Training [6.2 / 6.3] -- MSE, sampled and labelled on the GPU. One
// dispatch per step does sampling + labelling + forward + backward.
// -------------------------------------------------------------------------

[Differentiable]
float mse_loss(no_diff float3 p, no_diff float3 v, no_diff float3 gt, AppearanceNetwork network)
{
    float3 d = network.eval(p, v) - gt;
    return dot(d, d);
}

void calculate_albedo_grads(int idx, uint seed, int n, AppearanceNetwork network)
{
    Rng rng = Rng(seed, uint(idx));
    float u = rng.next_float(0.0f, 6.28318530718f);
    float v = rng.next_float(0.0f, 6.28318530718f);
    float3 p = torus_surface(float2(u, v));
    float3 gt = texture_albedo(p);
    bwd_diff(mse_loss)(p, float3(0, 0, 0), gt, network, 1.0f / float(n));
}

void calculate_radiance_grads(int idx, uint seed, int n, AppearanceNetwork network)
{
    Rng rng = Rng(seed, uint(idx));
    float u = rng.next_float(0.0f, 6.28318530718f);
    float v = rng.next_float(0.0f, 6.28318530718f);
    float3 p = torus_surface(float2(u, v));
    float3 nrm = torus_normal(p);
    float3 view = rng.next_unit_vector();
    if (dot(view, nrm) < 0.0f) view = -view;      // outward hemisphere
    float3 gt = radiance_gt(p, view);
    bwd_diff(mse_loss)(p, view, gt, network, 1.0f / float(n));
}

// -------------------------------------------------------------------------
// Rendering [6.2 / 6.3] -- sphere-trace the FROZEN geometry field
// (neural_sdf.Network), then colour each hit with this appearance field.
// Not a single geometry weight is touched.
// -------------------------------------------------------------------------

float3 render_albedo(int idx, int width, int height,
                     float3 eye, float3 fwd, float3 right, float3 up, float half_h,
                     Network geometry, AppearanceNetwork appearance,
                     float3 bg_color)
{
    int px = idx % width, py = idx / width;
    float aspect = float(width) / float(height);
    float u = (2.0f * (float(px) + 0.5f) / float(width) - 1.0f) * half_h * aspect;
    float v = (1.0f - 2.0f * (float(py) + 0.5f) / float(height)) * half_h;
    float3 dir = normalize(fwd + u * right + v * up);

    NeuralSdf sdf; sdf.network = geometry;
    float t;
    if (!sphere_trace(sdf, eye, dir, t)) return bg_color;
    float3 p = eye + t * dir;
    // Analytic torus normal, NOT normal_fd(sdf, p): The appearance section is about the
    // APPEARANCE field, and the frozen geometry is a known torus, so the
    // Lambert term takes the exact normal rather than the learned field's
    // own gradient. Shading with normal_fd here instead makes every ripple
    // in the (only approximately unit-gradient) learned SDF show up as
    // surface roughness, burying the thing the figure is actually about.
    float3 n = torus_normal(p);
    float shade = 0.2f + 0.8f * clamp(dot(n, LIGHT), 0.0f, 1.0f);
    return clamp(appearance.eval(p, float3(0, 0, 0)) * shade, 0.0f, 1.0f);
}

// Ground-truth albedo render, straight from `texture_albedo` -- no network
// at all, for Figure 6.5's comparison panel and its PSNR number.
float3 render_albedo_exact(int idx, int width, int height,
                           float3 eye, float3 fwd, float3 right, float3 up, float half_h,
                           Network geometry, float3 bg_color)
{
    int px = idx % width, py = idx / width;
    float aspect = float(width) / float(height);
    float u = (2.0f * (float(px) + 0.5f) / float(width) - 1.0f) * half_h * aspect;
    float v = (1.0f - 2.0f * (float(py) + 0.5f) / float(height)) * half_h;
    float3 dir = normalize(fwd + u * right + v * up);

    NeuralSdf sdf; sdf.network = geometry;
    float t;
    if (!sphere_trace(sdf, eye, dir, t)) return bg_color;
    float3 p = eye + t * dir;
    float3 n = torus_normal(p);        // analytic -- see render_albedo above
    float shade = 0.2f + 0.8f * clamp(dot(n, LIGHT), 0.0f, 1.0f);
    return clamp(texture_albedo(p) * shade, 0.0f, 1.0f);
}

// The radiance field already bakes shading in -- no separate Lambert term.
float3 render_radiance(int idx, int width, int height,
                       float3 eye, float3 fwd, float3 right, float3 up, float half_h,
                       Network geometry, AppearanceNetwork appearance,
                       float3 bg_color)
{
    int px = idx % width, py = idx / width;
    float aspect = float(width) / float(height);
    float u = (2.0f * (float(px) + 0.5f) / float(width) - 1.0f) * half_h * aspect;
    float v = (1.0f - 2.0f * (float(py) + 0.5f) / float(height)) * half_h;
    float3 dir = normalize(fwd + u * right + v * up);

    NeuralSdf sdf; sdf.network = geometry;
    float t;
    if (!sphere_trace(sdf, eye, dir, t)) return bg_color;
    float3 p = eye + t * dir;
    float3 view = -dir;                            // surface -> eye
    return clamp(appearance.eval(p, view), 0.0f, 1.0f);
}

// Ground-truth radiance render, straight from `radiance_gt` -- no network
// at all, for Figure 6.6's comparison rows and their PSNR numbers.
float3 render_radiance_exact(int idx, int width, int height,
                             float3 eye, float3 fwd, float3 right, float3 up, float half_h,
                             Network geometry, float3 bg_color)
{
    int px = idx % width, py = idx / width;
    float aspect = float(width) / float(height);
    float u = (2.0f * (float(px) + 0.5f) / float(width) - 1.0f) * half_h * aspect;
    float v = (1.0f - 2.0f * (float(py) + 0.5f) / float(height)) * half_h;
    float3 dir = normalize(fwd + u * right + v * up);

    NeuralSdf sdf; sdf.network = geometry;
    float t;
    if (!sphere_trace(sdf, eye, dir, t)) return bg_color;
    float3 p = eye + t * dir;
    float3 view = -dir;
    return radiance_gt(p, view);
}
