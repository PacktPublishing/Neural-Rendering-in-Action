// SPDX-License-Identifier: Apache-2.0
//
// THE network for Section 5 -- architecture, sampling, losses, and
// rendering hookups all live here; neural_sdf.py only drives it. There is
// no NumPy sampler and no NumPy backprop anywhere upstream of this file:
// every training kernel below draws its own batch (via common_rng.slang +
// common_shapes.slang) and backprops it in the same dispatch.
//
// Why `calculate_eikonal_grads` runs SEVEN threads per point (README.md has
// the measurements): the eikonal term needs seven network evaluations per
// point. Each thread redundantly repeats all seven forward passes, then
// backprops exactly one of the seven terms with a single bwd_diff. The
// alternatives give the same gradients but are slower or don't scale:
// seven bwd_diff calls in one thread run 2.5-4x slower (7x fewer threads
// to keep the GPU busy), and one bwd_diff over all seven evaluations is
// rejected by the GPU driver at hidden width 256.
//
// NetworkParameters and optimizer_step come from common_network.slang.
// HIDDEN defaults to 64 (the book uses 256) to keep eikonal training fast
// -- see README.md. Both L and HIDDEN are #defines, not Slang generics
// (see the note in neural_sdf.py's _module docstring for why: a generic
// Network<let L: int> did not survive bwd_diff on this toolchain).

import slangpy;
__exported import common_network;
__exported import common_shapes;
__exported import common_rng;
__exported import render;

#ifndef SDF_L
#define SDF_L 6
#endif
#ifndef SDF_HIDDEN
#define SDF_HIDDEN 64
#endif
static const int L = SDF_L;
static const int NUM_ENCODED = 3 + 6 * L;
static const int HIDDEN = SDF_HIDDEN;

struct Network
{
    NetworkParameters<NUM_ENCODED, HIDDEN> layer0;
    NetworkParameters<HIDDEN, HIDDEN> layer1;
    NetworkParameters<HIDDEN, 1> layer2;

    // Signed distance at a point: Fourier-encode, ReLU hidden layers,
    // LINEAR output (a distance can be any real number -- no final
    // activation, unlike Section 4's occupancy logit).
    [Differentiable]
    float eval(no_diff float3 p)
    {
        float[NUM_ENCODED] enc;
        enc[0] = p.x; enc[1] = p.y; enc[2] = p.z;
        [ForceUnroll]
        for (int k = 0; k < L; ++k)
        {
            float freq = (1 << k) * 3.14159265f;
            enc[3 + 6 * k + 0] = sin(freq * p.x);
            enc[3 + 6 * k + 1] = sin(freq * p.y);
            enc[3 + 6 * k + 2] = sin(freq * p.z);
            enc[3 + 6 * k + 3] = cos(freq * p.x);
            enc[3 + 6 * k + 4] = cos(freq * p.y);
            enc[3 + 6 * k + 5] = cos(freq * p.z);
        }
        float[HIDDEN] h0 = layer0.forward(enc);
        [ForceUnroll]
        for (int i = 0; i < HIDDEN; ++i) h0[i] = max(h0[i], 0.0f);
        float[HIDDEN] h1 = layer1.forward(h0);
        [ForceUnroll]
        for (int i = 0; i < HIDDEN; ++i) h1[i] = max(h1[i], 0.0f);
        float[1] out = layer2.forward(h1);
        return out[0];
    }
}

[Differentiable]
float eval_wrap(no_diff float3 p, Network network)
{
    return network.eval(p);
}

float predict(float3 p, Network network) { return network.eval(p); }

float predict_batch(int idx, Tensor<float3, 1> pts, Network network)
{
    return predict(pts.load(idx), network);
}

// -------------------------------------------------------------------------
// 5.2 -- supervised regression: exact smooth-torus distances, clamped-L1.
// Sampling (half uniform in the box, half near the surface + jitter),
// labelling (`torus_sdf`), and the clamped-L1 + far-field loss are all
// expressed as ordinary differentiable Slang arithmetic -- `clamp`'s
// zero-derivative outside its band IS the "clamp kills gradient" mask the
// book computes by hand in NumPy; autodiff gets it for free.
// -------------------------------------------------------------------------

[Differentiable]
float clamped_l1_loss(no_diff float3 p, no_diff float gt, no_diff float delta,
                      no_diff float beta_far, Network network)
{
    float pred = network.eval(p);
    float cp = clamp(pred, -delta, delta);
    float cg = clamp(gt, -delta, delta);
    return abs(cp - cg) + beta_far * abs(pred - gt);
}

void calculate_supervised_grads(int idx, uint seed, int n_uniform, float jitter,
                                float delta, float beta_far, Network network)
{
    Rng rng = Rng(seed, uint(idx));
    float3 p;
    if (idx < n_uniform)
    {
        p = rng.next_float3(-BOUND, BOUND);
    }
    else
    {
        float u = rng.next_float(0.0f, 6.28318530718f);
        float v = rng.next_float(0.0f, 6.28318530718f);
        p = torus_surface(float2(u, v)) + jitter * rng.next_gaussian3();
    }
    float gt = torus_sdf(p);
    bwd_diff(clamped_l1_loss)(p, gt, delta, beta_far, network, 1.0f);
}

// -------------------------------------------------------------------------
// Optional: Stanford bunny, supervised-SDF variant (neural_sdf.py's
// train_supervised_mesh, driven by train_bunny_sdf.py). An arbitrary mesh
// has no torus_sdf()-style closed form, so the label instead comes from a
// precomputed signed-distance grid (mesh_shape.build_or_load_sdf_grid,
// chapter root), trilinearly interpolated by grid_trilerp
// (common_shapes.slang) -- and near-surface samples come from a
// precomputed surface cloud (mesh_shape.load_and_cache) instead of
// torus_surface(). Same clamped_l1_loss as the torus path.
// -------------------------------------------------------------------------
void calculate_supervised_grads_mesh(int idx, uint seed, int n_uniform, float jitter,
                                     Tensor<float, 1> grid, float3 origin, float3 spacing, int res,
                                     Tensor<float3, 1> cloud, int n_cloud,
                                     float delta, float beta_far, Network network)
{
    Rng rng = Rng(seed, uint(idx));
    float3 p;
    if (idx < n_uniform)
    {
        p = rng.next_float3(-BOUND, BOUND);
    }
    else
    {
        int cidx = int(rng.next_float(0.0f, float(n_cloud)));
        p = cloud.load(cidx) + jitter * rng.next_gaussian3();
    }
    float gt = grid_trilerp(grid, origin, spacing, res, p);
    bwd_diff(clamped_l1_loss)(p, gt, delta, beta_far, network, 1.0f);
}

// -------------------------------------------------------------------------
// 5.5 -- an SDF from a surface point cloud alone (IGR-style): a surface
// term (sticks to zero), an eikonal term (be a distance field), a
// repulsion term (don't collapse to zero everywhere), plus a geometric
// warm-up toward a sphere SDF.
//
// Every term is a MEAN over its batch (dOut = 1/N), as in the original
// recipe. The surface, eikonal and repulsion gradients are summed into one
// Adam step, so their relative scale IS their relative weight: a summed
// surface term outweighs the other two ~1000x and the field collapses to
// f = 0 everywhere. The warm-up shares the same Adam moment buffers, so it
// has to be on the same scale as well.
// -------------------------------------------------------------------------

// Generate the fixed "laser scan" once: exact surface samples of the
// KNURLED torus (Section 4's occupancy target, same shape, no closed-form
// SDF -- that gap is this recipe's whole point).
float3 sample_surface_cloud(int idx, uint seed)
{
    Rng rng = Rng(seed, uint(idx));
    float theta = rng.next_float(0.0f, 6.28318530718f);
    float phi = rng.next_float(0.0f, 6.28318530718f);
    return detailed_torus_surface(float2(theta, phi));
}

[Differentiable]
float sphere_warmup_loss(no_diff float3 p, no_diff float r0, Network network)
{
    float pred = network.eval(p);
    return abs(pred - (length(p) - r0));
}

// Geometric init: 200 cheap supervised steps toward ||x|| - r0 give the
// field one clean inside region to deform, before the eikonal loss (which
// can otherwise lock in spurious interior pockets from a random init).
void calculate_warmup_grads(int idx, uint seed, int n_warmup, Network network)
{
    Rng rng = Rng(seed, uint(idx));
    float3 p = rng.next_float3(-BOUND, BOUND);
    bwd_diff(sphere_warmup_loss)(p, 0.75f, network, 1.0f / float(n_warmup));
}

[Differentiable]
float surface_loss(no_diff float3 p, Network network)
{
    return abs(network.eval(p));
}

// The surface term: sample the point cloud, jitter slightly, push f -> 0.
void calculate_surface_grads(int idx, uint seed, Tensor<float3, 1> cloud,
                             int n_cloud, float jitter, int n_surface, Network network)
{
    Rng rng = Rng(seed, uint(idx));
    int cidx = int(rng.next_float(0.0f, float(n_cloud)));
    float3 p = cloud.load(cidx) + jitter * rng.next_gaussian3();
    bwd_diff(surface_loss)(p, network, 1.0f / float(n_surface));
}

// The eikonal + repulsion term over "space" points (half uniform in the
// box, half near the cloud with bigger jitter). Seven threads per logical
// point (see the note at the top of this file): each
// redundantly computes the same six-shift finite-difference gradient and
// the repulsion value, then backprops exactly one of the seven terms
// (thread 0: repulsion at the unshifted point; threads 1-6: the eikonal
// coefficient at one shifted point).
void calculate_eikonal_grads(int idx7, uint seed, Tensor<float3, 1> cloud,
                             int n_cloud, int n_space, float lam_eik,
                             float lam_rep, float alpha, float fd_h,
                             Network network)
{
    int j = idx7 / 7;
    int s = idx7 % 7;

    Rng rng = Rng(seed, uint(j));
    float3 p;
    if (j % 2 == 0)
    {
        p = rng.next_float3(-BOUND, BOUND);
    }
    else
    {
        int cidx = int(rng.next_float(0.0f, float(n_cloud)));
        p = cloud.load(cidx) + 0.1f * rng.next_gaussian3();
    }

    float f0  = network.eval(p);
    float fxp = network.eval(p + float3(fd_h, 0, 0));
    float fxm = network.eval(p - float3(fd_h, 0, 0));
    float fyp = network.eval(p + float3(0, fd_h, 0));
    float fym = network.eval(p - float3(0, fd_h, 0));
    float fzp = network.eval(p + float3(0, 0, fd_h));
    float fzm = network.eval(p - float3(0, 0, fd_h));

    float3 g = float3(fxp - fxm, fyp - fym, fzp - fzm) / (2.0f * fd_h);
    float gnorm = length(g) + 1e-9f;
    float res = gnorm - 1.0f;
    float coef = lam_eik * 2.0f * res / gnorm / float(n_space) / (2.0f * fd_h);

    float3 target_p = p;
    float dpred = 0.0f;
    if (s == 0)
    {
        // repulsion, at the unshifted point
        dpred = lam_rep * (-alpha) * sign(f0) * exp(-alpha * abs(f0)) / float(n_space);
    }
    else
    {
        int axis = (s - 1) % 3;
        float sgn = (s <= 3) ? 1.0f : -1.0f;
        float3 shift = float3(0, 0, 0);
        shift[axis] = fd_h * sgn;
        target_p = p + shift;
        dpred = sgn * coef * g[axis];
    }
    bwd_diff(eval_wrap)(target_p, network, dpred);
}

// -------------------------------------------------------------------------
// Rendering hookups [5.4 / 5.3 / 5.6] -- render.slang's generic pipeline,
// specialized to this trained network. No weight is ever downloaded for
// this: march, normal, and shade all run on the GPU per pixel.
// -------------------------------------------------------------------------

struct NeuralSdf : Sdf
{
    Network network;
    float distance(float3 p) { return network.eval(p); }
}

float3 render_neural(int idx, int width, int height,
                     float3 eye, float3 fwd, float3 right, float3 up,
                     float half_h, float3 light, float3 base_color,
                     float3 bg_color, Network network)
{
    int px = idx % width, py = idx / width;
    NeuralSdf sdf; sdf.network = network;
    return render_pixel(sdf, px, py, width, height, eye, fwd, right, up,
                        half_h, light, base_color, bg_color);
}

// Normals visualized as RGB (0.5*n + 0.5), Part 5.3's figure.
float3 render_neural_normals(int idx, int width, int height,
                             float3 eye, float3 fwd, float3 right, float3 up,
                             float half_h, Network network)
{
    int px = idx % width, py = idx / width;
    float aspect = float(width) / float(height);
    float u = (2.0f * (float(px) + 0.5f) / float(width) - 1.0f) * half_h * aspect;
    float v = (1.0f - 2.0f * (float(py) + 0.5f) / float(height)) * half_h;
    float3 dir = normalize(fwd + u * right + v * up);

    NeuralSdf sdf; sdf.network = network;
    float t;
    if (!sphere_trace(sdf, eye, dir, t)) return float3(1, 1, 1);
    float3 p = eye + t * dir;
    return 0.5f * normal_fd(sdf, p) + 0.5f;
}

// CSG [5.6]: smooth_union(trained_field, analytic_sphere), sphere-traced
// directly -- not a single weight is touched.
struct UnionSdf : Sdf
{
    Network network;
    float k;
    float3 sphere_center;
    float sphere_r;
    float distance(float3 p)
    {
        return smooth_union(network.eval(p), sphere_sdf(p, sphere_center, sphere_r), k);
    }
}

float3 render_smooth_union(int idx, int width, int height,
                           float3 eye, float3 fwd, float3 right, float3 up,
                           float half_h, float3 light, float3 base_color,
                           float3 bg_color, Network network, float k,
                           float3 sphere_center, float sphere_r)
{
    int px = idx % width, py = idx / width;
    UnionSdf sdf;
    sdf.network = network; sdf.k = k;
    sdf.sphere_center = sphere_center; sdf.sphere_r = sphere_r;
    return render_pixel(sdf, px, py, width, height, eye, fwd, right, up,
                        half_h, light, base_color, bg_color);
}
