// SPDX-License-Identifier: Apache-2.0
//
// neural_occupancy.slang -- the neural occupancy field
// ========================================================================
// THE network for the occupancy section. There is exactly one, and this is it: the
// architecture, the positional encoding, and the loss all live here, and
// neural_occupancy.py drives it. Nothing else in this section defines a network.
//
//     f_theta(x, y, z)  ->  logit,  sigmoid(logit) = P(inside)
//
// NetworkParameters (one differentiable dense layer) and optimizer_step
// (Adam) come from common_network.slang, shared with the rest of the
// chapter.
//
// Two compile-time constants specialize the network. neural_occupancy.py supplies
// both by prepending #define lines to this source before compiling it --
// NOT via SlangPy's `options={"defines": ...}`, which does not reach the
// Slang preprocessor (Module.load_from_file passes the file straight to
// device.load_module and keeps `options` for call configuration only).
//
//   OCC_L       positional-encoding bands. 0 = raw coordinates, the
//               spectral-bias failure case of Part 4.
//   OCC_HIDDEN  width of each hidden layer.

import slangpy;
__exported import common_network;
__exported import common_shapes;
__exported import common_rng;

#ifndef OCC_L
#define OCC_L 10
#endif
#ifndef OCC_HIDDEN
#define OCC_HIDDEN 256
#endif

static const int L = OCC_L;
static const int HIDDEN = OCC_HIDDEN;
static const int NUM_ENCODED = 3 + 6 * L;

struct Network
{
    NetworkParameters<NUM_ENCODED, HIDDEN> layer0;
    NetworkParameters<HIDDEN, HIDDEN> layer1;
    NetworkParameters<HIDDEN, HIDDEN> layer2;
    NetworkParameters<HIDDEN, 1> layer3;

    // gamma(p) -> [ReLU hidden] x3 -> logit.
    //
    // The encoding writes each octave as three sines then three cosines,
    // [sin x, sin y, sin z, cos x, cos y, cos z]. That is the column order
    // neural_occupancy.positional_encoding() produces (it concatenates whole (N,3)
    // blocks) and the order every other section of this chapter uses. The
    // checkpoints on disk are read back by that NumPy encoder for the
    // Part 2 figure, so the two must agree column for column.
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
        float[HIDDEN] h2 = layer2.forward(h1);
        [ForceUnroll]
        for (int i = 0; i < HIDDEN; ++i) h2[i] = max(h2[i], 0.0f);
        float[1] out = layer3.forward(h2);
        return out[0];
    }
}

// Numerically stable binary cross-entropy on the logit. y is the 0/1
// inside/outside label, not a trained parameter.
[Differentiable]
float bce_loss(no_diff float3 p, no_diff float y, Network network)
{
    float logit = network.eval(p);
    return max(logit, 0.0f) - logit * y + log(1.0f + exp(-abs(logit)));
}

// One training point, start to finish: sample it (uniform in the box, or
// near the surface + jitter, chosen by idx < n_uniform), label it with the
// exact inside test, and backprop -- no pts/labels tensor uploaded from
// Python at all. common_rng.slang's Rng and common_shapes.slang's
// detailed_torus_inside/detailed_torus_surface do the sampling; one
// optimizer_step per layer then consumes the accumulated gradients (see
// neural_occupancy.train).
void calculate_grads(int idx, uint seed, int n_uniform, float jitter, Network network)
{
    Rng rng = Rng(seed, uint(idx));
    float3 p;
    if (idx < n_uniform)
    {
        p = rng.next_float3(-BOUND, BOUND);
    }
    else
    {
        float theta = rng.next_float(0.0f, 6.28318530718f);
        float phi = rng.next_float(0.0f, 6.28318530718f);
        p = detailed_torus_surface(float2(theta, phi)) + jitter * rng.next_gaussian3();
    }
    float y = detailed_torus_inside(p) ? 1.0f : 0.0f;
    bwd_diff(bce_loss)(p, y, network, 1.0f);
}

// Same sampling as calculate_grads, standalone and forward-only -- for
// Part 1's labelled-point scatter figure, which needs the points AND
// labels back in Python rather than a gradient.
float3 sample_point(int idx, uint seed, int n_uniform, float jitter)
{
    Rng rng = Rng(seed, uint(idx));
    if (idx < n_uniform) return rng.next_float3(-BOUND, BOUND);
    float theta = rng.next_float(0.0f, 6.28318530718f);
    float phi = rng.next_float(0.0f, 6.28318530718f);
    return detailed_torus_surface(float2(theta, phi)) + jitter * rng.next_gaussian3();
}

// Occupancy probability, for grid evaluation and marching cubes. Inference
// only, so deliberately not [Differentiable].
float predict_batch(int idx, Tensor<float3, 1> pts, Network network)
{
    float logit = network.eval(pts.load(idx));
    return 1.0f / (1.0f + exp(-clamp(logit, -30.0f, 30.0f)));
}

// -------------------------------------------------------------------------
// Optional: Stanford bunny mesh support (neural_occupancy.py's train_mesh,
// driven by train_bunny.py). Unlike the rippled torus above, an arbitrary
// mesh has no closed-form inside test or surface parametrization, so both
// come from data precomputed ONCE on the CPU (mesh_shape.py, via trimesh)
// and uploaded as GPU buffers: a dense occupancy grid (trilinearly
// interpolated by grid_trilerp, common_shapes.slang) and a cloud of exact
// surface points (indexed the same way neural_sdf.slang's
// calculate_surface_grads indexes its cloud). Same Network, same
// bce_loss -- only the sampling/labeling differs from calculate_grads.
// -------------------------------------------------------------------------

void calculate_grads_mesh(int idx, uint seed, int n_uniform, float jitter,
                          Tensor<float, 1> grid, float3 origin, float3 spacing, int res,
                          Tensor<float3, 1> cloud, int n_cloud, Network network)
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
    float y = grid_trilerp(grid, origin, spacing, res, p) > 0.5f ? 1.0f : 0.0f;
    bwd_diff(bce_loss)(p, y, network, 1.0f);
}

// Same sampling as calculate_grads_mesh, standalone and forward-only -- for
// train_mesh()'s reporting-only loss/accuracy readback (mirrors sample_point).
float3 sample_point_mesh(int idx, uint seed, int n_uniform, float jitter,
                         Tensor<float3, 1> cloud, int n_cloud)
{
    Rng rng = Rng(seed, uint(idx));
    if (idx < n_uniform) return rng.next_float3(-BOUND, BOUND);
    int cidx = int(rng.next_float(0.0f, float(n_cloud)));
    return cloud.load(cidx) + jitter * rng.next_gaussian3();
}

// Grid-based inside test at an arbitrary batch of points -- the mesh
// equivalent of detailed_torus_inside_batch (common_shapes.slang), used by
// train_mesh()'s reporting-only readback.
float mesh_inside_batch(int idx, Tensor<float3, 1> pts, Tensor<float, 1> grid,
                        float3 origin, float3 spacing, int res)
{
    return grid_trilerp(grid, origin, spacing, res, pts.load(idx)) > 0.5f ? 1.0f : 0.0f;
}
