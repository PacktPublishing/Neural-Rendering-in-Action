// SPDX-License-Identifier: Apache-2.0
//
// Figure 6.3, the race: a pure coordinate MLP vs a FeatureGrid + tiny-MLP
// hybrid, both trained on the knurled-torus occupancy task -- entirely on
// the GPU. Sampling (half uniform in the box, half near-surface + jitter)
// and labelling (`detailed_torus_inside`) happen inside
// `pure_calculate_grads`/`hybrid_calculate_grads` themselves, via
// `common_shapes.slang` + `common_rng.slang`; `race_gpu.py` only measures
// wall-clock time per dispatch, never samples or labels a batch itself.
//
// NetworkParameters and optimizer_step come from common_network.slang
// (shared by every *_train.slang module in this chapter) for the pure-MLP
// side and the hybrid's tiny decoder head. The FeatureGrid itself is new: a
// dense grid of learnable feature vectors, read through a [Differentiable]
// trilinear accessor whose backward derivative atomically scatters into an
// 8-corner neighbourhood of an AtomicTensor -- the same way
// NetworkParameters.forward_bwd accumulates weight gradients, just indexed
// by a grid cell instead of a (neuron, input) pair.

import slangpy;
__exported import common_network;
__exported import common_shapes;
__exported import common_rng;

#ifndef APP_L
#define APP_L 6
#endif
static const int L = APP_L;
static const int NUM_ENCODED = 3 + 6 * L;
static const int HIDDEN = 64;          // pure-MLP side (book uses 256; see README)

static const int GRID_RES = 32;
static const int GRID_FEAT = 4;
static const int HEAD_HIDDEN = 32;     // hybrid decoder (book uses 32 too)

// The same sampling `pure_calculate_grads`/`hybrid_calculate_grads` do
// internally, exposed standalone for the accuracy-curve readout (which
// needs actual points to run inference on, not just a gradient).
float3 sample_occ_point(int idx, uint seed, int n_uniform, float jitter)
{
    Rng rng = Rng(seed, uint(idx));
    if (idx < n_uniform) return rng.next_float3(-BOUND, BOUND);
    float u = rng.next_float(0.0f, 6.28318530718f);
    float v = rng.next_float(0.0f, 6.28318530718f);
    return detailed_torus_surface(float2(u, v)) + jitter * rng.next_gaussian3();
}

// -- pure-MLP racer: identical shape to neural_occupancy.slang's Network ---
struct PureNetwork
{
    NetworkParameters<NUM_ENCODED, HIDDEN> layer0;
    NetworkParameters<HIDDEN, HIDDEN> layer1;
    NetworkParameters<HIDDEN, 1> layer2;

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
float pure_bce_loss(no_diff float3 p, no_diff float y, PureNetwork network)
{
    float logit = network.eval(p);
    return max(logit, 0.0f) - logit * y + log(1.0f + exp(-abs(logit)));
}

// Sampling (half uniform in the box, half near the knurled-torus surface +
// jitter) and labelling (`detailed_torus_inside`) happen right here, on the
// GPU -- the race never uploads a batch from NumPy.
void pure_calculate_grads(int idx, uint seed, int n_uniform, float jitter, PureNetwork network)
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
        p = detailed_torus_surface(float2(u, v)) + jitter * rng.next_gaussian3();
    }
    float y = detailed_torus_inside(p) ? 1.0f : 0.0f;
    bwd_diff(pure_bce_loss)(p, y, network, 1.0f);
}

float pure_predict_batch(int idx, Tensor<float3, 1> pts, PureNetwork network)
{
    float logit = network.eval(pts.load(idx));
    return 1.0f / (1.0f + exp(-clamp(logit, -30.0f, 30.0f)));
}

// -- hybrid racer: FeatureGrid -> concat(feature, xyz) -> tiny MLP --------
struct FeatureGrid
{
    // Flattened (res, res, res, feat) grid, row-major, feat fastest-varying
    // -- matches app6.FeatureGrid.F's layout so a NumPy-side comparison, if
    // ever wanted, reads the same convention.
    Tensor<float, 1> cells;
    AtomicTensor<float, 1> cells_grad;

    [Differentiable]
    float get_cell(int ix, int iy, int iz, int c)
    {
        int idx = ((ix * GRID_RES + iy) * GRID_RES + iz) * GRID_FEAT + c;
        return cells.load(idx);
    }
    [BackwardDerivativeOf(get_cell)]
    void get_cell_bwd(int ix, int iy, int iz, int c, float grad)
    {
        int idx = ((ix * GRID_RES + iy) * GRID_RES + iz) * GRID_FEAT + c;
        cells_grad.add(idx, grad);
    }

    // Trilinear lookup, an ordinary differentiable function built out of the
    // accessor above -- bwd_diff scatters into the 8 touched corners
    // automatically, no manual np.add.at bookkeeping needed.
    [Differentiable]
    float[GRID_FEAT] lookup(no_diff float3 p, no_diff float bound)
    {
        no_diff float3 g = (p + bound) / (2.0f * bound) * float(GRID_RES - 1);
        g = clamp(g, float3(0), float3(GRID_RES - 1 - 1e-4f));
        no_diff int3 i0 = int3(g);
        no_diff float3 t = g - float3(i0);

        float[GRID_FEAT] out;
        [ForceUnroll]
        for (int c = 0; c < GRID_FEAT; ++c) out[c] = 0.0f;

        [ForceUnroll]
        for (int dx = 0; dx < 2; ++dx)
        [ForceUnroll]
        for (int dy = 0; dy < 2; ++dy)
        [ForceUnroll]
        for (int dz = 0; dz < 2; ++dz)
        {
            no_diff float w = abs(1 - dx - t.x) * abs(1 - dy - t.y) * abs(1 - dz - t.z);
            [ForceUnroll]
            for (int c = 0; c < GRID_FEAT; ++c)
                out[c] += w * get_cell(i0.x + dx, i0.y + dy, i0.z + dz, c);
        }
        return out;
    }
}

struct HybridNetwork
{
    FeatureGrid grid;
    NetworkParameters<GRID_FEAT + 3, HEAD_HIDDEN> head0;
    NetworkParameters<HEAD_HIDDEN, HEAD_HIDDEN> head1;
    NetworkParameters<HEAD_HIDDEN, 1> head2;

    [Differentiable]
    float eval(no_diff float3 p, no_diff float bound)
    {
        float[GRID_FEAT] f = grid.lookup(p, bound);
        float[GRID_FEAT + 3] x;
        [ForceUnroll]
        for (int c = 0; c < GRID_FEAT; ++c) x[c] = f[c];
        x[GRID_FEAT + 0] = p.x; x[GRID_FEAT + 1] = p.y; x[GRID_FEAT + 2] = p.z;

        float[HEAD_HIDDEN] h0 = head0.forward(x);
        [ForceUnroll]
        for (int i = 0; i < HEAD_HIDDEN; ++i) h0[i] = max(h0[i], 0.0f);
        float[HEAD_HIDDEN] h1 = head1.forward(h0);
        [ForceUnroll]
        for (int i = 0; i < HEAD_HIDDEN; ++i) h1[i] = max(h1[i], 0.0f);
        float[1] out = head2.forward(h1);
        return out[0];
    }
}

[Differentiable]
float hybrid_bce_loss(no_diff float3 p, no_diff float y, no_diff float bound, HybridNetwork network)
{
    float logit = network.eval(p, bound);
    return max(logit, 0.0f) - logit * y + log(1.0f + exp(-abs(logit)));
}

// Same fused sampling as pure_calculate_grads, for the hybrid racer.
void hybrid_calculate_grads(int idx, uint seed, int n_uniform, float jitter,
                            HybridNetwork network, float bound)
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
        p = detailed_torus_surface(float2(u, v)) + jitter * rng.next_gaussian3();
    }
    float y = detailed_torus_inside(p) ? 1.0f : 0.0f;
    bwd_diff(hybrid_bce_loss)(p, y, bound, network, 1.0f);
}

float hybrid_predict_batch(int idx, Tensor<float3, 1> pts, HybridNetwork network, float bound)
{
    float logit = network.eval(pts.load(idx), bound);
    return 1.0f / (1.0f + exp(-clamp(logit, -30.0f, 30.0f)));
}
