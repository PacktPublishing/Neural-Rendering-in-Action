// SPDX-License-Identifier: Apache-2.0
//
// GPU/SlangPy training for Section 6's appearance MLPs (6.2 albedo c(x),
// 6.3 radiance c(x, v)) -- same pattern as Section 4/5. HAS_VIEW selects
// which of the two networks this compiled module trains; the Python driver
// loads two separately-specialized modules, one per network.
//
// Section 6.4's FeatureGrid/HybridField "race" (Figure 6.3) is NOT ported:
// it is a deliberate wall-clock CPU speed comparison between two NumPy
// implementations (make_figures6.py's own docstring: "its numbers are
// machine-dependent by nature"), not a training-quality demo, so moving it
// to the GPU would defeat its point.
//
// See Section4/README.md and Section5/README_slangpy.md for the compile-time
// workarounds this reuses (L as a #define, MaxIters over ForceUnroll on wide
// loops, a kernel warm-up before training, persistent GPU buffers). As in
// Section 5, the positional-encoding column order matches encode() in app6.py
// exactly -- per octave [sin(x),sin(y),sin(z), cos(x),cos(y),cos(z)] -- since
// trained weights get saved into a checkpoint that app6.MLP.forward() reads
// back with its own encoding.

import slangpy;

#ifndef APP_L
#define APP_L 6
#endif
#ifndef HAS_VIEW
#define HAS_VIEW 0
#endif
static const int L = APP_L;
static const int NUM_ENCODED = 3 + 6 * L + (HAS_VIEW != 0 ? 3 : 0);
static const int HIDDEN = 64;

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;
    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float get_bias(int neuron) { return biases.load(neuron); }
    [Differentiable]
    float get_weight(int neuron, int input) { return weights.load(neuron, input); }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float grad) { biases_grad.add(neuron, grad); }
    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float grad) { weights_grad.add(neuron, input, grad); }

    [Differentiable]
    float[Outputs] forward(float[Inputs] x)
    {
        float[Outputs] y;
        [MaxIters(Outputs)]
        for (int row = 0; row < Outputs; ++row)
        {
            var sum = get_bias(row);
            [MaxIters(Inputs)]
            for (int col = 0; col < Inputs; ++col)
                sum += get_weight(row, col) * x[col];
            y[row] = sum;
        }
        return y;
    }
}

struct Network
{
    NetworkParameters<NUM_ENCODED, HIDDEN> layer0;
    NetworkParameters<HIDDEN, HIDDEN> layer1;
    NetworkParameters<HIDDEN, 3> layer2;

    [Differentiable]
    float3 eval(no_diff float3 p, no_diff float3 v)
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
#if HAS_VIEW
        enc[3 + 6 * L + 0] = v.x;
        enc[3 + 6 * L + 1] = v.y;
        enc[3 + 6 * L + 2] = v.z;
#endif
        float[HIDDEN] h0 = layer0.forward(enc);
        [MaxIters(HIDDEN)]
        for (int i = 0; i < HIDDEN; ++i)
            h0[i] = max(h0[i], 0.0f);
        float[HIDDEN] h1 = layer1.forward(h0);
        [MaxIters(HIDDEN)]
        for (int i = 0; i < HIDDEN; ++i)
            h1[i] = max(h1[i], 0.0f);
        float[3] out = layer2.forward(h1);
        return float3(out[0], out[1], out[2]);
    }
}

[Differentiable]
float3 eval_wrap(no_diff float3 p, no_diff float3 v, Network network)
{
    return network.eval(p, v);
}

float3 predict(float3 p, float3 v, Network network)
{
    return network.eval(p, v);
}

float3 predict_batch(int idx, Tensor<float3, 1> pts, Tensor<float3, 1> views, Network network)
{
    return predict(pts.load(idx), views.load(idx), network);
}

// Backprop an externally supplied per-point RGB output gradient (computed on
// the CPU from plain MSE: 2*(pred-gt)/N, exactly as train_albedo_field /
// train_radiance_field's opt.step(*net.backward(2*(pred-gt)/pred.size)) does)
// into the network's weight/bias gradients.
void calculate_grads_from_dpred(int idx, Tensor<float3, 1> pts, Tensor<float3, 1> views,
                                Tensor<float3, 1> dpred, Network network)
{
    float3 p = pts.load(idx);
    float3 v = views.load(idx);
    bwd_diff(eval_wrap)(p, v, network, dpred.load(idx));
}

void optimizer_step(inout float primal, inout float grad, inout float mean,
                    inout float variance, float learning_rate, int iteration)
{
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    float mHat = mean / (1.0f - pow(ADAM_BETA_1, iteration));
    float vHat = variance / (1.0f - pow(ADAM_BETA_2, iteration));
    float step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);

    primal -= step;
    grad = 0.0f;
}
