// SPDX-License-Identifier: Apache-2.0
//
// GPU/SlangPy port of the MLP + manual backprop + Adam optimizer from
// neural_occupancy.py. The geometry, sampling, marching-cubes extraction,
// and plotting all stay in Python/NumPy/scikit-image/matplotlib -- only the
// differentiable network and its training step move to Slang, following the
// same NetworkParameters pattern used in Chapter 4.
//
// L (the positional-encoding order) is a preprocessor define, not a Slang
// generic parameter -- generic structs did not survive bwd_diff in testing
// (the compiler either hung or crashed differentiating through a struct
// whose field types depend on a generic int parameter). Plain structs sized
// from a #define compile and train correctly, so Python passes L in via the
// `defines` compiler option and reloads the module if L changes.

import slangpy;

#ifndef OCC_L
#define OCC_L 6
#endif
static const int L = OCC_L;
static const int NUM_ENCODED = 3 + 6 * L;
// The book's NumPy version uses 256-wide hidden layers. Differentiating a
// hand-written per-element weight loop at that width did not finish
// compiling in this Slang toolchain even after 15+ minutes (32-wide: ~1.5s,
// 64-wide: ~9-25s depending on loop strategy -- clearly superlinear, and
// 256-wide falls off that curve entirely). 64 is the widest that compiles
// reliably in single-digit seconds; going wider would need the network
// rewritten around Slang's native matrix<T,M,N> ops instead of a scalar
// per-element loop, which is a larger change than this migration covers.
static const int HIDDEN = 64;

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;

    // The network is shared across every sampled point in the batch, so
    // gradients from all of them accumulate into the same weights -- hence
    // the atomic tensor, exactly as in Chapter 4.
    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float get_bias(int neuron)
    {
        return biases.load(neuron);
    }
    [Differentiable]
    float get_weight(int neuron, int input)
    {
        return weights.load(neuron, input);
    }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float grad)
    {
        biases_grad.add(neuron, grad);
    }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float grad)
    {
        weights_grad.add(neuron, input, grad);
    }

    [Differentiable]
    float[Outputs] forward(float[Inputs] x)
    {
        float[Outputs] y;
        // MaxIters on both loops: Chapter 4 could unroll (ForceUnroll) its inner
        // loop at widths up to 32, but differentiating a fully unrolled loop
        // scales very badly with width -- MaxIters keeps compile time in check
        // at the cost of a slightly slower kernel. See HIDDEN's comment above.
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
    NetworkParameters<HIDDEN, 1> layer2;

    // Occupancy logit at a 3D point: Fourier-encode, then ReLU hidden
    // layers, then a linear output -- same shape as the book's MLP class,
    // just evaluated on the GPU. L = 0 reproduces the raw-coordinate,
    // spectral-bias failure case from the book.
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
            enc[3 + 6 * k + 1] = cos(freq * p.x);
            enc[3 + 6 * k + 2] = sin(freq * p.y);
            enc[3 + 6 * k + 3] = cos(freq * p.y);
            enc[3 + 6 * k + 4] = sin(freq * p.z);
            enc[3 + 6 * k + 5] = cos(freq * p.z);
        }
        float[HIDDEN] h0 = layer0.forward(enc);
        [MaxIters(HIDDEN)]
        for (int i = 0; i < HIDDEN; ++i)
            h0[i] = max(h0[i], 0.0f);
        float[HIDDEN] h1 = layer1.forward(h0);
        [MaxIters(HIDDEN)]
        for (int i = 0; i < HIDDEN; ++i)
            h1[i] = max(h1[i], 0.0f);
        float[1] out = layer2.forward(h1);
        return out[0];
    }
}

// Stable BCE-with-logits, differentiable end to end through the network.
// y is the 0/1 inside/outside label; not itself a trained parameter.
[Differentiable]
float bce_loss(no_diff float3 p, no_diff float y, Network network)
{
    float logit = network.eval(p);
    return max(logit, 0.0f) - logit * y + log(1.0f + exp(-abs(logit)));
}

// Occupancy probability at a point, for grid evaluation / marching cubes.
// Not [Differentiable]: inference only, not part of the training graph.
float predict(float3 p, Network network)
{
    float logit = network.eval(p);
    return 1.0f / (1.0f + exp(-clamp(logit, -30.0f, 30.0f)));
}

// One training step for point idx: read its position/label, backprop the
// BCE loss into the network's (atomic) weight and bias gradients.
void calculate_grads(int idx, Tensor<float3, 1> pts, Tensor<float, 1> labels, Network network)
{
    float3 p = pts.load(idx);
    float y = labels.load(idx);
    bwd_diff(bce_loss)(p, y, network, 1.0f);
}

// Batched inference for grid_eval(): one call per output element.
float predict_batch(int idx, Tensor<float3, 1> pts, Network network)
{
    return predict(pts.load(idx), network);
}

// Adam step, identical to Chapter 4's optimizer_step.
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
