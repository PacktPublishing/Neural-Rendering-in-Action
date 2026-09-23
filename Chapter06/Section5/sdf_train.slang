// SPDX-License-Identifier: Apache-2.0
//
// GPU/SlangPy training for Section 5's SDFNet (5.2, the supervised-regression
// recipe). Same NetworkParameters pattern as Chapter 4 and Section 4; see
// Section4/neural_occupancy.slang for the compile-time workarounds this
// reuses (L as a #define rather than a generic, MaxIters instead of
// ForceUnroll on wide loops, and a kernel warm-up before training).
//
// clamped_l1's gradient (sign() and a hard <delta threshold) isn't something
// worth expressing through Slang's autodiff -- it's simpler and just as
// correct to compute d(loss)/d(pred) on the CPU in NumPy, exactly as the
// book's clamped_l1() already does, and hand that per-point gradient to
// bwd_diff() here to backprop into the weights. So this module exposes three
// entry points: a forward-only predict_batch (to get predictions back to
// NumPy), calculate_grads_from_dpred (backprop an externally supplied
// per-point output gradient into the network), and optimizer_step (Adam,
// unchanged from Chapter 4/Section 4).
//
// Sphere tracing, normals (input_gradient), CSG, and marching cubes are
// NOT ported here: make_figures5.py already does all of that in plain NumPy
// from a saved .npz checkpoint, so this module only needs to produce a
// checkpoint in that same format -- it never needs to run at render time.

import slangpy;

#ifndef SDF_L
#define SDF_L 6
#endif
static const int L = SDF_L;
static const int NUM_ENCODED = 3 + 6 * L;
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
    NetworkParameters<HIDDEN, 1> layer2;

    // Signed distance at a 3D point: Fourier-encode, ReLU hidden layers,
    // LINEAR output (unlike Section 4's occupancy logit, a distance can be
    // any real number, so there is no final activation).
    [Differentiable]
    float eval(no_diff float3 p)
    {
        float[NUM_ENCODED] enc;
        enc[0] = p.x; enc[1] = p.y; enc[2] = p.z;
        // Column order must match sdf5.SDFNet.encode() exactly, since trained
        // weights get saved into a checkpoint that SDFNet.forward() reads
        // back: per octave, [sin(x),sin(y),sin(z), cos(x),cos(y),cos(z)] --
        // not sin/cos interleaved per axis.
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

// Differentiable pass-through so bwd_diff has a free function to hang off
// of (matches the pattern that worked in Section 4 -- calling bwd_diff on a
// struct method directly crashed the compiler in testing).
[Differentiable]
float eval_wrap(no_diff float3 p, Network network)
{
    return network.eval(p);
}

// Forward-only inference, for pulling predictions back to NumPy.
float predict(float3 p, Network network)
{
    return network.eval(p);
}

float predict_batch(int idx, Tensor<float3, 1> pts, Network network)
{
    return predict(pts.load(idx), network);
}

// Backprop an externally supplied per-point output gradient (computed on
// the CPU from clamped_l1, exactly as the book's net.backward(d_c + beta_far
// * d_far) does) into the network's weight/bias gradients.
void calculate_grads_from_dpred(int idx, Tensor<float3, 1> pts, Tensor<float, 1> dpred, Network network)
{
    float3 p = pts.load(idx);
    float g = dpred.load(idx);
    bwd_diff(eval_wrap)(p, network, g);
}

// Adam step, identical to Chapter 4 / Section 4.
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
