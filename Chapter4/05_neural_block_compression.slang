// SPDX-License-Identifier: Apache-2.0
//
// Step 6: Neural Block Compression
//
// Instead of storing separate full-res textures for each PBR channel, we store a
// single small latent texture (64x64 with 8 channels) and a tiny MLP that decodes
// all PBR channels simultaneously.  This mirrors how GPU block-compression formats
// (BC1/BC6H) store compressed blocks, except the "codec" is a learned neural network.
//
// Compression ratio example (256x256 source textures):
//   Uncompressed: 256x256 x 6 channels x 4 bytes = 1.5 MB
//   NBC:          64x64  x 8 latents  x 4 bytes  = 128 KB  (+ ~6 KB weights)
//   Ratio: ~11x
//
// Network outputs (indices 0-5), all mapped to [0,1] via sigmoid:
//   [0,1,2] albedo RGB
//   [3,4]   normal XY (stored representation; Z reconstructed on decode)
//   [5]     roughness

import slangpy;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

[Differentiable]
float leakyReLU(float x, float negativeSlope = 0.01f)
{
    return x > 0.0f ? x : x * negativeSlope;
}

[Differentiable]
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

[Differentiable]
float3 gamma_correct(float3 color)
{
    return pow(color, 1.0f / 2.2f);
}

// ---------------------------------------------------------------------------
// Network building blocks (identical to 05)
// ---------------------------------------------------------------------------

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;
    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float get_bias(int neuron) { return biases.get({neuron}); }

    [Differentiable]
    float get_weight(int neuron, int input) { return weights.get({neuron, input}); }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float grad) { biases_grad.set({neuron}, grad); }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float grad) { weights_grad.set({neuron, input}, grad); }

    [Differentiable]
    float[Outputs] forward(float[Inputs] x)
    {
        float[Outputs] y;
        [MaxIters(Outputs)]
        for (int row = 0; row < Outputs; ++row)
        {
            var sum = get_bias(row);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col)
                sum += get_weight(row, col) * x[col];
            y[row] = sum;
        }
        return y;
    }
}

struct LatentTexture<int NumLatents>
{
    Tensor<float, 3> texture;
    AtomicTensor<float, 3> texture_grads;

    [BackwardDerivative(get_latent_bwd)]
    float get_latent(int x, int y, int c) { return texture[y, x, c]; }

    void get_latent_bwd(int x, int y, int c, float g) { texture_grads.set({y, x, c}, g); }

    [Differentiable]
    float[NumLatents] sample(no_diff inout float2 uv)
    {
        int2 size = {texture.shape[1], texture.shape[0]};
        uv *= float2(size - 1);
        int2 c0 = clamp(int2(uv), int2(0), size - 1);
        int2 c1 = min(c0 + 1, size - 1);
        uv = saturate(uv - float2(c0));

        float result[NumLatents];
        [ForceUnroll]
        for (int i = 0; i < NumLatents; ++i)
        {
            float v00 = get_latent(c0.x, c0.y, i);
            float v10 = get_latent(c1.x, c0.y, i);
            float v01 = get_latent(c0.x, c1.y, i);
            float v11 = get_latent(c1.x, c1.y, i);
            result[i] = lerp(lerp(v00, v10, uv.x), lerp(v01, v11, uv.x), uv.y);
        }
        return result;
    }
}

// ---------------------------------------------------------------------------
// Network: 8 latents -> 32 -> 32 -> 6 outputs
// eval() returns a plain float[6]; no struct so the backward pass stays simple.
// ---------------------------------------------------------------------------

struct Network
{
    static const int NumLatents = 8;

    LatentTexture<NumLatents>         latent_texture;
    NetworkParameters<NumLatents, 32> layer0;
    NetworkParameters<32, 32>         layer1;
    NetworkParameters<32, 6>          layer2;

    [Differentiable]
    float[6] eval(no_diff float2 uv)
    {
        float enc[NumLatents] = latent_texture.sample(uv);

        float h0[32] = layer0.forward(enc);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i) h0[i] = leakyReLU(h0[i]);

        float h1[32] = layer1.forward(h0);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i) h1[i] = leakyReLU(h1[i]);

        float raw[6] = layer2.forward(h1);
        [ForceUnroll]
        for (int i = 0; i < 6; ++i) raw[i] = sigmoid(raw[i]);

        return raw;
    }
}

// ---------------------------------------------------------------------------
// Display functions (forward-only, called from Python)
// ---------------------------------------------------------------------------

// Panel 2: decoded albedo
float3 render_albedo(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float out[6] = network.eval(uv);
    return float3(out[0], out[1], out[2]);
}

// Panel 4: decoded normal map visualised as RGB
//   Normal XY stored in [0,1] -> remap to [-1,1] -> reconstruct Z -> pack to [0,1]
float3 render_normal(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float out[6] = network.eval(uv);
    float2 n  = float2(out[3], out[4]) * 2.0f - 1.0f;
    float  nz = sqrt(max(0.001f, 1.0f - dot(n, n)));
    return float3(n, nz) * 0.5f + 0.5f;
}

// Albedo loss image for display (matches 03 signature pattern)
float3 loss(int2 pixel, int2 resolution, no_diff float3 albedo_ref, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float out[6] = network.eval(uv);
    float3 color = float3(out[0], out[1], out[2]);
    float3 error = gamma_correct(color) - gamma_correct(albedo_ref);
    return error * error;
}

// ---------------------------------------------------------------------------
// Training loss: all PBR channels packed into float3 (matches 05 bwd_diff pattern)
//   albedo_ref   = (R, G, B) linear
//   material_ref = (normal_x, normal_y, roughness) in [0,1]
//
// Pairs of channel errors are summed into each float3 component so that the
// single 1.0f seed gradient weights all channels equally.
// ---------------------------------------------------------------------------

[Differentiable]
float3 training_loss(int2 pixel, int2 resolution,
                     no_diff float3 albedo_ref,
                     no_diff float3 material_ref,
                     Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    float out[6] = network.eval(uv);

    float3 albedo = float3(out[0], out[1], out[2]);
    float3 ae  = gamma_correct(albedo) - gamma_correct(albedo_ref);
    float  ne0 = out[3] - material_ref.x;
    float  ne1 = out[4] - material_ref.y;
    float  re  = out[5] - material_ref.z;

    return float3(ae.x * ae.x + ne0 * ne0,
                  ae.y * ae.y + ne1 * ne1,
                  ae.z * ae.z + re  * re);
}

// ---------------------------------------------------------------------------
// LCG random + Adam optimiser (identical to 05)
// ---------------------------------------------------------------------------

struct LCG
{
    uint state;
    __init(uint seed) { state = seed; }

    [mutating] uint next_uint()
    {
        const uint A = 1664525u, C = 1013904223u;
        state = A * state + C;
        return state;
    }

    [mutating] float next_float() { return (next_uint() >> 8) * 0x1p-24; }
}

void optimizer_step(inout float primal, inout float grad,
                    inout float mean,   inout float variance,
                    float learning_rate, int iteration)
{
    const float B1 = 0.9f, B2 = 0.999f, EPS = 1e-8f;
    mean     = lerp(grad,        mean,     B1);
    variance = lerp(grad * grad, variance, B2);
    float mHat = mean     / (1.0f - pow(B1, iteration));
    float vHat = variance / (1.0f - pow(B2, iteration));
    primal -= learning_rate * mHat / (sqrt(vHat) + EPS);
    grad = 0.0f;
}

// ---------------------------------------------------------------------------
// Gradient accumulation entry point
// ---------------------------------------------------------------------------

void calculate_grads(uint seed, int2 batch_index, int2 batch_size,
                     Tensor<float3, 2> albedo_ref,
                     Tensor<float3, 2> material_ref,
                     Network network)
{
    LCG lcg = LCG(seed);
    float2 uv  = (batch_index + float2(lcg.next_float(), lcg.next_float())) / float2(batch_size);
    int2 res   = {albedo_ref.shape[1], albedo_ref.shape[0]};
    int2 pixel = int2(uv * float2(res));

    float3 ref_albedo   = albedo_ref.getv(pixel);
    float3 ref_material = material_ref.getv(pixel);

    bwd_diff(training_loss)(pixel, res, ref_albedo, ref_material, network, 1.0f);
}
