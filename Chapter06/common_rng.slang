// SPDX-License-Identifier: Apache-2.0
//
// A tiny GPU random-number generator, shared by every Section 5/6 sampling
// kernel that used to draw its training batches with NumPy's
// `np.random.default_rng` on the CPU. There is no CPU RNG anywhere in this
// chapter's training loops any more: each kernel invocation seeds its own
// `Rng` from (a per-run seed, the thread index), so every point AND its
// label are generated on the GPU, never downloaded-then-reuploaded.
//
// This is an ordinary (non-differentiable) hash-based generator -- PCG-style
// integer hashing (O'Neill 2014; the specific mix here is the widely used
// "pcg_hash" variant popularized for GPU shaders by Mark Jarzynski / Nathan
// Reed) -- good enough for Monte-Carlo training data, not for anything
// cryptographic.

uint pcg_hash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

public struct Rng
{
    uint state;

    // Combine a per-call seed with the thread index so every dispatch draws
    // an independent stream and every thread within it draws a different one.
    public __init(uint seed, uint idx)
    {
        state = pcg_hash(seed * 9781u + idx * 2654435761u + 1u);
    }

    [mutating]
    public float next_float()
    {
        state = pcg_hash(state);
        return float(state) * (1.0f / 4294967296.0f);   // in [0, 1)
    }

    [mutating]
    public float next_float(float lo, float hi)
    {
        return lo + (hi - lo) * next_float();
    }

    [mutating]
    public float3 next_float3(float lo, float hi)
    {
        return float3(next_float(lo, hi), next_float(lo, hi), next_float(lo, hi));
    }

    // Standard normal via Box-Muller.
    [mutating]
    public float next_gaussian()
    {
        float u1 = max(next_float(), 1e-7f);
        float u2 = next_float();
        return sqrt(-2.0f * log(u1)) * cos(6.28318530718f * u2);
    }

    [mutating]
    public float3 next_gaussian3()
    {
        return float3(next_gaussian(), next_gaussian(), next_gaussian());
    }

    // Uniform-on-sphere direction (normalized Gaussian vector).
    [mutating]
    public float3 next_unit_vector()
    {
        float3 v = next_gaussian3();
        float n = length(v);
        return n > 1e-12f ? v / n : float3(0, 0, 1);
    }
}
