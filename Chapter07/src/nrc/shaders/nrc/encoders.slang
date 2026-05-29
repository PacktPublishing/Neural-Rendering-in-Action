implementing nrc;

namespace nrc
{
    internal static const float kPi = 3.14159265;

    // Generic consumers can inspect an encoder result without knowing its
    // concrete storage type.
    public interface IEncoding
    {
        public static const int count;
        public float getLane(int index);
    }

    // Fixed-size encoded values keep their raw array private while providing
    // safe construction and read access to other modules.
    public struct ScalarEncoding<let N : int> : IEncoding
    {
        public static const int count = N;
        internal float lanes[N];

        public float getLane(int index)
        {
            return lanes[index];
        }

        [mutating]
        public void setLane(int index, float value)
        {
            lanes[index] = value;
        }
    }

    public interface IEncoder<TInput>
    {
        associatedtype Encoding : IEncoding;
        Encoding encode(TInput value);
    }

    public TEncoder.Encoding encodeWith<
        TInput,
        TEncoder : IEncoder<TInput>>(
        TEncoder encoder,
        TInput value)
    {
        return encoder.encode(value);
    }

    // Sine-only positional encoding used by the original Chapter 7 NRC.
    public struct FrequencyEncoder<let N : int> : IEncoder<float>
    {
        public typealias Encoding = ScalarEncoding<N>;

        public Encoding encode(float value)
        {
            static_assert(N > 0, "FrequencyEncoder requires at least one lane");
            Encoding result;
            float phase = kPi * value;
            float frequency = 1.0;
            [ForceUnroll]
            for(int index = 0; index < N; ++index)
            {
                result.lanes[index] = sin(frequency * phase);
                frequency *= 2.0;
            }
            return result;
        }
    }

    // Integral of the normalized quartic kernel (15/16) * (1 - u*u)^2
    // on [-1, 1]. Here 0.9375 = 15/16; integrating 1 - 2*u^2 + u^4
    // gives u - (2/3)*u^3 + (1/5)*u^5, and +0.5 sets CDF(0) = 0.5.
    // inverseRadius scales distance to u; clamping extends the CDF to
    // zero and one outside the kernel's support.
    internal float quarticCdf(float value, float inverseRadius)
    {
        float u = value * inverseRadius;
        float u2 = u * u;
        float u4 = u2 * u2;
        return clamp(
            0.9375 * u * (1.0 - (2.0 / 3.0) * u2 + (1.0 / 5.0) * u4) + 0.5,
            0.0,
            1.0);
    }

    // Quartic one-blob encoding. Chapter 7 instantiates this with four bins;
    // the bin count remains a compile-time choice of the implementation.
    public struct OneBlobEncoder<let N : int> : IEncoder<float>
    {
        public typealias Encoding = ScalarEncoding<N>;

        // Integrate a kernel of radius 1/N over each bin. Inputs are clamped
        // to [0, 1]; bins do not wrap, and mass outside this interval is
        // discarded without renormalizing the remaining bin values.
        public Encoding encode(float value)
        {
            static_assert(N > 0, "OneBlobEncoder requires at least one bin");
            Encoding result;
            float x = saturate(value);
            float inverseRadius = float(N);
            [ForceUnroll]
            for(int index = 0; index < N; ++index)
            {
                float lower = float(index) / float(N);
                float upper = float(index + 1) / float(N);
                result.lanes[index] = quarticCdf(upper - x, inverseRadius)
                    - quarticCdf(lower - x, inverseRadius);
            }
            return result;
        }
    }
}
