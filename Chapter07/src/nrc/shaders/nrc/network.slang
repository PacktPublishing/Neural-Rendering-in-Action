implementing nrc;

namespace nrc
{
    // The default Chapter 7 configuration remains shared with the C++ layout
    // code and push constants, while Network itself is reusable at other
    // compile-time dimensions.
    public static const int kNetworkInputCount = 64;
    public static const int kNetworkHiddenCount = 64;
    public static const int kNetworkOutputCount = 3;
    public static const int kNetworkHiddenLayerCount = 5;
    internal static const CoopVecMatrixLayout kNetworkMatrixLayout =
        CoopVecMatrixLayout::TrainingOptimal;
    internal static const CoopVecComponentType kNetworkComponentType =
        CoopVecComponentType::Float16;

    internal struct GradientBuffer : IDifferentiablePtrType
    {
        typealias Differential = GradientBuffer;
        RWByteAddressBuffer buffer;
        __init(RWByteAddressBuffer value) { buffer = value; }
    }

    internal struct ParameterBuffer : IDifferentiablePtrType
    {
        typealias Differential = GradientBuffer;
        ByteAddressBuffer buffer;
        __init(ByteAddressBuffer value) { buffer = value; }
    }

    // The RTXNS linear primitive uses one component type/layout for both
    // weights and their derivatives. Keep the forward matrix in FP16, but
    // give autodiff a separate FP32 accumulation layout queried by the host.
    // Optimal layouts are opaque: FP32 offsets are never inferred by doubling
    // their FP16 counterparts.
    internal CoopVec<float16_t, M> trainingLinear<let M : int, let K : int>(
        CoopVec<float16_t, K> input,
        ParameterBuffer parameters,
        uint4 offsets)
    {
        return coopVecMatMulAdd<float16_t, M>(
            input, CoopVecComponentType::Float16,
            parameters.buffer, offsets.x, CoopVecComponentType::Float16,
            parameters.buffer, offsets.y, CoopVecComponentType::Float16,
            kNetworkMatrixLayout, false, 0);
    }

    [BackwardDerivativeOf(trainingLinear)]
    internal void trainingLinearBackward<let M : int, let K : int>(
        inout DifferentialPair<CoopVec<float16_t, K>> input,
        DifferentialPtrPair<ParameterBuffer> parameters,
        uint4 offsets,
        CoopVec<float16_t, M> gradient)
    {
        // Vulkan requires FP16 outer-product operands, but independently
        // supports FP32 accumulation. Bias reduction takes its storage type
        // from its vector operand, so widen that vector explicitly.
        coopVecOuterProductAccumulate(
            gradient, input.p, parameters.d.buffer, offsets.z, 0,
            kNetworkMatrixLayout, CoopVecComponentType::Float32);
        var biasGradient = CoopVec<float, M>(0.0);
        [ForceUnroll]
        for(int lane = 0; lane < M; ++lane)
            biasGradient[lane] = float(gradient[lane]);
        coopVecReduceSumAccumulate(biasGradient, parameters.d.buffer, offsets.w);
        input = diffPair(input.p, coopVecMatMul<float16_t, K>(
            gradient, CoopVecComponentType::Float16,
            parameters.p.buffer, offsets.x, CoopVecComponentType::Float16,
            kNetworkMatrixLayout, true, 0));
    }

    // Keeps cooperative-vector storage private to the RTXNS facade while
    // allowing a network to have any compile-time output width.
    public struct NetworkValues<let N : int>
    {
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

    public struct Network<
        let InputCount : int,
        let HiddenCount : int,
        let OutputCount : int,
        let HiddenLayerCount : int>
    {
        public static const int LayerCount = HiddenLayerCount + 1;

        private void validateDimensions()
        {
            static_assert(InputCount > 0, "Network input count must be positive");
            static_assert(HiddenCount > 0, "Network hidden count must be positive");
            static_assert(OutputCount > 0, "Network output count must be positive");
            static_assert(
                HiddenLayerCount > 0,
                "Network requires at least one hidden layer");
        }

        private CoopVec<float16_t, InputCount> toInput(
            ScalarEncoding<InputCount> encoded)
        {
            var input = CoopVec<float16_t, InputCount>(float16_t(0));
            [ForceUnroll]
            for(int lane = 0; lane < InputCount; ++lane)
            {
                input[lane] = float16_t(encoded.getLane(lane));
            }
            return input;
        }

        private NetworkValues<OutputCount> fromOutput(
            CoopVec<float16_t, OutputCount> output)
        {
            NetworkValues<OutputCount> result;
            [ForceUnroll]
            for(int lane = 0; lane < OutputCount; ++lane)
            {
                result.lanes[lane] = float(output[lane]);
            }
            return result;
        }

        private CoopVec<float16_t, OutputCount> toGradient(
            NetworkValues<OutputCount> upstreamGradient)
        {
            var gradient = CoopVec<float16_t, OutputCount>(float16_t(0));
            [ForceUnroll]
            for(int lane = 0; lane < OutputCount; ++lane)
            {
                gradient[lane] = float16_t(upstreamGradient.getLane(lane));
            }
            return gradient;
        }

        // Keeping activation construction in separate methods avoids the
        // adjacent-template-constructor parser issue in Slang 2026.5.
        private rtxns::mlp::ReLUAct<float16_t, HiddenCount> makeRelu()
        {
            rtxns::mlp::ReLUAct<float16_t, HiddenCount> activation;
            return activation;
        }

        private rtxns::mlp::LinearAct<float16_t, OutputCount> makeLinear()
        {
            rtxns::mlp::LinearAct<float16_t, OutputCount> activation =
                rtxns::mlp::LinearAct<float16_t, OutputCount>(float16_t(1.0));
            return activation;
        }

        public NetworkValues<OutputCount> inference(
            ByteAddressBuffer weights,
            uint matrixOffset[LayerCount],
            uint biasOffset[LayerCount],
            ScalarEncoding<InputCount> encoded)
        {
            validateDimensions();
            var input = toInput(encoded);
            var mlp = rtxns::mlp::InferenceMLP<
                float16_t,
                HiddenLayerCount,
                InputCount,
                HiddenCount,
                OutputCount,
                kNetworkMatrixLayout,
                kNetworkComponentType>(weights, matrixOffset, biasOffset);
            var relu = makeRelu();
            var linear = makeLinear();
            return fromOutput(mlp.forward(input, relu, linear));
        }

        public NetworkValues<OutputCount> forward(
            ByteAddressBuffer weights,
            RWByteAddressBuffer derivatives,
            uint matrixOffset[LayerCount],
            uint biasOffset[LayerCount],
            ScalarEncoding<InputCount> encoded)
        {
            validateDimensions();
            var input = toInput(encoded);
            var mlp = rtxns::mlp::TrainingMLP<
                float16_t,
                HiddenLayerCount,
                InputCount,
                HiddenCount,
                OutputCount,
                kNetworkMatrixLayout,
                kNetworkComponentType>(
                    weights, derivatives, matrixOffset, biasOffset);
            var relu = makeRelu();
            var linear = makeLinear();
            return fromOutput(mlp.forward(input, relu, linear));
        }

        // backward differentiates this expression and recomputes its forward
        // values. Keep its layer topology and activations consistent with
        // forward above; both must evaluate the same unclamped network.
        [Differentiable]
        private static CoopVec<float16_t, OutputCount> trainingForward(
            CoopVec<float16_t, InputCount> input,
            ParameterBuffer parameters,
            uint4 offsets[LayerCount])
        {
            var hidden = rtxns::relu(trainingLinear<HiddenCount, InputCount>(
                input, parameters, offsets[0]));
            [ForceUnroll]
            for(int layer = 1; layer < HiddenLayerCount; ++layer)
                hidden = rtxns::relu(trainingLinear<HiddenCount, HiddenCount>(
                    hidden, parameters, offsets[layer]));
            return trainingLinear<OutputCount, HiddenCount>(
                hidden, parameters, offsets[HiddenLayerCount]);
        }

        public void backward(
            ByteAddressBuffer weights,
            RWByteAddressBuffer derivatives,
            uint matrixOffset[LayerCount],
            uint biasOffset[LayerCount],
            uint gradientMatrixOffset[LayerCount],
            uint gradientBiasOffset[LayerCount],
            ScalarEncoding<InputCount> encoded,
            NetworkValues<OutputCount> upstreamGradient)
        {
            validateDimensions();
            uint4 offsets[LayerCount];
            [ForceUnroll]
            for(int layer = 0; layer < LayerCount; ++layer)
                offsets[layer] = uint4(matrixOffset[layer], biasOffset[layer],
                    gradientMatrixOffset[layer], gradientBiasOffset[layer]);

            var input = diffPair(toInput(encoded));
            var parameters = ParameterBuffer(weights);
            var gradientBuffer = GradientBuffer(derivatives);
            bwd_diff(trainingForward)(input,
                DifferentialPtrPair<ParameterBuffer>(parameters, gradientBuffer),
                offsets, toGradient(upstreamGradient));
        }
    }

    public typealias NrcNetwork = Network<
        kNetworkInputCount,
        kNetworkHiddenCount,
        kNetworkOutputCount,
        kNetworkHiddenLayerCount>;

    public static const int kNetworkLayerCount = NrcNetwork::LayerCount;

    // Semantic convenience methods for the working 64x64x3 Chapter 7 NRC.
    // The dimension-generic forward/backward implementation remains owned by
    // Network above; this extension converts semantic features and RGB values
    // and applies the nonnegative output floor used for inference.
    public extension Network<
        kNetworkInputCount,
        kNetworkHiddenCount,
        kNetworkOutputCount,
        kNetworkHiddenLayerCount>
    {
        // Return nonnegative, reflectance-normalized scattered radiance.
        // Callers reconstruct physical radiance with reflectanceFactor.
        public float3 inference(
            ByteAddressBuffer weights,
            uint matrixOffset[kNetworkLayerCount],
            uint biasOffset[kNetworkLayerCount],
            NrcVertexFeatures features)
        {
            NrcFeatureEncoder encoder;
            EncodedNrcFeatures encoded = encodeWith(encoder, features);
            NetworkValues<3> output = this.inference(
                weights, matrixOffset, biasOffset, encoded);
            return max(
                float3(output.getLane(0), output.getLane(1), output.getLane(2)),
                float3(0));
        }

        // Return the raw normalized RGB prediction for the training loss.
        // Preserve negative linear outputs so their loss gradient can correct them.
        public float3 forward(
            ByteAddressBuffer weights,
            RWByteAddressBuffer derivatives,
            uint matrixOffset[kNetworkLayerCount],
            uint biasOffset[kNetworkLayerCount],
            NrcVertexFeatures features)
        {
            NrcFeatureEncoder encoder;
            EncodedNrcFeatures encoded = encodeWith(encoder, features);
            NetworkValues<3> output = this.forward(
                weights, derivatives, matrixOffset, biasOffset, encoded);
            return float3(
                output.getLane(0), output.getLane(1), output.getLane(2));
        }

        public void backward(
            ByteAddressBuffer weights,
            RWByteAddressBuffer derivatives,
            uint matrixOffset[kNetworkLayerCount],
            uint biasOffset[kNetworkLayerCount],
            uint gradientMatrixOffset[kNetworkLayerCount],
            uint gradientBiasOffset[kNetworkLayerCount],
            NrcVertexFeatures features,
            float3 upstreamGradient)
        {
            NrcFeatureEncoder encoder;
            EncodedNrcFeatures encoded = encodeWith(encoder, features);
            NetworkValues<3> gradient;
            gradient.setLane(0, upstreamGradient.x);
            gradient.setLane(1, upstreamGradient.y);
            gradient.setLane(2, upstreamGradient.z);
            this.backward(
                weights,
                derivatives,
                matrixOffset,
                biasOffset,
                gradientMatrixOffset,
                gradientBiasOffset,
                encoded,
                gradient);
        }
    }
}
