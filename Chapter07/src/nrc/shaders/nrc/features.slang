implementing nrc;

namespace nrc
{
    internal static const int kFrequencyCount = 12;
    internal static const int kOneBlobCount = 4;
    public static const int kEncodedFeatureCount = 64;

    internal static const int kPositionXOffset = 0;
    internal static const int kPositionYOffset =
        kPositionXOffset + kFrequencyCount;
    internal static const int kPositionZOffset =
        kPositionYOffset + kFrequencyCount;
    internal static const int kViewThetaOffset =
        kPositionZOffset + kFrequencyCount;
    internal static const int kViewPhiOffset =
        kViewThetaOffset + kOneBlobCount;
    internal static const int kNormalThetaOffset =
        kViewPhiOffset + kOneBlobCount;
    internal static const int kNormalPhiOffset =
        kNormalThetaOffset + kOneBlobCount;
    internal static const int kRoughnessOffset =
        kNormalPhiOffset + kOneBlobCount;
    internal static const int kMaterialOffset =
        kRoughnessOffset + kOneBlobCount;

    // Maps a unit direction to normalized spherical (theta, phi).
    internal float2 encodeDirection(float3 direction)
    {
        float3 value = normalize(direction);
        float theta = acos(clamp(value.z, -1.0, 1.0)) * (1.0 / kPi);
        float phi = atan2(value.y, value.x) * (1.0 / (2.0 * kPi)) + 0.5;
        return float2(theta, phi);
    }

    // Supply position already normalized to the scene bounds. Directions are
    // world-space: viewDirection points toward the preceding path vertex
    // (-ray.Direction), and normal is the shading normal. Roughness and RGB
    // diffuse/specular reflectance must already be prepared by the renderer.
    public NrcVertexFeatures makeVertexFeatures(
        float3 position,
        float3 viewDirection,
        float3 normal,
        float roughness,
        float3 diffuseAlbedo,
        float3 specularAlbedo)
    {
        NrcVertexFeatures result;
        result.position = position;
        result.viewDirSph = encodeDirection(viewDirection);
        result.normalSph = encodeDirection(normal);
        result.roughness = roughness;
        result.diffuseAlbedo = diffuseAlbedo;
        result.specularAlbedo = specularAlbedo;
        return result;
    }

    // Use one positive factor for both normalization and reconstruction.
    // Flooring only the training divisor darkened channels below the floor
    // when cache queries multiplied by their smaller, unfloored reflectance.
    public float3 reflectanceFactor(NrcVertexFeatures features)
    {
        return max(features.diffuseAlbedo + features.specularAlbedo, float3(1e-3));
    }

    public typealias EncodedNrcFeatures =
        ScalarEncoding<kEncodedFeatureCount>;

    // The authoritative composite encoder. It uses the same generic encoder
    // interface as the reusable frequency and one-blob implementations while
    // owning the private 64-lane packing convention.
    public struct NrcFeatureEncoder : IEncoder<NrcVertexFeatures>
    {
        public typealias Encoding = EncodedNrcFeatures;

        public Encoding encode(NrcVertexFeatures features)
        {
            static_assert(
                kMaterialOffset + 8 == kEncodedFeatureCount,
                "NrcFeatureEncoder packing must fill exactly 64 lanes");
            Encoding result;
            [ForceUnroll]
            for(int index = 0; index < kEncodedFeatureCount; ++index)
            {
                result.lanes[index] = 0.0;
            }

            FrequencyEncoder<kFrequencyCount> frequencyEncoder;
            ScalarEncoding<kFrequencyCount> frequenciesX = encodeWith(
                frequencyEncoder, features.position.x);
            ScalarEncoding<kFrequencyCount> frequenciesY = encodeWith(
                frequencyEncoder, features.position.y);
            ScalarEncoding<kFrequencyCount> frequenciesZ = encodeWith(
                frequencyEncoder, features.position.z);
            [ForceUnroll]
            for(int index = 0; index < kFrequencyCount; ++index)
            {
                result.lanes[kPositionXOffset + index] =
                    frequenciesX.getLane(index);
                result.lanes[kPositionYOffset + index] =
                    frequenciesY.getLane(index);
                result.lanes[kPositionZOffset + index] =
                    frequenciesZ.getLane(index);
            }

            OneBlobEncoder<kOneBlobCount> oneBlobEncoder;
            ScalarEncoding<kOneBlobCount> viewTheta = encodeWith(
                oneBlobEncoder, features.viewDirSph.x);
            ScalarEncoding<kOneBlobCount> viewPhi = encodeWith(
                oneBlobEncoder, features.viewDirSph.y);
            ScalarEncoding<kOneBlobCount> normalTheta = encodeWith(
                oneBlobEncoder, features.normalSph.x);
            ScalarEncoding<kOneBlobCount> normalPhi = encodeWith(
                oneBlobEncoder, features.normalSph.y);
            // Map nonnegative roughness into [0, 1) before one-blob encoding.
            ScalarEncoding<kOneBlobCount> roughness = encodeWith(
                oneBlobEncoder, 1.0 - exp(-features.roughness));

            [ForceUnroll]
            for(int index = 0; index < kOneBlobCount; ++index)
            {
                result.lanes[kViewThetaOffset + index] =
                    viewTheta.getLane(index);
                result.lanes[kViewPhiOffset + index] =
                    viewPhi.getLane(index);
                result.lanes[kNormalThetaOffset + index] =
                    normalTheta.getLane(index);
                result.lanes[kNormalPhiOffset + index] =
                    normalPhi.getLane(index);
                result.lanes[kRoughnessOffset + index] =
                    roughness.getLane(index);
            }

            result.lanes[kMaterialOffset + 0] = features.diffuseAlbedo.x;
            result.lanes[kMaterialOffset + 1] = features.diffuseAlbedo.y;
            result.lanes[kMaterialOffset + 2] = features.diffuseAlbedo.z;
            result.lanes[kMaterialOffset + 3] = features.specularAlbedo.x;
            result.lanes[kMaterialOffset + 4] = features.specularAlbedo.y;
            result.lanes[kMaterialOffset + 5] = features.specularAlbedo.z;
            // Constant lanes complete the fixed 64-value input layout.
            result.lanes[kMaterialOffset + 6] = 1.0;
            result.lanes[kMaterialOffset + 7] = 1.0;

            return result;
        }
    }
}
