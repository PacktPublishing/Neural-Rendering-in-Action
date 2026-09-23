implementing nrc;

namespace nrc
{
    // GPU record flags. Names replace raw masks in pass code while preserving
    // the existing packed representation.
    public static const uint kTrainingRecordValid = 0x1u;
    public static const uint kTrainingRecordUnbiasedRr = 0x2u;
    public static const uint kTrainingRecordVertexIndexMask = 0x3Fu;
    public static const uint kTrainingRecordVertexIndexShift = 2u;
    public static const uint kTrainingRecordSuffixTerminated = 0x100u;

    public static const int kMaxPendingTrainingRecords = 8;

    // Semantic features shared by record construction and the network facade.
    // Directions use the existing normalized spherical representation.
    public struct NrcVertexFeatures
    {
        public float3 position;
        public float2 viewDirSph;
        public float2 normalSph;
        public float roughness;
        public float3 diffuseAlbedo;
        public float3 specularAlbedo;
    }

    // Query and training records share the same semantic vertex prefix. Their
    // distinct payloads retain the existing scalar-layout leaf offsets and
    // 80/140-byte buffer strides.
    public struct NrcRecord<TPayload>
    {
        public NrcVertexFeatures features;
        public TPayload payload;

        public NrcVertexFeatures getFeatures()
        {
            return features;
        }
    }

    public struct QueryPayload
    {
        public uint pixelIdx;
        public float3 pathThroughput;
        public float _pad0;
        public float _pad1;
    }

    public struct TrainingPayload
    {
        // Sampled scattered radiance at the starting vertex, including its
        // direct-light estimate and transported later lighting, but excluding
        // its own emission. This is in physical radiance units.
        public float3 suffixDirect;
        // Local transport weight multiplying the cache endpoint's radiance.
        public float3 suffixThroughput;
        // Used only when the flags permit a bootstrapped suffix tail.
        public NrcVertexFeatures suffixEndFeatures;
        public uint flags;
    }

    public typealias QueryRecord = NrcRecord<QueryPayload>;
    public typealias TrainingRecord = NrcRecord<TrainingPayload>;

    public extension NrcRecord<TrainingPayload>
    {
        public NrcVertexFeatures getEndFeatures()
        {
            return payload.suffixEndFeatures;
        }

        public bool isValid()
        {
            return (payload.flags & kTrainingRecordValid) != 0u;
        }

        public bool isUnbiasedRr()
        {
            return (payload.flags & kTrainingRecordUnbiasedRr) != 0u;
        }

        public bool isSuffixTerminated()
        {
            return (payload.flags & kTrainingRecordSuffixTerminated) != 0u;
        }
    }

    // Ray-generation state that has not yet been completed into a GPU record.
    // These semantic types are local path state rather than host-visible ABI.
    public struct PendingTrainingRecord
    {
        public NrcVertexFeatures features;
        public float3 suffixDirect;
        // Transport from this record's vertex to the current path vertex.
        // Kept locally so a zero channel in the camera prefix cannot destroy
        // the radiance target at this vertex.
        public float3 suffixThroughput;
        public uint vertexIndex;
    }

    public struct TrainingSample
    {
        public NrcVertexFeatures features;
        public float3 target;
    }

    // target must already be scattered radiance divided by reflectanceFactor
    // at features. This constructor only pairs features with that normalized RGB.
    public TrainingSample makeTrainingSample(
        NrcVertexFeatures features,
        float3 target)
    {
        TrainingSample sample;
        sample.features = features;
        sample.target = target;
        return sample;
    }
}
