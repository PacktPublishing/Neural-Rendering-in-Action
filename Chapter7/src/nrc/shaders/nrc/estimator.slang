implementing nrc;

namespace nrc
{
    // Compare accumulated area spread with this fraction of the camera's
    // primary footprint. This controls when a path may query the cache.
    internal static const float kSpreadThreshold = 0.01;

    // Permute the populated record domain, not a wrapped 32-bit product
    // reduced modulo its count. Each operation below is a bijection on the
    // surrounding power-of-two domain. Cycle walking restricts that bijection
    // to [0, recordCount); starting inside that set guarantees termination.
    // With the current <=65536-record pool the surrounding domain is at most
    // 65536 entries and fewer than half its entries are rejected.
    public uint shuffledTrainingIndex(uint sampleIndex, uint recordCount, uint seed)
    {
        if(recordCount <= 1u) return 0u;
        uint mask = recordCount - 1u;
        mask |= mask >> 1u;
        mask |= mask >> 2u;
        mask |= mask >> 4u;
        mask |= mask >> 8u;
        mask |= mask >> 16u;

        uint index = sampleIndex % recordCount;
        [loop]
        do
        {
            index ^= seed & mask;
            index = (index * 0x9e3779b1u) & mask;
            index ^= index >> 7u;
            index = (index * 0x85ebca6bu) & mask;
            index ^= index >> 5u;
            index = (index + (seed >> 16u)) & mask;
        }
        while(index >= recordCount);
        return index;
    }

    // Return the incoming segment's square-root-area contribution:
    // distance / sqrt(pdf * arrivalCosine). The PDF is the solid-angle density
    // sampled at the previous vertex; cosTheta is the absolute arrival cosine
    // against the current surface normal, supplied by the caller.
    public float spreadContribution(
        float segmentLength,
        float pdf,
        float cosTheta)
    {
        // Delta directions have zero spread. The renderer uses a negative
        // PDF (DIRAC); very large positive PDFs are also treated as impulses.
        if(pdf < 0.0 || pdf > 1e20)
        {
            return 0.0;
        }
        // Numerical floors regularize grazing arrivals and tiny densities.
        // They are separate from kSpreadThreshold, the relative area cutoff.
        float denominator = max(pdf * max(cosTheta, 1e-3), 1e-8);
        return sqrt(segmentLength * segmentLength / denominator);
    }

    // Sum the square-root-area terms along the path, then square that sum.
    // primaryFootprint is already an area; do not sum segment areas instead.
    public bool spreadExceeds(float spreadSqrtSum, float primaryFootprint)
    {
        float spread = spreadSqrtSum * spreadSqrtSum;
        return spread > kSpreadThreshold * primaryFootprint;
    }

    public bool isCachePositionValid(float3 normalizedPosition)
    {
        return all(normalizedPosition >= float3(0))
            && all(normalizedPosition <= float3(1));
    }

    // Only a bootstrapped training suffix may end at a cache query. The caller
    // restarts the square-root-area sum at the rendering query; suffixSpread
    // is that running sum, despite its shorter name. Full RR paths bypass this
    // test and continue to natural termination. Surface eligibility is checked
    // separately by the caller.
    public bool shouldTerminateTrainingSuffix(
        bool unbiased, bool hasRenderingQuery,
        float suffixSpread, float primaryFootprint)
    {
        return !unbiased && hasRenderingQuery
            && spreadExceeds(suffixSpread, primaryFootprint);
    }

    // Deterministic per-pixel selection. Unsigned 32-bit arithmetic intentionally
    // wraps modulo 2^32. The fixed primes 1973, 9277, and 26699 distinguish the
    // x, y, and frame contributions to the seed. Their exact values are
    // implementation choices; primality alone does not ensure a well-mixed hash.
    public bool isTrainingRay(uint2 pixel, uint frameIndex, uint threshold)
    {
        uint hash = pixel.x * 1973u + pixel.y * 9277u + frameIndex * 26699u;

        // MurmurHash3's fmix32 finalizer: these two multipliers and the shift
        // sequence 16, 13, 16 spread small input changes across the output bits.
        // https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
        hash = (hash ^ (hash >> 16)) * 0x85ebca6bu;
        hash = (hash ^ (hash >> 13)) * 0xc2b2ae35u;
        hash = hash ^ (hash >> 16);

        // The host-controlled threshold sets an approximate selection fraction
        // of threshold / 2^32 over the viewport, without a per-tile quota.
        return hash < threshold;
    }

    // Flattened-pixel variant used by the path tracer. 2654435761 is
    // 0x9E3779B1, the large prime also used as XXH32's PRIME32_1. Multiplying
    // by it mixes the pixel index into the seed; 26699 supplies the frame term.
    // https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md
    public bool isTrainingRay(uint pixelIndex, uint frameIndex, uint threshold)
    {
        uint hash = pixelIndex * 2654435761u + frameIndex * 26699u;

        // Same MurmurHash3 finalizer and threshold interpretation as above.
        hash = (hash ^ (hash >> 16)) * 0x85ebca6bu;
        hash = (hash ^ (hash >> 13)) * 0xc2b2ae35u;
        hash = hash ^ (hash >> 16);
        return hash < threshold;
    }

    // Called for selected training paths to choose suffixes that continue to
    // natural termination with Russian roulette and no cache bootstrapping.
    // The different seed and mixing sequence are intended to decorrelate this
    // decision from training selection; they do not guarantee independence.
    public bool isUnbiasedRrPath(uint pixelIndex, uint frameIndex)
    {
        // 0x9E3779B9 = floor(2^32 / phi), where phi is the golden ratio.
        // Its decimal value is 2654435769, eight larger than the multiplier in
        // isTrainingRay. The frame coefficient reuses a MurmurHash3 constant.
        uint hash = pixelIndex * 0x9E3779B9u + frameIndex * 0x85EBCA6Bu;

        // Custom mix: 0xc2b2ae35 also occurs in MurmurHash3, while 0x27d4eb2d
        // occurs in Wang's integer hash. This combination is its own sequence.
        // https://burtleburtle.net/bob/hash/integer.html
        hash = (hash ^ (hash >> 16)) * 0xc2b2ae35u;
        hash = (hash ^ (hash >> 13)) * 0x27d4eb2du;
        hash = hash ^ (hash >> 16);

        // 0xF is binary 1111: accept one of the 16 possible low-bit patterns.
        // This targets approximately 1/16 of training paths, not an exact quota
        // in each frame. The multipliers mix bits; this mask sets the fraction.
        return (hash & 0xFu) == 0u;
    }

    public QueryRecord makeQueryRecord(
        NrcVertexFeatures features,
        uint pixelIndex,
        float3 pathThroughput)
    {
        QueryRecord record;
        record.features = features;
        record.payload.pixelIdx = pixelIndex;
        record.payload.pathThroughput = pathThroughput;
        record.payload._pad0 = 0.0;
        record.payload._pad1 = 0.0;
        return record;
    }

    // Start a scattered-radiance target at this vertex with unit local
    // throughput, independent of the camera prefix. Create it after folding
    // this vertex's emission into older records, but before its direct lighting
    // and outgoing BSDF sample: its target includes scattering, not own emission.
    public PendingTrainingRecord makePendingTrainingRecord(
        NrcVertexFeatures features,
        uint vertexIndex)
    {
        PendingTrainingRecord record;
        record.features = features;
        record.suffixDirect = float3(0);
        record.suffixThroughput = float3(1);
        record.vertexIndex = vertexIndex;
        return record;
    }

    // localRadiance includes any sampling/MIS/visibility weights but no camera
    // throughput. The record's own transport brings this contribution back to
    // its target vertex. Fold local lighting before advancing outgoing transport.
    public void accumulateTrainingContribution(
        inout PendingTrainingRecord pending, float3 localRadiance)
    {
        pending.suffixDirect += pending.suffixThroughput * localRadiance;
    }

    // Compose an incremental BSDF, medium, or RR-survival factor into this
    // record's local transport, rather than assigning camera throughput.
    // Incoming segment attenuation precedes lighting at the arrival point;
    // outgoing BSDF and surviving RR factors follow the current local lighting.
    public void advanceTrainingThroughput(
        inout PendingTrainingRecord pending, float3 factor)
    {
        pending.suffixThroughput *= factor;
    }

    // Package traced local radiance and transport for later target completion.
    // Natural termination (including absorption or RR) zeros suffixThroughput
    // so no cache tail remains. Otherwise suffixEndFeatures identifies the
    // endpoint whose scattered radiance is still untraced; its own emission
    // has already been folded into pending records.
    // Flags preserve validity, the full-RR strategy, the masked source vertex
    // index, and natural termination; full-RR records never bootstrap.
    public TrainingRecord completeTrainingRecord(
        PendingTrainingRecord pending,
        NrcVertexFeatures suffixEndFeatures,
        bool isUnbiasedRr,
        bool suffixTerminated)
    {
        TrainingRecord record;
        record.features = pending.features;
        record.payload.suffixDirect = pending.suffixDirect;
        record.payload.suffixThroughput = suffixTerminated
            ? float3(0) : pending.suffixThroughput;
        record.payload.suffixEndFeatures = suffixEndFeatures;
        record.payload.flags = kTrainingRecordValid
            | (isUnbiasedRr ? kTrainingRecordUnbiasedRr : 0u)
            | ((pending.vertexIndex & kTrainingRecordVertexIndexMask)
                << kTrainingRecordVertexIndexShift)
            | (suffixTerminated ? kTrainingRecordSuffixTerminated : 0u);
        return record;
    }
}
