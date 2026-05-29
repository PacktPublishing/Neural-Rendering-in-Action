// Host-side sanity check for the record allocation sizes used by nrc_pass.cpp.
// Compiled Slang reflection/SPIR-V validation is required separately because
// these C++ sizeof assertions cannot detect shader-only layout drift.

#include <cstdint>
#include <cstdio>

// Must mirror the scalar leaf layout of nrc/records.slang::TrainingRecord.
// §2: oct -> sph rename; `valid` -> `flags` (same uint).
struct TrainingRecordCpu {
    float    position[3];
    float    viewDirSph[2];
    float    normalSph[2];
    float    roughness;
    float    diffuseAlbedo[3];
    float    specularAlbedo[3];
    float    suffixDirect[3];
    float    suffixThroughput[3];
    float    suffixEndPosition[3];
    float    suffixEndViewDirSph[2];
    float    suffixEndNormalSph[2];
    float    suffixEndRoughness;
    float    suffixEndDiffuse[3];
    float    suffixEndSpecular[3];
    uint32_t flags;
};

struct QueryRecordCpu {
    float    position[3];
    float    viewDirSph[2];
    float    normalSph[2];
    float    roughness;
    float    diffuseAlbedo[3];
    float    specularAlbedo[3];
    uint32_t pixelIdx;
    float    pathThroughput[3];
    float    _pad0;
    float    _pad1;
};

static_assert(sizeof(TrainingRecordCpu) == 140,
              "TrainingRecord must be 140 B to match nrc_pass.cpp::kTrainingRecordBytes");
static_assert(sizeof(QueryRecordCpu) == 80,
              "QueryRecord must be 80 B to match nrc_pass.cpp::kQueryRecordBytes");

int main() {
    std::printf("TrainingRecord size: %zu (expected 140)\n",
                sizeof(TrainingRecordCpu));
    std::printf("QueryRecord size:    %zu (expected 80)\n",
                sizeof(QueryRecordCpu));
    return 0;
}
