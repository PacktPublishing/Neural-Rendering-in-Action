// NrcPass — host-side owner of all Neural Radiance Cache GPU resources.
//
// Responsibilities:
//   - Allocate and destroy the buffers
//   - Own the compute pipelines for cache query, training-record completion,
//     training (forward+backward), and Adam optimizer.
//   - Provide a descriptor set the host's ray-gen shader binds at set index 3
//     so the path tracer can emit training records and query records.
//   - Dispatch all four compute passes in the correct order each frame.
//
// Wires lifecycle hooks and buffer allocation.

#pragma once

// When the host defines VK_NO_PROTOTYPES (volk loader pattern, used by
// nvpro_core2), include volk.h to pick up the function-pointer declarations.
// Otherwise fall back to plain vulkan.h so this header is portable.
#if defined(VK_NO_PROTOTYPES)
  #include <volk.h>
#else
  #include <vulkan/vulkan.h>
#endif
#include <cstdint>
#include "nrc/nrc_mlp_layout.h"

namespace nrc {

struct NrcParams {
    // Frame resolution the path tracer is dispatched at.
    uint32_t frameWidth  = 1920;
    uint32_t frameHeight = 1080;

    // NRC paper defaults
    //
    // 4 Adam steps × 16384 records/step × disjoint LCG-shuffled
    // slices of a single 65536-record pool per frame. Matches paper §3.5.
    uint32_t trainingRecordsPerFrame    = 65536;
    uint32_t recordsPerBatch            = 16384;  // per Adam step
    uint32_t adamStepsPerFrame          = 4;
    // §3 P-controller target: the training-ray mask is adjusted each frame
    // to keep emission near this many *paths* (~2.5 records/path avg on
    // shader-ball, so ~60k records). Paper §3.5 adaptive tiling.
    uint32_t trainingPathsTarget        = 26214;  // ≈ 65536 / 2.5
    uint32_t suffixLength               = 2;     // bounces in training-ray suffix

    float    spreadThresholdC           = 0.01f; // terminate when spread > c · primary_footprint
    float    emaAlpha                   = 0.99f;
};

class NrcPass {
public:
    NrcPass();
    ~NrcPass();

    // Non-copyable, non-movable — owns raw Vulkan handles.
    NrcPass(const NrcPass&)            = delete;
    NrcPass& operator=(const NrcPass&) = delete;

    // Allocate buffers; load pipelines in a follow-up task.
    void init(VkDevice device, VkPhysicalDevice physDev, const NrcParams& params);

    // Release everything allocated by init().
    void destroy();

    // Called from the host's per-frame command buffer, before the ray tracer.
    // Resets atomic counters and issues a barrier.
    void recordFrame(VkCommandBuffer cmd, uint32_t frameIdx);

    // Called from the host's per-frame command buffer, after the ray tracer
    // has run. Copies the atomic counters to a host-visible staging buffer
    // if a readback was requested. Future tasks will dispatch the cache-query
    // and training passes here.
    void recordFramePost(VkCommandBuffer cmd, uint32_t frameIdx);

    const NrcParams& params() const { return m_params; }

    // Descriptor set 3 contract with the ray-gen shader. Host binds this set
    // at index 3 before vkCmdTraceRaysKHR; the Slang patch in
    // shaders/nrc_patch_pathtrace.slang declares matching [[vk::binding(N,3)]].
    //
    // Bindings:
    //   0 : RWStructuredBuffer<TrainingRecord>  g_trainingRecords
    //   1 : RWStructuredBuffer<QueryRecord>     g_queryRecords
    //   2 : RWStructuredBuffer<uint>            g_trainingCount
    //   3 : RWStructuredBuffer<uint>            g_queryCount
    //   4 : RWStructuredBuffer<float4>          g_nrcContrib
    //   5 : RWStructuredBuffer<float4>          g_pathPrefix  (§1 same-frame)
    VkDescriptorSetLayout descriptorSetLayout() const { return m_descSetLayout; }
    VkDescriptorSet       descriptorSet()       const { return m_descSet; }

    // §1 compose pass: host calls this each frame before recordFramePost so
    // the compose dispatch can write into the correct VkImageView (the path
    // tracer's eResultImage gbuffer view, VK_FORMAT_R32G32B32A32_SFLOAT).
    //
    // The compose pass also needs to know how the host accumulates sample
    // history so its per-frame output matches ray-gen's NRC-off path.
    void setOutImageView(VkImageView view) { m_outImageView = view; }
    void setAccumulationParams(uint32_t totalSamples, uint32_t numSamples,
                               bool firstFrameOrReset) {
        m_accumTotalSamples      = totalSamples;
        m_accumNumSamples        = numSamples;
        m_accumFirstFrameOrReset = firstFrameOrReset;
    }

    // §1 bugfix: tell NrcPass the actual per-frame render size so compose and
    // cache-query use the same row stride for g_pathPrefix / g_nrcContrib that
    // ray-gen used when writing them (ray-gen indexes by the gbuffer size).
    // nrc_pass_init() only takes a one-time size — the renderer's gbuffer may
    // differ (e.g. 1280x720 vs the 1920x1080 init default). If width/height is
    // 0, dispatches fall back to m_params.frameWidth/Height.
    //
    // Buffer sanity: this must never exceed m_params.frameWidth *
    // m_params.frameHeight (which is what m_contribBuffer and m_pathPrefixBuffer
    // are sized to). Init defaults to 1920x1080, the largest common render
    // size — see TODO(Task 6) near the allocations in nrc_pass.cpp.
    void setRenderSize(uint32_t width, uint32_t height) {
        m_renderWidth  = width;
        m_renderHeight = height;
    }

    // Debug readback. After recordFrame() has run and the command buffer has
    // been submitted and queue-waited, call this to get the number of training
    // and query records emitted by the ray-gen shader this frame.
    // Task 9: used to verify query emission is happening.
    // Returns {0,0} if the readback buffer hasn't been populated.
    struct FrameCounts { uint32_t training = 0; uint32_t queries = 0; };
    FrameCounts readbackCounts() const;

    // Request the counters be copied to a host-visible staging buffer at the
    // end of the next recordFrame() call. The copy is recorded into the same
    // command buffer; readback requires a queue wait afterwards.
    void requestReadback();

private:
    struct Buffer {
        VkBuffer       handle = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size   = 0;
    };

    Buffer allocate(VkDeviceSize size, VkBufferUsageFlags usage);
    void   freeBuffer(Buffer& b);

    void   createDescriptorSet();
    void   destroyDescriptorSet();
    VkBuffer inferenceWeightsBuffer() const;
    void   updateInferenceWeightDescriptors();

    // Cache-query compute pipeline.
    void   createCacheQueryPipeline();
    void   destroyCacheQueryPipeline();
    void   dispatchCacheQuery(VkCommandBuffer cmd);

    // §1 compose compute pipeline: sums g_pathPrefix + g_nrcContrib into
    // the ray-tracer's output image, with frame-accumulation blending.
    void   createComposePipeline();
    void   destroyComposePipeline();
    void   dispatchCompose(VkCommandBuffer cmd);

    // §1 pipeline-ordering barriers (thin wrappers around vkCmdPipelineBarrier
    // to keep recordFramePost readable).
    void   issueBarrierRayGenToCompute(VkCommandBuffer cmd);
    void   issueBarrierOptimizerToCacheQuery(VkCommandBuffer cmd);
    void   issueBarrierToCompose(VkCommandBuffer cmd);

    // Record-completion compute pipeline: builds training targets.
    void   createRecordCompletePipeline();
    void   destroyRecordCompletePipeline();
    void   dispatchRecordCompletion(VkCommandBuffer cmd);

    // Training compute pipeline: relative-L2 loss + backward via autodiff,
    // writes gradients into m_gradients.
    void   createTrainPipeline();
    void   destroyTrainPipeline();
    void   dispatchTrain(VkCommandBuffer cmd);

    // Optimizer compute pipeline: Adam step + EMA update.
    void   createOptimizerPipeline();
    void   destroyOptimizerPipeline();
    void   dispatchOptimizer(VkCommandBuffer cmd);

    // Xavier-initialize weights on CPU, upload via a host-coherent staging
    // buffer, then use vkCmdConvertCooperativeVectorMatrixNV to produce
    // TrainingOptimal weights in m_primaryWeights and m_emaWeights.
    // Called on first recordFrame() using the caller's command buffer.
    void   fillWeightStaging(uint32_t seed);
    void   resetTrainingState();
    void   initializeWeightsFirstFrame(VkCommandBuffer cmd);

    NrcParams        m_params;
    VkDevice         m_device         = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    bool             m_initialized    = false;
    bool             m_enabled        = true;   // default ON; toggle via UI/CLI

    // Buffers
    Buffer m_trainingRecords;   // 16,384 records × 140 B
    Buffer m_queryRecords;      // frameWidth × frameHeight × ~80 B
    Buffer m_trainingCount;     // single uint, atomic-incremented by ray-gen
    Buffer m_queryCount;        // single uint, atomic-incremented by ray-gen
    Buffer m_primaryWeights;    // ~25 KB rounded up, row-major for optimizer
    Buffer m_optimalWeights;    // same size, tiled 16×16 for matmul
    Buffer m_emaWeights;        // only set read by cache-query pass
    Buffer m_adamMoments;       // 2 fp32 moments per fp16 parameter
    Buffer m_gradients;         // primary weight size, atomic-add target

    // Descriptor set 3 (exposed to the host's ray-gen shader).
    VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descPool      = VK_NULL_HANDLE;
    VkDescriptorSet       m_descSet       = VK_NULL_HANDLE;

    // Host-visible staging buffer for debug readback of the atomic counters.
    // Used by Task 9 smoke test; can be removed later.
    Buffer                m_countReadback;
    void*                 m_countReadbackMapped = nullptr;
    bool                  m_readbackRequested   = false;

    // MLP weight buffer layout (computed once at init).
    MlpLayout             m_mlpLayout;

    // NRC cache-contribution image: float4 per pixel. Written by the
    // cache-query compute pass. After §1 (same-frame dispatch), this is read
    // by the compose pass within the same frame — not by next frame's ray-gen.
    Buffer                m_contribBuffer;

    // §1 path-prefix buffer: float4 per pixel. Written by ray-gen with the
    // frame's direct-lighting prefix (pre-MLP-tail); read by the compose pass.
    Buffer                m_pathPrefixBuffer;

    // Cache-query compute pipeline (Task 10e).
    VkPipeline            m_cacheQueryPipeline     = VK_NULL_HANDLE;
    VkPipelineLayout      m_cacheQueryLayout       = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_cacheQueryDescLayout   = VK_NULL_HANDLE;
    VkDescriptorPool      m_cacheQueryDescPool     = VK_NULL_HANDLE;
    VkDescriptorSet       m_cacheQueryDescSet      = VK_NULL_HANDLE;

    // §1 compose compute pipeline. The descriptor set has a STORAGE_IMAGE at
    // binding 2 whose view is updated per-frame by setOutImageView() — the
    // gbuffer eResultImage view isn't available at NrcPass::init time.
    VkPipeline            m_composePipeline        = VK_NULL_HANDLE;
    VkPipelineLayout      m_composeLayout          = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_composeDescLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      m_composeDescPool        = VK_NULL_HANDLE;
    VkDescriptorSet       m_composeDescSet         = VK_NULL_HANDLE;

    VkImageView           m_outImageView           = VK_NULL_HANDLE;
    VkImageView           m_boundComposeImageView  = VK_NULL_HANDLE;  // last view written into compose descriptor
    uint32_t              m_accumTotalSamples      = 0;
    uint32_t              m_accumNumSamples        = 1;
    bool                  m_accumFirstFrameOrReset = true;

    // §1 bugfix: actual per-frame render size, set by setRenderSize() from the
    // host each frame (the renderer's gbuffer resolution). If 0, dispatches
    // fall back to m_params.frameWidth/frameHeight (set by init).
    uint32_t              m_renderWidth            = 0;
    uint32_t              m_renderHeight           = 0;

    // Training-targets buffer (Task 11b): float4 per training record.
    Buffer                m_trainingTargets;

    // Record-completion pipeline (Task 11b).
    VkPipeline            m_recordCompletePipeline   = VK_NULL_HANDLE;
    VkPipelineLayout      m_recordCompleteLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_recordCompleteDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_recordCompleteDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_recordCompleteDescSet    = VK_NULL_HANDLE;

    // Training pipeline (Task 11c).
    VkPipeline            m_trainPipeline            = VK_NULL_HANDLE;
    VkPipelineLayout      m_trainLayout              = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_trainDescLayout          = VK_NULL_HANDLE;
    VkDescriptorPool      m_trainDescPool            = VK_NULL_HANDLE;
    VkDescriptorSet       m_trainDescSet             = VK_NULL_HANDLE;

    // Optimizer pipeline (Task 11d).
    VkPipeline            m_optimizerPipeline        = VK_NULL_HANDLE;
    VkPipelineLayout      m_optimizerLayout          = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_optimizerDescLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      m_optimizerDescPool        = VK_NULL_HANDLE;
    VkDescriptorSet       m_optimizerDescSet         = VK_NULL_HANDLE;

    // Adam step counter (for bias correction).
    uint32_t              m_adamStep = 0;

    // Lock freezes optimizer updates while
    // still allowing cache-query/inference to use the last trained weights.
    bool                  m_trainingLocked        = false;
    bool                  m_trainOneFrameRequested = false;
    bool                  m_retrainRequested      = false;
    bool                  m_useEmaWeights         = false;
    uint32_t              m_retrainGeneration     = 0u;

    // §3 (Task 3): P-controller state.
    //
    // m_trainingRayThreshold is compared against the full 32-bit ray-gen
    // hash: threshold = p · 2^32 where p = trainingPathsTarget /
    // (frameWidth·frameHeight). Initial value is set in init() once the
    // render size is known; updated each frame after warmup based on the
    // previous frame's trainingCount readback.
    uint32_t              m_trainingRayThreshold   = 0u;
    // Last-frame's actual training-record emission count, read from the
    // host-visible readback staging buffer. Plumbed into the train shader as
    // pc.recordCount so the LCG shuffle modulo uses the real record pool.
    // One frame of staleness is acceptable — the P-controller is slow.
    uint32_t              m_lastKnownRecordCount   = 0u;
    // Monotonic frame counter incremented by recordFrame(). Used to gate the
    // P-controller during the first 128 frames (spec §6 B28 warmup).
    uint32_t              m_frameIdx               = 0u;
    // §3 (Task 3): which Adam step (0..adamStepsPerFrame-1) we're about to
    // dispatch. Set by recordFramePost's Adam loop before each
    // dispatchTrain() call so the LCG shuffle picks a different slice.
    uint32_t              m_trainStepIdx           = 0u;

    // Host-visible staging buffer for the initial row-major weight upload.
    // Held alive until first frame; freed after initializeWeightsFirstFrame.
    Buffer                m_weightStaging;
    void*                 m_weightStagingMapped = nullptr;
    bool                  m_weightsInitialized   = false;

public:
    // Accessors the cache-query / training passes need at dispatch time.
    VkBuffer              primaryWeightsBuffer() const { return m_primaryWeights.handle; }
    VkBuffer              emaWeightsBuffer()     const { return m_emaWeights.handle; }
    VkBuffer              gradientsBuffer()      const { return m_gradients.handle; }
    VkBuffer              queryRecordsBuffer()   const { return m_queryRecords.handle; }
    VkBuffer              queryCountBuffer()     const { return m_queryCount.handle; }
    VkBuffer              contribBuffer()        const { return m_contribBuffer.handle; }
    const MlpLayout&      mlpLayout()            const { return m_mlpLayout; }

    // Task 10: enable/disable NRC at runtime. When disabled, the ray-gen
    // shader skips emission (via push constant) and the cache-query pass is
    // not dispatched.
    void setEnabled(bool v) { m_enabled = v; }
    bool enabled() const    { return m_enabled; }

    // Training controls.
    void setTrainingLocked(bool v) { m_trainingLocked = v; }
    bool trainingLocked() const { return m_trainingLocked; }
    void setUseEmaWeights(bool v) {
        m_useEmaWeights = v;
        updateInferenceWeightDescriptors();
    }
    bool useEmaWeights() const { return m_useEmaWeights; }
    void requestTrainOneFrame() { m_trainOneFrameRequested = true; }
    void requestRetrain() { m_retrainRequested = true; }

    // §3 (Task 3): host-driven training-ray selection threshold. The ray-gen
    // shader hashes (pixelIdx, frameIdx) to a full 32-bit value and emits a
    // training path when hash < threshold. The P-controller updates this
    // each frame (post-warmup) to drive emitted records toward
    // trainingRecordsPerFrame. Initial value is calibrated for shader-ball at
    // 1280×720 targeting ~26k training paths (~60k records).
    uint32_t trainingRayThreshold() const {
        return (m_trainingLocked && !m_trainOneFrameRequested) ? 0u : m_trainingRayThreshold;
    }

    // §3 (Task 3): the ray-gen shader now reads kTrainingCapacity from the
    // path-tracer PC. Return the current capacity (== trainingRecordsPerFrame).
    uint32_t trainingCapacity() const { return m_params.trainingRecordsPerFrame; }

    // Query-record capacity in records. Ray-gen emits at most one query per
    // pixel when NRC is constrained to 1 spp, but this guard prevents a bad
    // configuration from writing beyond the query buffer.
    uint32_t queryCapacity() const { return m_params.frameWidth * m_params.frameHeight; }
};

}  // namespace nrc
