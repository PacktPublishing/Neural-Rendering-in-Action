// C-shim that exposes NrcPass to the forked vk_gltf_renderer host.
//
// The host is a C++20 codebase, but its renderer_pathtracer.cpp is a large
// upstream file we want to perturb minimally. We expose plain-C entry points
// so the integration diff in the host stays small and free of C++ template/
// header dependencies from nrc_pass.h.

#pragma once

// See nrc_pass.h for the volk/vulkan header-selection rationale.
#if defined(VK_NO_PROTOTYPES)
  #include <volk.h>
#else
  #include <vulkan/vulkan.h>
#endif
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Called once after the Vulkan device is created.
// frameWidth/frameHeight should match the render target the path tracer
// dispatches at; they size the query buffer.
void nrc_pass_init(VkDevice device, VkPhysicalDevice physDev,
                   uint32_t frameWidth, uint32_t frameHeight);

// Called once before the Vulkan device is destroyed.
void nrc_pass_destroy(void);

// Called every frame, from the host's command-buffer recording function,
// before the path tracer dispatch.
void nrc_pass_record_frame(VkCommandBuffer cmd, uint32_t frameIdx);

// Called every frame, after the path tracer dispatch has been recorded.
void nrc_pass_record_frame_post(VkCommandBuffer cmd, uint32_t frameIdx);

// Debug: request a copy of the atomic counters (training/query) to a host-
// visible buffer at the next recordFramePost. Call this once before the
// frame whose counts you want to read.
void nrc_pass_request_readback(void);

// After the frame's queue submit has completed (queue-waited), read back the
// counts. trainingOut and queriesOut may be NULL.
void nrc_pass_readback_counts(uint32_t* trainingOut, uint32_t* queriesOut);

// NRC on/off.
void nrc_pass_set_enabled(int onFlag);  // 0 = off, nonzero = on
int  nrc_pass_enabled(void);             // 1 if enabled, 0 if not (or not initialized)

// Lock freezes weight updates while cache
// inference keeps using the current weights; Train 1 Frame runs one optimizer
// frame while locked; Retrain resets weights/optimizer state on the next frame.
void nrc_pass_set_training_locked(int lockFlag);
int  nrc_pass_training_locked(void);
void nrc_pass_set_use_ema_weights(int useEmaFlag);
int  nrc_pass_use_ema_weights(void);
void nrc_pass_train_one_frame(void);
void nrc_pass_retrain(void);

// Returns the descriptor set the host should bind at set index 3 for its
// ray-gen shader so the path tracer can write training and query records.
// Returns VK_NULL_HANDLE if nrc_pass_init() has not been called yet.
VkDescriptorSet nrc_pass_descriptor_set(void);

// Returns the matching descriptor set layout so the host can build its
// pipeline layout. Populated by a later task.
VkDescriptorSetLayout nrc_pass_descriptor_set_layout(void);

// §1: the host must tell NrcPass which output image view the compose pass
// should write the final pixel into. Typically the renderer's eResultImage
// gbuffer view (VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_GENERAL).
// Call each frame before nrc_pass_record_frame_post.
void nrc_pass_set_out_image_view(VkImageView view);

// §1: tell NrcPass the current frame's sample-accumulation parameters so
// the compose pass can replicate the host's accumulation blend.
//   totalSamples      — samples already accumulated before this frame
//   numSamples        — samples being taken this frame (usually 1)
//   firstFrameOrReset — nonzero when the result image has no history to
//                       blend with (first frame, DLSS temporal reset, etc.)
void nrc_pass_set_accumulation_params(uint32_t totalSamples,
                                      uint32_t numSamples,
                                      int      firstFrameOrReset);

// Call each frame before nrc_pass_record_frame_post so NRC's compose +
// cache-query passes use the right stride for g_pathPrefix / g_nrcContrib
// indexing when the renderer's gbuffer is not 1920x1080.
void nrc_pass_set_render_size(uint32_t width, uint32_t height);

// §3 (Task 3): current training-ray selection threshold. The ray-gen shader
// treats (pixel,frame) hashed < threshold as a training path. The P-controller
// inside NrcPass updates this each frame (post-128-frame warmup) to drive
// emitted records toward the 65k pool target.
uint32_t nrc_pass_training_ray_threshold(void);

// §3 (Task 3): current record-buffer capacity in records (== 65536 today).
// Ray-gen pushes this in its PC so nrcEmitTrainingStack can drop batches
// that would overflow the pool.
uint32_t nrc_pass_training_capacity(void);

// Current query-record capacity in records. Used by ray-gen as a final
// write-bound guard for g_queryRecords.
uint32_t nrc_pass_query_capacity(void);

#ifdef __cplusplus
}  // extern "C"
#endif
