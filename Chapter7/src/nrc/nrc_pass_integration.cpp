#include "nrc/nrc_pass_integration.h"
#include "nrc/nrc_pass.h"

#include <memory>

namespace {
std::unique_ptr<nrc::NrcPass> g_pass;
}  // namespace

extern "C" {

void nrc_pass_init(VkDevice device, VkPhysicalDevice physDev,
                   uint32_t frameWidth, uint32_t frameHeight) {
    g_pass = std::make_unique<nrc::NrcPass>();
    nrc::NrcParams params;
    params.frameWidth  = frameWidth;
    params.frameHeight = frameHeight;
    g_pass->init(device, physDev, params);
}

void nrc_pass_destroy(void) {
    if (g_pass) {
        g_pass->destroy();
        g_pass.reset();
    }
}

void nrc_pass_record_frame(VkCommandBuffer cmd, uint32_t frameIdx) {
    if (g_pass) g_pass->recordFrame(cmd, frameIdx);
}

void nrc_pass_record_frame_post(VkCommandBuffer cmd, uint32_t frameIdx) {
    if (g_pass) g_pass->recordFramePost(cmd, frameIdx);
}

void nrc_pass_request_readback(void) {
    if (g_pass) g_pass->requestReadback();
}

void nrc_pass_readback_counts(uint32_t* trainingOut, uint32_t* queriesOut) {
    if (!g_pass) return;
    auto c = g_pass->readbackCounts();
    if (trainingOut) *trainingOut = c.training;
    if (queriesOut)  *queriesOut  = c.queries;
}

void nrc_pass_set_enabled(int onFlag) {
    if (g_pass) g_pass->setEnabled(onFlag != 0);
}

int nrc_pass_enabled(void) {
    return (g_pass && g_pass->enabled()) ? 1 : 0;
}

void nrc_pass_set_training_locked(int lockFlag) {
    if (g_pass) g_pass->setTrainingLocked(lockFlag != 0);
}

int nrc_pass_training_locked(void) {
    return (g_pass && g_pass->trainingLocked()) ? 1 : 0;
}

void nrc_pass_set_use_ema_weights(int useEmaFlag) {
    if (g_pass) g_pass->setUseEmaWeights(useEmaFlag != 0);
}

int nrc_pass_use_ema_weights(void) {
    return (g_pass && g_pass->useEmaWeights()) ? 1 : 0;
}

void nrc_pass_train_one_frame(void) {
    if (g_pass) g_pass->requestTrainOneFrame();
}

void nrc_pass_retrain(void) {
    if (g_pass) g_pass->requestRetrain();
}

VkDescriptorSet nrc_pass_descriptor_set(void) {
    return g_pass ? g_pass->descriptorSet() : VK_NULL_HANDLE;
}

VkDescriptorSetLayout nrc_pass_descriptor_set_layout(void) {
    return g_pass ? g_pass->descriptorSetLayout() : VK_NULL_HANDLE;
}

void nrc_pass_set_out_image_view(VkImageView view) {
    if (g_pass) g_pass->setOutImageView(view);
}

void nrc_pass_set_accumulation_params(uint32_t totalSamples,
                                      uint32_t numSamples,
                                      int      firstFrameOrReset) {
    if (g_pass) g_pass->setAccumulationParams(totalSamples, numSamples,
                                              firstFrameOrReset != 0);
}

void nrc_pass_set_render_size(uint32_t width, uint32_t height) {
    if (g_pass) g_pass->setRenderSize(width, height);
}

uint32_t nrc_pass_training_ray_threshold(void) {
    return g_pass ? g_pass->trainingRayThreshold() : 0u;
}

uint32_t nrc_pass_training_capacity(void) {
    return g_pass ? g_pass->trainingCapacity() : 0u;
}

uint32_t nrc_pass_query_capacity(void) {
    return g_pass ? g_pass->queryCapacity() : 0u;
}

}  // extern "C"
