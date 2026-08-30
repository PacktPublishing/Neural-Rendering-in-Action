// Only compile this test when differentiable rendering is enabled.
#if defined(ENABLE_DIFFERENTIABLE_RENDERING)

#include <AuroraTestHelpers.h>
#include <TestHelpers.h>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

// stb_image_write for saving debug images.
#pragma warning(push)
#pragma warning(disable : 4996)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#pragma warning(pop)

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace Aurora;

// ============================================================
// Shared test configuration
//
// Resolution and sample counts for every differentiable rendering test below.
// Change them here; no test declares its own. Higher values cost time roughly
// linearly in width * height * samples * steps.
// ============================================================

// Render resolution used by all tests.
static constexpr int kWidth  = 1024;
static constexpr int kHeight = 1024;

// Samples per pixel for tests that need a reasonably converged image.
static constexpr int kTestSpp = 16;

// Samples per pixel for the Cornell box. It is lit only by indirect bounces
// from an emissive ceiling panel, so it needs far more samples than the other
// scenes before the loss is stable enough to optimize against.
static constexpr int kTestSppIndirect = 1000;

// Samples per pixel for the numerical gradient checks. This must stay at 1:
// GradientAccumShader replays the path records of the LAST sample only, so at
// one sample the replayed path and the rendered image come from the same
// sample and the AD/FD comparison is exact. Raising it makes those tests
// statistical rather than exact
static constexpr int kGradCheckSpp = 1;

namespace
{

// ---- Debug image helpers ----

// Save a Float_RGBA buffer as a PNG (tone-mapped to [0,1] via clamp).
static void saveFloatRGBAasPNG(
    const std::string& path, const float* pPixels, int width, int height)
{
    std::vector<uint8_t> ldr(width * height * 4);
    for (int i = 0; i < width * height * 4; i++)
    {
        float v  = std::max(0.0f, std::min(1.0f, pPixels[i]));
        ldr[i]   = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
    stbi_write_png(path.c_str(), width, height, 4, ldr.data(), width * 4);
}

// Save a per-pixel diff image: |target - rendered| * 4 (amplified for visibility).
static void saveDiffPNG(const std::string& path, const float* pTarget, const float* pRendered,
    int width, int height)
{
    std::vector<uint8_t> diff(width * height * 4);
    for (int i = 0; i < width * height * 4; i++)
    {
        float d  = std::abs(pTarget[i] - pRendered[i]) * 4.0f;
        d        = std::max(0.0f, std::min(1.0f, d));
        diff[i]  = static_cast<uint8_t>(d * 255.0f + 0.5f);
    }
    stbi_write_png(path.c_str(), width, height, 4, diff.data(), width * 4);
}

// Save a Mitsuba-3-style signed gradient image.
//
// For each pixel the loss gradient is  lossGrad = 2*(rendered - target).
// The image maps this to a diverging red-gray-blue colormap:
//   positive gradient  → warm (red/orange)
//   zero gradient      → neutral gray
//   negative gradient  → cool (blue/cyan)
//
// If fixedScale > 0, that value is used as the normalization denominator so
// multiple frames share the same color mapping and the image genuinely fades
// toward gray as gradients shrink.  When fixedScale <= 0 (default), the peak
// magnitude of this frame is used (auto-scale).
//
// *outMaxAbs (optional): receives the computed peak |grad| for this frame,
// useful for capturing the step-0 scale and reusing it for later steps.
//
// Background pixels (alpha < 0.5) are left black.
static void saveGradientVisPNG(const std::string& path,
    const float* pRendered, const float* pTarget,
    int width, int height,
    float fixedScale = -1.0f, float* outMaxAbs = nullptr)
{
    const int N = width * height;

    // Compute per-pixel loss gradient and find the peak magnitude.
    std::vector<float> grad(N * 3);
    float maxAbs = 1e-8f;
    for (int i = 0; i < N; i++)
    {
        if (pRendered[i * 4 + 3] < 0.5f)
        {
            grad[i * 3 + 0] = grad[i * 3 + 1] = grad[i * 3 + 2] = 0.0f;
            continue;
        }
        for (int c = 0; c < 3; c++)
        {
            float g = 2.0f * (pRendered[i * 4 + c] - pTarget[i * 4 + c]);
            grad[i * 3 + c] = g;
            maxAbs = std::max(maxAbs, std::abs(g));
        }
    }

    if (outMaxAbs)
        *outMaxAbs = maxAbs;

    float scale = (fixedScale > 0.0f) ? fixedScale : maxAbs;

    // Map to diverging colormap: blue (−1) ← gray (0) → red (+1).
    std::vector<uint8_t> ldr(N * 4);
    for (int i = 0; i < N; i++)
    {
        if (pRendered[i * 4 + 3] < 0.5f)
        {
            ldr[i * 4 + 0] = ldr[i * 4 + 1] = ldr[i * 4 + 2] = 0;
            ldr[i * 4 + 3] = 255;
            continue;
        }
        for (int c = 0; c < 3; c++)
        {
            float t = grad[i * 3 + c] / scale; // in [-1, +1] (clamped below)
            float v = 0.5f + 0.5f * t;           // in [ 0,  1]
            ldr[i * 4 + c] = static_cast<uint8_t>(std::max(0.0f, std::min(1.0f, v)) * 255.0f + 0.5f);
        }
        ldr[i * 4 + 3] = 255;
    }
    stbi_write_png(path.c_str(), width, height, 4, ldr.data(), width * 4);
}

// ---- Test fixture ----

class DiffRenderingTest : public TestHelpers::FixtureBase
{
public:
    DiffRenderingTest() {}
    ~DiffRenderingTest() {}
};

// ============================================================
// Gradient sign test (emission-based)
//
// Using emission instead of base_color for the gradient sign test since its easier to control and test.
//   base_color is modulated by lighting (environment + direct light). 
//   Emission is purely additive: shadeEmission() = material.emission * material.emissionColor.
//   It is completely independent of lighting, so:
//     target  = emission (0,0,1) → pure blue pixels
//     rendered = emission (1,0,0) → pure red pixels
//   The difference is clean and the gradient signs are unambiguous.
//
// Procedure:
//   1. Render teapot with emissive BLUE material (emissionColor=(0,0,1), emission=2) into
//      a Float_RGBA buffer. This is the target image.
//   2. Change to emissive RED (emissionColor=(1,0,0), emission=2).
//   3. Call setDiffTargetImage() with the blue render.
//   4. Render again — backward pass computes dL/d_material.
//   5. Check gradient signs:
//      - grad[emissionColor.r] > 0  (rendered red > target red=0)
//      - grad[emissionColor.b] < 0  (rendered blue=0 < target blue=1)
//
// Debug images are saved to ./OutputImages/DiffRenderDebug/:
//   target_blue.png   — the blue reference render
//   rendered_red.png  — the red perturbed render (Float_RGBA, tone-mapped)
//   diff.png          — |target - rendered| * 4 (amplified)
// ============================================================
TEST_P(DiffRenderingTest, TestGradientSign)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }


    // Create debug output directory.
    const std::string kDebugDir = "./OutputImages/DiffRenderDebug";
    std::filesystem::create_directories(kDebugDir);

    // ---- Setup ----
    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);

    // Position camera so the teapot fills the frame (same framing used in TestRendererGroundPlane).
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    Path geomPath = createTeapotGeometry(*pScene);

    const Path kMaterialPath = "DiffTestMaterial";
    pScene->setMaterialType(kMaterialPath);

    const Path kInstancePath = "DiffTestInstance";
    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance(kInstancePath, geomPath, instProps));

    // ---- Render with emissive BLUE → capture as Float_RGBA target ----
    {
        Properties blueProps;
        blueProps["emission_color"] = vec3(0.0f, 0.0f, 1.0f);
        blueProps["emission"]       = 2.0f;
        blueProps["base_color"]     = vec3(0.0f, 0.0f, 0.0f); // suppress base diffuse
        pScene->setMaterialProperties(kMaterialPath, blueProps);
    }

    IRenderBufferPtr pRefBuffer =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pRefBuffer } });

    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    size_t refStride = 0;
    const float* pRefPixels =
        reinterpret_cast<const float*>(pRefBuffer->data(refStride, true));
    ASSERT_NE(pRefPixels, nullptr);

    // Save target image.
    saveFloatRGBAasPNG(kDebugDir + "/target_blue.png", pRefPixels, kWidth, kHeight);

    // Log center pixel of target.
    const float* pCenterRef = pRefPixels + (kHeight / 2 * kWidth + kWidth / 2) * 4;
    AU_INFO("DiffRenderTest: target center pixel RGBA = (%.3f, %.3f, %.3f, %.3f)",
        pCenterRef[0], pCenterRef[1], pCenterRef[2], pCenterRef[3]);

    // ---- Change to emissive RED ----
    {
        Properties redProps;
        redProps["emission_color"] = vec3(1.0f, 0.0f, 0.0f);
        redProps["emission"]       = 2.0f;
        redProps["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kMaterialPath, redProps);
    }

    // Use a second Float_RGBA buffer for the perturbed render so we can save it and diff it.
    IRenderBufferPtr pRedBuffer =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pRedBuffer } });

    // ---- Set the blue render as the diff target ----
    IImage::InitData targetInitData;
    targetInitData.pImageData = pRefPixels;
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "DiffTestTargetImage";
    pRenderer->setDiffTargetImage(targetInitData);

    // ---- Render with RED (backward pass runs automatically) ----
    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    // Read back the red render and save debug images.
    size_t redStride = 0;
    const float* pRedPixels =
        reinterpret_cast<const float*>(pRedBuffer->data(redStride, true));
    ASSERT_NE(pRedPixels, nullptr);

    saveFloatRGBAasPNG(kDebugDir + "/rendered_red.png", pRedPixels, kWidth, kHeight);
    saveDiffPNG(kDebugDir + "/diff.png", pRefPixels, pRedPixels, kWidth, kHeight);

    const float* pCenterRed = pRedPixels + (kHeight / 2 * kWidth + kWidth / 2) * 4;
    AU_INFO("DiffRenderTest: rendered center pixel RGBA = (%.3f, %.3f, %.3f, %.3f)",
        pCenterRed[0], pCenterRed[1], pCenterRed[2], pCenterRed[3]);

    AU_INFO("DiffRenderTest: debug images saved to %s", kDebugDir.c_str());

    // ---- Read back gradients and check signs ----
    vector<float> grads = pRenderer->getMaterialGradients();

    ASSERT_EQ(grads.size(), 15u)
        << "getMaterialGradients() returned empty — no valid surface hits. "
           "Check that the teapot is visible.";

    // Gradient layout (MATERIAL_GRAD_STRIDE = 16: 15 gradient floats + coverage flag):
    //   [0-2]  baseColor.xyz
    //   [3]    specularRoughness
    //   [4]    metalness
    //   [5-7]  emissionColor.xyz   ← we check these
    //   [8]    emission
    //   [9]    specular
    //   [10-12] specularColor.xyz
    //   [13]   specularIOR
    //   [14]   specularAnisotropy
    AU_INFO("GradientSign — mean material gradients:");
    AU_INFO("  baseColor:         (%.6f, %.6f, %.6f)", grads[0], grads[1], grads[2]);
    AU_INFO("  specularRoughness: %.6f", grads[3]);
    AU_INFO("  metalness:         %.6f", grads[4]);
    AU_INFO("  emissionColor:     (%.6f, %.6f, %.6f)", grads[5], grads[6], grads[7]);
    AU_INFO("  emission:          %.6f", grads[8]);
    AU_INFO("  specular:          %.6f", grads[9]);
    AU_INFO("  specularColor:     (%.6f, %.6f, %.6f)", grads[10], grads[11], grads[12]);
    AU_INFO("  specularIOR:       %.6f", grads[13]);
    AU_INFO("  specularAnisotropy:%.6f", grads[14]);

    const float gradEmitR = grads[5]; // emissionColor.r
    const float gradEmitB = grads[7]; // emissionColor.b

    // rendered_red > target_red (0) → dL/d_emissionColor.r > 0
    EXPECT_GT(gradEmitR, 0.0f)
        << "Expected grad[emissionColor.r] > 0 (rendered red=" << pCenterRed[0]
        << " > target red=" << pCenterRef[0] << "), but got " << gradEmitR;

    // rendered_blue (0) < target_blue (1) → dL/d_emissionColor.b < 0
    EXPECT_LT(gradEmitB, 0.0f)
        << "Expected grad[emissionColor.b] < 0 (rendered blue=" << pCenterRed[2]
        << " < target blue=" << pCenterRef[2] << "), but got " << gradEmitB;
}

// ============================================================
// Emission numerical gradient check (end-to-end correctness)
//
// Verifies that the AD gradients from GradientAccumShader match finite differences
// computed by rendering twice with a small parameter perturbation.
//
// Why emission only (not base_color):
//   base_color requires lights; stochastic path tracing noise corrupts FD estimates.
//   Emission is deterministic (no light sampling), so L(p) and L(p+eps) are noise-free
//   and the FD estimate is exact.
//
// Method for each emission parameter p:
//   1. Render with value p   → compute mean per-pixel L2 loss L(p).
//   2. Render with value p+eps → compute mean per-pixel L2 loss L(p+eps).
//   3. FD = (L(p+eps) - L(p)) / eps
//   4. AD = getMaterialGradients()[field_index]   (already mean per valid pixel)
//   5. Check: |AD - FD| / |FD| < kRelTol
//
// Scene: teapot, no lights, emissive material.
// Target: emissionColor=(0,0,1), emission=2  (blue).
// Perturbed: one parameter shifted by eps.
// ============================================================
TEST_P(DiffRenderingTest, TestEmissionGradCheck)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr float kEps     = 0.01f;  // small perturbation; emission is deterministic so no MC noise
    constexpr float kRelTol  = 0.05f;  // 5% relative tolerance

    // Create debug output directory.
    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/EmissionGradCheck";
    std::filesystem::create_directories(kDebugDir);

    // Surface-hit pixel mask: populated after the base render.
    // computeMeanLoss uses this to match the pixel set used by GradientAccumShader
    // (isValid pixels only), so FD and AD are normalized by the same count.
    // We use the alpha channel (alpha=1 for surface hits, alpha=0 for background) as the mask.
    // isAlphaEnabled must be true for this to work.
    vector<float> basePixelMask;

    // ---- Helper: compute mean per-pixel L2 loss between two Float_RGBA buffers ----
    // Only counts pixels where the BASE render has alpha > 0.5 (surface hits).
    // Must match the pixel mask used by GradientAccumShader (isValid pixels only).
    auto computeMeanLoss = [&](const float* pRendered, const float* pTarget,
                                int width, int height) -> double
    {
        double sum        = 0.0;
        int    validCount = 0;
        for (int i = 0; i < width * height; i++)
        {
            // Use the base render alpha as the pixel mask (alpha=1 for surface hits).
            // This matches GradientAccumShader's isValid mask exactly.
            float maskAlpha = basePixelMask.empty() ? pRendered[i * 4 + 3] : basePixelMask[i * 4 + 3];
            if (maskAlpha < 0.5f)
                continue;
            validCount++;
            for (int c = 0; c < 3; c++)
            {
                float diff = pRendered[i * 4 + c] - pTarget[i * 4 + c];
                sum += diff * diff;
            }
        }
        return validCount > 0 ? sum / validCount : 0.0;
    };

    // ---- Setup ----
    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    // Disable gamma correction so the render buffer receives raw HDR float values.
    // With gamma correction enabled (the default on Windows), the post-processing shader
    // calls saturate(color) which clamps values to [0,1]. Emission values > 1.0 would be
    // clamped, making perturbed renders identical to the base render (FD = 0).
    // The backward pass (GradientAccumShader) reads from gResult (the direct HDR texture,
    // before post-processing), so disabling gamma correction makes the FD and AD consistent.
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);

    // Enable alpha output so the render buffer's alpha channel carries the surface-hit mask
    // (alpha=1 for surface hits, alpha=0 for background misses). This lets computeMeanLoss
    // use the same pixel set as GradientAccumShader (isValid pixels only).
    pRenderer->options().setBoolean("alphaEnabled", true);

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    Path geomPath = createTeapotGeometry(*pScene);

    const Path kMaterialPath = "DiffTestMaterial";
    pScene->setMaterialType(kMaterialPath);

    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance("DiffTestInstance", geomPath, instProps));

    // Base material: emissive blue, no diffuse, no lights.
    const vec3  kBaseEmitColor = vec3(1.0f, 0.5f, 0.2f); // non-trivial values for all channels
    const float kBaseEmission  = 2.0f;

    // Target image: render with a different color so loss is non-zero.
    const vec3  kTargetEmitColor = vec3(0.0f, 0.0f, 1.0f);
    const float kTargetEmission  = 2.0f;

    // ---- Render target image ----
    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    {
        Properties p;
        p["emission_color"] = kTargetEmitColor;
        p["emission"]       = kTargetEmission;
        p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kMaterialPath, p);
    }
    pRenderer->render(0, 1);
    pRenderer->waitForTask();
    size_t targetStride = 0;
    const float* pTargetPixels =
        reinterpret_cast<const float*>(pTargetBuf->data(targetStride, true));
    ASSERT_NE(pTargetPixels, nullptr);

    // Copy target pixels to CPU so we can compute loss after each render.
    vector<float> targetPixels(pTargetPixels, pTargetPixels + kWidth * kHeight * 4);
    saveFloatRGBAasPNG(kDebugDir + "/target.png", targetPixels.data(), kWidth, kHeight);

    // Set the diff target image (used by the backward pass).
    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "DiffEmissionTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    // Parameters to test: {name, grad_index, perturbed emitColor, perturbed emission}
    struct ParamTest
    {
        const char* name;
        int         gradIdx;
        vec3        emitColor;
        float       emission;
    };

    // We test each emission parameter independently by perturbing one at a time.
    vector<ParamTest> params = {
        { "emissionColor.r", 5, vec3(kBaseEmitColor.x + kEps, kBaseEmitColor.y, kBaseEmitColor.z), kBaseEmission },
        { "emissionColor.g", 6, vec3(kBaseEmitColor.x, kBaseEmitColor.y + kEps, kBaseEmitColor.z), kBaseEmission },
        { "emissionColor.b", 7, vec3(kBaseEmitColor.x, kBaseEmitColor.y, kBaseEmitColor.z + kEps), kBaseEmission },
        { "emission",        8, kBaseEmitColor,                                                     kBaseEmission + kEps },
    };

    // ---- Render base (p) and get AD gradients ----
    // Use a dedicated buffer so _hasPendingData is fresh for readback.
    IRenderBufferPtr pBaseBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pBaseBuf } });
    {
        Properties p;
        p["emission_color"] = kBaseEmitColor;
        p["emission"]       = kBaseEmission;
        p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kMaterialPath, p);
    }
    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    size_t baseStride = 0;
    const float* pBasePixels =
        reinterpret_cast<const float*>(pBaseBuf->data(baseStride, true));
    ASSERT_NE(pBasePixels, nullptr);
    vector<float> basePixels(pBasePixels, pBasePixels + kWidth * kHeight * 4);
    // Populate the pixel mask used by computeMeanLoss (must happen before lossBase call).
    basePixelMask = basePixels;
    saveFloatRGBAasPNG(kDebugDir + "/base_render.png", basePixels.data(), kWidth, kHeight);
    saveDiffPNG(kDebugDir + "/base_vs_target_diff.png",
        targetPixels.data(), basePixels.data(), kWidth, kHeight);

    double lossBase = computeMeanLoss(basePixels.data(), targetPixels.data(), kWidth, kHeight);

    vector<float> adGrads = pRenderer->getMaterialGradients();
    ASSERT_EQ(adGrads.size(), 15u) << "No valid surface hits in base render.";

    AU_INFO("EmissionGradCheck — base loss = %.6f", lossBase);

    // ---- For each parameter: render perturbed, compute FD, compare to AD ----
    bool allPassed = true;
    for (const auto& pt : params)
    {
        // Render with p + eps.
        // A new render buffer is created each iteration so _hasPendingData is fresh —
        // reusing the same buffer would return stale cached data after the first data() call.
        IRenderBufferPtr pPertBuf =
            pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
        pRenderer->setTargets({ { AOV::kFinal, pPertBuf } });
        {
            Properties p;
            p["emission_color"] = pt.emitColor;
            p["emission"]       = pt.emission;
            p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
            pScene->setMaterialProperties(kMaterialPath, p);
        }
        pRenderer->render(0, 1);
        pRenderer->waitForTask();

        size_t pertStride = 0;
        const float* pPertPixels =
            reinterpret_cast<const float*>(pPertBuf->data(pertStride, true));
        ASSERT_NE(pPertPixels, nullptr);
        vector<float> pertPixels(pPertPixels, pPertPixels + kWidth * kHeight * 4);

        // Save perturbed render and diff vs base (amplified to show the tiny eps change).
        std::string safeName = std::string(pt.name);
        std::replace(safeName.begin(), safeName.end(), '.', '_');
        saveFloatRGBAasPNG(kDebugDir + "/perturbed_" + safeName + ".png",
            pertPixels.data(), kWidth, kHeight);
        saveDiffPNG(kDebugDir + "/perturbed_" + safeName + "_vs_base.png",
            basePixels.data(), pertPixels.data(), kWidth, kHeight);

        double lossPert = computeMeanLoss(pertPixels.data(), targetPixels.data(), kWidth, kHeight);
        double fd       = (lossPert - lossBase) / kEps;
        double ad       = adGrads[pt.gradIdx];
        double relErr   = (std::abs(fd) > 1e-8) ? std::abs(ad - fd) / std::abs(fd) : std::abs(ad - fd);

        AU_INFO("  %-20s  AD=%+.6f  FD=%+.6f  relErr=%.4f  %s",
            pt.name, ad, fd, relErr, relErr < kRelTol ? "PASS" : "FAIL");

        EXPECT_LT(relErr, kRelTol)
            << "Numerical gradient check FAILED for " << pt.name
            << ": AD=" << ad << " FD=" << fd << " relErr=" << relErr;

        if (relErr >= kRelTol)
            allPassed = false;
    }

    AU_INFO("EmissionGradCheck — %s", allPassed ? "ALL PASSED" : "SOME FAILED");
    AU_INFO("EmissionGradCheck — debug images saved to %s", kDebugDir.c_str());
}

// ============================================================
// base_color numerical gradient checks (BRDF path)
//
// TestEmissionGradCheck validates only the emission path, which bypasses the lighting
// estimator entirely (shadeEmission adds straight to radiance). These tests are
// its mirror image: emission is zero and base_color carries the signal, so the
// gradient must travel through evaluateMaterial and the lighting estimators.
//
// The scene is lit by two independent mechanisms, so each is validated in
// isolation and then combined:
//
//   Environment  — gradient environment only, distant light off
//                  (environment MIS estimator replay).
//   DistantLight — distant light only, environment black
//                  (directional-light estimator replay).
//   SingleBounce — distant light with traceDepth 1
//                  (the per-bounce replay term with no indirect lighting).
//   Combined     — both mechanisms together.
//
// The replay differentiates the recorded shading, not the path sampling itself
// (detached sampling), so scenes dominated by indirect lighting show a lower
// AD/FD ratio; these scenes are direct-dominated and sit near 1.0.
//
// Each logs AD/FD per channel. A systematic scale error in the replayed
// estimator shows up as a ratio that is consistent across channels, which
// distinguishes it from Monte Carlo noise (which scatters around 1.0).
// ============================================================

namespace
{

// Shared body for the base_color finite-difference checks. The caller sets up
// whatever lighting it wants to isolate; this renders target / base / perturbed
// and compares AD against FD for the three base_color channels.
void runBaseColorGradCheck(Aurora::IRenderer* pRenderer, Aurora::IScene* pScene,
    const Aurora::Path& materialPath, const char* label, const std::string& debugDir,
    int width, int height, int spp, float eps, float relTol)
{
    using namespace Aurora;

    const vec3 kBaseColor   = vec3(0.6f, 0.4f, 0.2f);
    const vec3 kTargetColor = vec3(0.2f, 0.5f, 0.7f);

    vector<float> basePixelMask;

    // Mean per-pixel L2 over surface-hit pixels only, matching GradientAccumShader's
    // isValid mask so FD and AD share a denominator.
    auto computeMeanLoss = [&](const float* pRendered, const float* pTarget) -> double
    {
        double sum        = 0.0;
        int    validCount = 0;
        for (int i = 0; i < width * height; i++)
        {
            float maskAlpha = basePixelMask.empty() ? pRendered[i * 4 + 3] : basePixelMask[i * 4 + 3];
            if (maskAlpha < 0.5f)
                continue;
            validCount++;
            for (int c = 0; c < 3; c++)
            {
                float diff = pRendered[i * 4 + c] - pTarget[i * 4 + c];
                sum += diff * diff;
            }
        }
        return validCount > 0 ? sum / validCount : 0.0;
    };

    auto renderWith = [&](const vec3& baseColor) -> vector<float>
    {
        IRenderBufferPtr pBuf = pRenderer->createRenderBuffer(width, height, ImageFormat::Float_RGBA);
        pRenderer->setTargets({ { AOV::kFinal, pBuf } });
        Properties p;
        p["base_color"] = baseColor;
        p["emission"]   = 0.0f;
        p["metalness"]  = 0.0f;
        pScene->setMaterialProperties(materialPath, p);
        pRenderer->render(0, spp);
        pRenderer->waitForTask();
        size_t       stride  = 0;
        const float* pPixels = reinterpret_cast<const float*>(pBuf->data(stride, true));
        EXPECT_NE(pPixels, nullptr);
        if (!pPixels)
            return {};
        return vector<float>(pPixels, pPixels + width * height * 4);
    };

    // ---- Target ----
    vector<float> targetPixels = renderWith(kTargetColor);
    ASSERT_FALSE(targetPixels.empty());
    saveFloatRGBAasPNG(debugDir + "/target.png", targetPixels.data(), width, height);

    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = width;
    targetInitData.height     = height;
    targetInitData.name       = "DiffBaseColorTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    // ---- Base render + AD gradients ----
    vector<float> basePixels = renderWith(kBaseColor);
    ASSERT_FALSE(basePixels.empty());
    basePixelMask = basePixels;
    saveFloatRGBAasPNG(debugDir + "/base.png", basePixels.data(), width, height);

    double        lossBase = computeMeanLoss(basePixels.data(), targetPixels.data());
    vector<float> adGrads  = pRenderer->getMaterialGradients();
    ASSERT_EQ(adGrads.size(), 15u) << "No valid surface hits in base render.";

    AU_INFO("BaseColorGradCheck [%s] — base loss = %.6f (SPP=%d)", label, lossBase, spp);

    struct ParamTest
    {
        const char* name;
        int         gradIdx;
        vec3        baseColor;
    };
    const vector<ParamTest> params = {
        { "baseColor.r", 0, vec3(kBaseColor.x + eps, kBaseColor.y, kBaseColor.z) },
        { "baseColor.g", 1, vec3(kBaseColor.x, kBaseColor.y + eps, kBaseColor.z) },
        { "baseColor.b", 2, vec3(kBaseColor.x, kBaseColor.y, kBaseColor.z + eps) },
    };

    bool allPassed = true;
    for (const auto& pt : params)
    {
        vector<float> pertPixels = renderWith(pt.baseColor);
        ASSERT_FALSE(pertPixels.empty());

        std::string safeName = std::string(pt.name);
        std::replace(safeName.begin(), safeName.end(), '.', '_');
        saveFloatRGBAasPNG(debugDir + "/perturbed_" + safeName + ".png",
            pertPixels.data(), width, height);

        double lossPert = computeMeanLoss(pertPixels.data(), targetPixels.data());
        double fd       = (lossPert - lossBase) / eps;
        double ad       = adGrads[pt.gradIdx];
        double relErr = (std::abs(fd) > 1e-8) ? std::abs(ad - fd) / std::abs(fd) : std::abs(ad - fd);
        double ratio  = (std::abs(fd) > 1e-8) ? ad / fd : 0.0;

        AU_INFO("  [%s] %-14s  AD=%+.6f  FD=%+.6f  AD/FD=%+.4f  relErr=%.4f  %s",
            label, pt.name, ad, fd, ratio, relErr, relErr < relTol ? "PASS" : "FAIL");

        EXPECT_LT(relErr, relTol) << "[" << label << "] AD/FD mismatch for " << pt.name
                                  << ": AD=" << ad << " FD=" << fd << " ratio=" << ratio;
        if (relErr >= relTol)
            allPassed = false;
    }

    AU_INFO("BaseColorGradCheck [%s] — %s", label, allPassed ? "ALL PASSED" : "SOME FAILED");
}

} // namespace

// Tuning for the base_color gradient checks (resolution and SPP come from the
// shared configuration at the top of this file).
static constexpr float kFDEps    = 0.02f;
// 25% tolerance: the replay is subject to the same documented ~15% systematic bias
// as the multi-bounce numerical check (detached sampling: neither the MIS weights
// nor the throughput's own dependence on the material are differentiated), plus
// single-sample replay effects. Measured healthy values sit at 0.03-0.23 relErr.
static constexpr float kFDRelTol = 0.25f;

// Environment lighting only — the path GradientAccumShader does replay.
TEST_P(DiffRenderingTest, TestBaseColorGradCheckEnvironment)
{
    if (!isDirectX() || !backendSupported())
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/BaseColorGradCheck_Environment";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // Distant light off: the gradient environment is the only illumination.
    defaultDistantLight()->values().setFloat(Names::LightProperties::kIntensity, 0.0f);

    Path       geomPath      = createTeapotGeometry(*pScene);
    const Path kMaterialPath = "DiffGradCheckEnvMaterial";
    pScene->setMaterialType(kMaterialPath);
    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance("DiffGradCheckEnvInstance", geomPath, instProps));

    runBaseColorGradCheck(pRenderer.get(), pScene.get(), kMaterialPath, "environment", kDebugDir,
        kWidth, kHeight, kGradCheckSpp, kFDEps, kFDRelTol);
}

// Distant light, restricted to a SINGLE bounce.
//
// With traceDepth = 1 there is no indirect lighting at all, so this validates
// the directional-light replay term (record slots 11/12) in isolation:
// AD/FD ~ 1.0 across all channels.
TEST_P(DiffRenderingTest, TestBaseColorGradCheckSingleBounce)
{
    if (!isDirectX() || !backendSupported())
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/BaseColorGradCheck_SingleBounce";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    // The whole point of this variant: one bounce only.
    pRenderer->options().setInt("traceDepth", 1);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // Black out the environment gradient so only the distant light illuminates.
    const Path kEnvPath = "DiffGradCheckBlackEnv1B";
    pScene->setEnvironmentProperties(kEnvPath,
        { { Names::EnvironmentProperties::kLightTop, vec3(0.0f, 0.0f, 0.0f) },
            { Names::EnvironmentProperties::kLightBottom, vec3(0.0f, 0.0f, 0.0f) } });
    pScene->setEnvironment(kEnvPath);

    defaultDistantLight()->values().setFloat3(
        Names::LightProperties::kDirection, value_ptr(vec3(0, -1, -1)));
    defaultDistantLight()->values().setFloat(Names::LightProperties::kIntensity, 3.0f);

    Path       geomPath      = createTeapotGeometry(*pScene);
    const Path kMaterialPath = "DiffGradCheckSingleBounceMaterial";
    pScene->setMaterialType(kMaterialPath);
    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance("DiffGradCheckSingleBounceInstance", geomPath, instProps));

    runBaseColorGradCheck(pRenderer.get(), pScene.get(), kMaterialPath, "single bounce",
        kDebugDir, kWidth, kHeight, kGradCheckSpp, kFDEps, kFDRelTol);
}

// Distant light only, at the default trace depth.
//
// The light faces the camera side of the teapot, so the direct response
// dominates the image and the recorded-path replay captures nearly all of the
// baseColor response (AD/FD ~ 1.0).
TEST_P(DiffRenderingTest, TestBaseColorGradCheckDistantLight)
{
    if (!isDirectX() || !backendSupported())
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/BaseColorGradCheck_DistantLight";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // Black out the environment gradient so only the distant light illuminates.
    const Path kEnvPath = "DiffGradCheckBlackEnv";
    pScene->setEnvironmentProperties(kEnvPath,
        { { Names::EnvironmentProperties::kLightTop, vec3(0.0f, 0.0f, 0.0f) },
            { Names::EnvironmentProperties::kLightBottom, vec3(0.0f, 0.0f, 0.0f) } });
    pScene->setEnvironment(kEnvPath);

    // The light faces the side of the teapot the camera sees.
    defaultDistantLight()->values().setFloat3(
        Names::LightProperties::kDirection, value_ptr(vec3(0, -1, 1)));
    defaultDistantLight()->values().setFloat(Names::LightProperties::kIntensity, 3.0f);

    Path       geomPath      = createTeapotGeometry(*pScene);
    const Path kMaterialPath = "DiffGradCheckDistantLightMaterial";
    pScene->setMaterialType(kMaterialPath);
    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance("DiffGradCheckDistantLightInstance", geomPath, instProps));

    runBaseColorGradCheck(pRenderer.get(), pScene.get(), kMaterialPath, "distant light",
        kDebugDir, kWidth, kHeight, kGradCheckSpp, kFDEps, kFDRelTol);
}

// Both mechanisms together — the realistic case.
TEST_P(DiffRenderingTest, TestBaseColorGradCheckCombined)
{
    if (!isDirectX() || !backendSupported())
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/BaseColorGradCheck_Combined";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    defaultDistantLight()->values().setFloat3(
        Names::LightProperties::kDirection, value_ptr(vec3(0, -1, -1)));
    defaultDistantLight()->values().setFloat(Names::LightProperties::kIntensity, 3.0f);

    Path       geomPath      = createTeapotGeometry(*pScene);
    const Path kMaterialPath = "DiffGradCheckCombinedMaterial";
    pScene->setMaterialType(kMaterialPath);
    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance("DiffGradCheckCombinedInstance", geomPath, instProps));

    runBaseColorGradCheck(pRenderer.get(), pScene.get(), kMaterialPath, "combined", kDebugDir,
        kWidth, kHeight, kGradCheckSpp, kFDEps, kFDRelTol);
}


// ============================================================
// Emission optimization loop (end-to-end C++)
//
// Demonstrates the full differentiable rendering pipeline in C++:
//   render → getMaterialGradients() → SGD update → repeat
//
// Why emission only:
//   Emission is deterministic (no light sampling), so the loss surface is smooth
//   and gradient descent converges cleanly without MC noise.
//
// Procedure:
//   1. Render teapot with emissive BLUE (target).
//   2. Start with emissive RED material.
//   3. Loop kNumSteps times:
//        a. Render (forward + backward pass runs automatically).
//        b. Read back mean gradients via getMaterialGradients().
//        c. SGD update: param -= lr * grad  (on emissionColor only).
//        d. Clamp emissionColor to [0, 1].
//        e. Record loss.
//   4. Assert loss decreases monotonically for at least kMinMonoSteps consecutive steps.
//   5. Assert final loss < initial loss * kLossReductionFactor.
//
// Pass criteria:
//   - Loss decreases monotonically for >= 10 consecutive steps.
//   - Final loss < 10% of initial loss.
// ============================================================
TEST_P(DiffRenderingTest, TestEmissionOptimizationLoop)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int   kNumSteps           = 10;
    constexpr float kLR                 = 0.05f;  // SGD learning rate
    constexpr int   kMinMonoSteps       = 8;     // minimum consecutive loss-decreasing steps
    constexpr float kLossReductionFactor = 0.10f; // final loss must be < 10% of initial

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/EmissionOptim";
    std::filesystem::create_directories(kDebugDir);

    // ---- Setup ----
    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    Path geomPath = createTeapotGeometry(*pScene);

    const Path kMaterialPath = "DiffOptMaterial";
    pScene->setMaterialType(kMaterialPath);

    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance("DiffOptInstance", geomPath, instProps));

    // ---- Render target: emissive BLUE ----
    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    {
        Properties p;
        p["emission_color"] = vec3(0.0f, 0.0f, 1.0f);
        p["emission"]       = 2.0f;
        p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kMaterialPath, p);
    }
    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    size_t targetStride = 0;
    const float* pTargetRaw =
        reinterpret_cast<const float*>(pTargetBuf->data(targetStride, true));
    ASSERT_NE(pTargetRaw, nullptr);
    vector<float> targetPixels(pTargetRaw, pTargetRaw + kWidth * kHeight * 4);
    saveFloatRGBAasPNG(kDebugDir + "/target.png", targetPixels.data(), kWidth, kHeight);

    // Upload target to GPU for the backward pass.
    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "DiffOptTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    // ---- Helper: compute mean L2 loss over surface-hit pixels (alpha > 0.5) ----
    auto computeLoss = [&](const float* pRendered, const float* pTarget,
                            int width, int height) -> double
    {
        double sum   = 0.0;
        int    count = 0;
        for (int i = 0; i < width * height; i++)
        {
            if (pRendered[i * 4 + 3] < 0.5f)
                continue;
            count++;
            for (int c = 0; c < 3; c++)
            {
                float d = pRendered[i * 4 + c] - pTarget[i * 4 + c];
                sum += d * d;
            }
        }
        return count > 0 ? sum / count : 0.0;
    };

    // ---- Optimization loop ----
    // Current material parameters (emissionColor only; emission scalar fixed at 2).
    vec3  emitColor = vec3(1.0f, 0.0f, 0.0f); // start: RED
    float emission  = 2.0f;
    float gradScale = -1.0f;

    vector<double> losses;
    losses.reserve(kNumSteps);

    for (int step = 0; step < kNumSteps; step++)
    {
        // Apply current material parameters.
        {
            Properties p;
            p["emission_color"] = emitColor;
            p["emission"]       = emission;
            p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
            pScene->setMaterialProperties(kMaterialPath, p);
        }

        // New buffer each step so _hasPendingData is fresh for readback.
        IRenderBufferPtr pOptBuf =
            pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
        pRenderer->setTargets({ { AOV::kFinal, pOptBuf } });

        // Forward + backward pass.
        pRenderer->render(0, 1);
        pRenderer->waitForTask();

        // Read rendered pixels and compute loss.
        size_t optStride = 0;
        const float* pOptRaw =
            reinterpret_cast<const float*>(pOptBuf->data(optStride, true));
        ASSERT_NE(pOptRaw, nullptr);
        double loss = computeLoss(pOptRaw, targetPixels.data(), kWidth, kHeight);
        losses.push_back(loss);

        {
            char fname[64];
            snprintf(fname, sizeof(fname), "/step_%03d.png", step);
            saveFloatRGBAasPNG(kDebugDir + fname, pOptRaw, kWidth, kHeight);

            if (step == 0 || step == kNumSteps / 2 || step == kNumSteps - 1)
            {
                snprintf(fname, sizeof(fname), "/grad_%03d.png", step);
                float frameMax = 0.0f;
                saveGradientVisPNG(kDebugDir + fname,
                    pOptRaw, targetPixels.data(), kWidth, kHeight,
                    gradScale, &frameMax);
                if (step == 0)
                    gradScale = frameMax;
                snprintf(fname, sizeof(fname), "/diff_%03d.png", step);
                saveDiffPNG(kDebugDir + fname,
                    targetPixels.data(), pOptRaw, kWidth, kHeight);
            }
        }

        if (step == kNumSteps - 1)
            saveFloatRGBAasPNG(kDebugDir + "/final.png", pOptRaw, kWidth, kHeight);

        // Read back AD gradients.
        vector<float> grads = pRenderer->getMaterialGradients();
        ASSERT_EQ(grads.size(), 15u) << "No valid surface hits at step " << step;

        // SGD update on emissionColor (indices 5, 6, 7).
        emitColor.x -= kLR * grads[5];
        emitColor.y -= kLR * grads[6];
        emitColor.z -= kLR * grads[7];

        // Clamp to physically valid range.
        emitColor.x = std::max(0.0f, std::min(1.0f, emitColor.x));
        emitColor.y = std::max(0.0f, std::min(1.0f, emitColor.y));
        emitColor.z = std::max(0.0f, std::min(1.0f, emitColor.z));

        
        {
            AU_INFO("  Step %3d: loss=%.6f  emitColor=(%.3f, %.3f, %.3f)",
                step, loss, emitColor.x, emitColor.y, emitColor.z);
        }
    }

    // ---- Check 1: loss decreased monotonically for >= kMinMonoSteps consecutive steps ----
    int maxMonoRun = 1, currentRun = 1;
    for (int i = 1; i < (int)losses.size(); i++)
    {
        if (losses[i] <= losses[i - 1])
        {
            currentRun++;
            maxMonoRun = std::max(maxMonoRun, currentRun);
        }
        else
        {
            currentRun = 1;
        }
    }
    AU_INFO("EmissionOptim — longest monotone decrease: %d steps (need %d)",
        maxMonoRun, kMinMonoSteps);
    EXPECT_GE(maxMonoRun, kMinMonoSteps)
        << "Loss did not decrease monotonically for " << kMinMonoSteps
        << " consecutive steps. Longest run: " << maxMonoRun;

    // ---- Check 2: final loss < kLossReductionFactor * initial loss ----
    double initialLoss = losses.front();
    double finalLoss   = losses.back();
    AU_INFO("EmissionOptim — initial loss=%.6f  final loss=%.6f  ratio=%.4f",
        initialLoss, finalLoss, finalLoss / initialLoss);
    EXPECT_LT(finalLoss, initialLoss * kLossReductionFactor)
        << "Final loss (" << finalLoss << ") is not < "
        << kLossReductionFactor << " * initial loss (" << initialLoss << ")";

    AU_INFO("EmissionOptim — debug images saved to %s", kDebugDir.c_str());
}

// ============================================================
// TestBaseColorOptimizationLoop — Single-bounce base_color optimization
//
// Scene: A single diffuse teapot (metalness=0) illuminated by a distant light
// and the default gradient environment. The teapot's direct appearance depends
// on its base_color through the BRDF.
//
// Target: teapot with blue base_color (0.05, 0.05, 0.9).
// Start:  teapot with red  base_color (0.90, 0.05, 0.05).
//
// The optimizer uses SGD on the base_color gradient (indices 0,1,2) to
// converge the teapot's appearance from red to blue.
//
// Unlike the emission test (TestEmissionOptimizationLoop), this exercises the
// BRDF differentiation path (bwd_diff(evaluateMaterial)), which is the core
// of physically based differentiable rendering.
//
// ============================================================
TEST_P(DiffRenderingTest, TestBaseColorOptimizationLoop)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int   kSPP                = kTestSpp;
    constexpr int   kNumSteps           = 50;
    constexpr float kLR                 = 0.50f;
    constexpr int   kMinMonoSteps       = 8;
    constexpr float kLossReductionFactor = 0.30f;

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/BaseColorOptim";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    defaultDistantLight()->values().setFloat3(
        Names::LightProperties::kDirection, value_ptr(vec3(0, -1, -1)));
    defaultDistantLight()->values().setFloat(
        Names::LightProperties::kIntensity, 3.0f);

    Path geomPath = createTeapotGeometry(*pScene);
    const Path kMaterialPath = "BaseColorOptMaterial";
    pScene->setMaterialType(kMaterialPath);

    Properties instProps;
    instProps[Names::InstanceProperties::kMaterial] = kMaterialPath;
    ASSERT_TRUE(pScene->addInstance("BaseColorOptTeapot", geomPath, instProps));

    // ---- Render target: blue diffuse teapot ----
    {
        Properties p;
        p["base_color"]         = vec3(0.05f, 0.05f, 0.9f);
        p["metalness"]          = 0.0f;
        p["specular_roughness"] = 0.3f;
        p["emission"]           = 0.0f;
        pScene->setMaterialProperties(kMaterialPath, p);
    }

    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    pRenderer->render(0, kSPP);
    pRenderer->waitForTask();

    size_t targetStride = 0;
    const float* pTargetRaw =
        reinterpret_cast<const float*>(pTargetBuf->data(targetStride, true));
    ASSERT_NE(pTargetRaw, nullptr);
    vector<float> targetPixels(pTargetRaw, pTargetRaw + kWidth * kHeight * 4);
    saveFloatRGBAasPNG(kDebugDir + "/target.png", targetPixels.data(), kWidth, kHeight);

    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "BaseColorOptTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    auto computeLoss = [&](const float* pRendered, const float* pTarget,
                            int width, int height) -> double
    {
        double sum   = 0.0;
        int    count = 0;
        for (int i = 0; i < width * height; i++)
        {
            if (pRendered[i * 4 + 3] < 0.5f)
                continue;
            count++;
            for (int c = 0; c < 3; c++)
            {
                float d = pRendered[i * 4 + c] - pTarget[i * 4 + c];
                sum += d * d;
            }
        }
        return count > 0 ? sum / count : 0.0;
    };

    vec3 baseColor = vec3(0.9f, 0.05f, 0.05f);
    float gradScale = -1.0f;

    vector<double> losses;
    losses.reserve(kNumSteps);

    for (int step = 0; step < kNumSteps; step++)
    {
        {
            Properties p;
            p["base_color"]         = baseColor;
            p["metalness"]          = 0.0f;
            p["specular_roughness"] = 0.3f;
            p["emission"]           = 0.0f;
            pScene->setMaterialProperties(kMaterialPath, p);
        }

        IRenderBufferPtr pOptBuf =
            pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
        pRenderer->setTargets({ { AOV::kFinal, pOptBuf } });
        pRenderer->render(0, kSPP);
        pRenderer->waitForTask();

        size_t optStride = 0;
        const float* pOptRaw =
            reinterpret_cast<const float*>(pOptBuf->data(optStride, true));
        ASSERT_NE(pOptRaw, nullptr);
        double loss = computeLoss(pOptRaw, targetPixels.data(), kWidth, kHeight);
        losses.push_back(loss);

        {
            char fname[64];
            snprintf(fname, sizeof(fname), "/step_%03d.png", step);
            saveFloatRGBAasPNG(kDebugDir + fname, pOptRaw, kWidth, kHeight);

            if (step == 0 || step == kNumSteps / 2 || step == kNumSteps - 1)
            {
                snprintf(fname, sizeof(fname), "/grad_%03d.png", step);
                float frameMax = 0.0f;
                saveGradientVisPNG(kDebugDir + fname,
                    pOptRaw, targetPixels.data(), kWidth, kHeight,
                    gradScale, &frameMax);
                if (step == 0)
                    gradScale = frameMax;
                snprintf(fname, sizeof(fname), "/diff_%03d.png", step);
                saveDiffPNG(kDebugDir + fname,
                    targetPixels.data(), pOptRaw, kWidth, kHeight);
            }
        }

        vector<float> grads = pRenderer->getMaterialGradients();
        ASSERT_EQ(grads.size(), 15u) << "No valid surface hits at step " << step;

        AU_INFO("  Step %3d: loss=%.6f  baseColor=(%.3f, %.3f, %.3f)  grad=(%.4f, %.4f, %.4f)",
            step, loss, baseColor.x, baseColor.y, baseColor.z,
            grads[0], grads[1], grads[2]);

        constexpr float kEps = 0.01f;
        baseColor.x -= kLR * grads[0];
        baseColor.y -= kLR * grads[1];
        baseColor.z -= kLR * grads[2];
        baseColor.x = std::max(kEps, std::min(1.0f, baseColor.x));
        baseColor.y = std::max(kEps, std::min(1.0f, baseColor.y));
        baseColor.z = std::max(kEps, std::min(1.0f, baseColor.z));
    }

    int maxMonoRun = 1, currentRun = 1;
    for (int i = 1; i < (int)losses.size(); i++)
    {
        if (losses[i] <= losses[i - 1])
        {
            currentRun++;
            maxMonoRun = std::max(maxMonoRun, currentRun);
        }
        else
        {
            currentRun = 1;
        }
    }
    AU_INFO("BaseColor Optim — longest monotone decrease: %d steps (need %d)",
        maxMonoRun, kMinMonoSteps);
    EXPECT_GE(maxMonoRun, kMinMonoSteps)
        << "Loss did not decrease monotonically for " << kMinMonoSteps
        << " consecutive steps. Longest run: " << maxMonoRun;

    double initialLoss = losses.front();
    double finalLoss   = losses.back();
    AU_INFO("BaseColor Optim — initial loss=%.6f  final loss=%.6f  ratio=%.4f",
        initialLoss, finalLoss, finalLoss / initialLoss);
    EXPECT_LT(finalLoss, initialLoss * kLossReductionFactor)
        << "Final loss (" << finalLoss << ") is not < "
        << kLossReductionFactor << " * initial loss (" << initialLoss << ")";

    AU_INFO("BaseColor Optim — debug images saved to %s", kDebugDir.c_str());
}

// ============================================================
// TestCornellBoxGradientVis — Cornell box with gradient visualization
//
// A classic Cornell box built from 5 planes + ceiling light panel + diffuse
// teapot. The LEFT wall's base_color is the optimized parameter (RED → GREEN).
// Gradient images are saved at every step so the user can see Mitsuba-3-style
// per-pixel loss gradient visualizations that reveal color bleeding paths.
// ============================================================
TEST_P(DiffRenderingTest, TestCornellBoxGradientVis)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int   kSPP                = kTestSppIndirect;
    constexpr int   kNumSteps           = 100;
    // LR retuned from 0.50 after the backward pass gained the estimator scale
    // factors (visibility * radiance / pdf): inside the box the environment is
    // mostly occluded, so the correctly-scaled gradients are ~4-5x smaller than
    // the old unscaled ones and the same trajectory needs a ~5x larger step.
    constexpr float kLR                 = 2.5f;
    constexpr int   kMinMonoSteps       = 8;
    constexpr float kLossReductionFactor = 0.50f;

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/CornellBox";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    pRenderer->options().setInt("traceDepth", 5);
    setDefaultRendererCamera(vec3(0, 1.5f, -5.5f), vec3(0, 1.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // Disable the default distant light — only the emissive ceiling panel lights
    // the box, producing the classic Cornell box indirect-illumination look.
    defaultDistantLight()->values().setFloat(
        Names::LightProperties::kIntensity, 0.0f);

    // createPlaneGeometry() → 2x2 XY quad at Z=0, normal (0,0,-1).

    // ---- Back wall: normal -Z (faces camera), translate to Z=+3 ----
    Path backWallGeom = createPlaneGeometry(*pScene);
    const Path kBackWallMtl = "CornellBackWallMtl";
    pScene->setMaterialType(kBackWallMtl);
    {
        Properties p;
        p["base_color"] = vec3(0.75f, 0.75f, 0.75f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kBackWallMtl, p);
    }
    mat4 backWallXform = glm::translate(vec3(0, 1.5f, 3.0f))
        * glm::scale(vec3(3.0f, 1.5f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("CornellBackWall", backWallGeom,
        { { Names::InstanceProperties::kMaterial, kBackWallMtl },
          { Names::InstanceProperties::kTransform, backWallXform } }));

    // ---- Floor: rotate +90° around X → normal +Y, at Y=0 ----
    Path floorGeom = createPlaneGeometry(*pScene);
    const Path kFloorMtl = "CornellFloorMtl";
    pScene->setMaterialType(kFloorMtl);
    {
        Properties p;
        p["base_color"] = vec3(0.75f, 0.75f, 0.75f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kFloorMtl, p);
    }
    mat4 floorXform = glm::translate(vec3(0, 0, 1.5f))
        * glm::rotate(glm::radians(90.0f), vec3(1, 0, 0))
        * glm::scale(vec3(3.0f, 3.0f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("CornellFloor", floorGeom,
        { { Names::InstanceProperties::kMaterial, kFloorMtl },
          { Names::InstanceProperties::kTransform, floorXform } }));

    // ---- Ceiling: rotate -90° around X → normal -Y, at Y=3 ----
    Path ceilGeom = createPlaneGeometry(*pScene);
    const Path kCeilMtl = "CornellCeilMtl";
    pScene->setMaterialType(kCeilMtl);
    {
        Properties p;
        p["base_color"] = vec3(0.75f, 0.75f, 0.75f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kCeilMtl, p);
    }
    mat4 ceilXform = glm::translate(vec3(0, 3.0f, 1.5f))
        * glm::rotate(glm::radians(-90.0f), vec3(1, 0, 0))
        * glm::scale(vec3(3.0f, 3.0f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("CornellCeiling", ceilGeom,
        { { Names::InstanceProperties::kMaterial, kCeilMtl },
          { Names::InstanceProperties::kTransform, ceilXform } }));

    // ---- Left wall: rotate -90° around Y → normal +X, at X=-3 ----
    Path leftWallGeom = createPlaneGeometry(*pScene);
    const Path kLeftWallMtl = "CornellLeftWallMtl";
    pScene->setMaterialType(kLeftWallMtl);
    mat4 leftWallXform = glm::translate(vec3(-3.0f, 1.5f, 1.5f))
        * glm::rotate(glm::radians(-90.0f), vec3(0, 1, 0))
        * glm::scale(vec3(3.0f, 1.5f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("CornellLeftWall", leftWallGeom,
        { { Names::InstanceProperties::kMaterial, kLeftWallMtl },
          { Names::InstanceProperties::kTransform, leftWallXform } }));

    // ---- Right wall: rotate +90° around Y → normal -X, at X=+3 ----
    Path rightWallGeom = createPlaneGeometry(*pScene);
    const Path kRightWallMtl = "CornellRightWallMtl";
    pScene->setMaterialType(kRightWallMtl);
    {
        Properties p;
        p["base_color"] = vec3(0.75f, 0.75f, 0.75f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kRightWallMtl, p);
    }
    mat4 rightWallXform = glm::translate(vec3(3.0f, 1.5f, 1.5f))
        * glm::rotate(glm::radians(90.0f), vec3(0, 1, 0))
        * glm::scale(vec3(3.0f, 1.5f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("CornellRightWall", rightWallGeom,
        { { Names::InstanceProperties::kMaterial, kRightWallMtl },
          { Names::InstanceProperties::kTransform, rightWallXform } }));

    // ---- Light panel: emissive quad on ceiling, facing down ----
    Path lightGeom = createPlaneGeometry(*pScene);
    const Path kLightMtl = "CornellLightMtl";
    pScene->setMaterialType(kLightMtl);
    {
        Properties p;
        p["emission_color"] = vec3(1.0f, 1.0f, 1.0f);
        p["emission"]       = 10.0f;
        p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kLightMtl, p);
    }
    mat4 lightXform = glm::translate(vec3(0, 2.98f, 1.5f))
        * glm::rotate(glm::radians(-90.0f), vec3(1, 0, 0))
        * glm::scale(vec3(1.0f, 1.0f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("CornellLight", lightGeom,
        { { Names::InstanceProperties::kMaterial, kLightMtl },
          { Names::InstanceProperties::kTransform, lightXform } }));

    // ---- Teapot: small diffuse object in the center of the box ----
    Path teapotGeom = createTeapotGeometry(*pScene);
    const Path kTeapotMtl = "CornellTeapotMtl";
    pScene->setMaterialType(kTeapotMtl);
    {
        Properties p;
        p["base_color"]         = vec3(0.8f, 0.8f, 0.8f);
        p["metalness"]          = 0.0f;
        p["specular_roughness"] = 0.4f;
        p["emission"]           = 0.0f;
        pScene->setMaterialProperties(kTeapotMtl, p);
    }
    ASSERT_TRUE(pScene->addInstance("CornellTeapot", teapotGeom,
        { { Names::InstanceProperties::kMaterial, kTeapotMtl } }));

    // ---- Render TARGET: left wall = GREEN ----
    {
        Properties p;
        p["base_color"] = vec3(0.1f, 0.8f, 0.1f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kLeftWallMtl, p);
    }

    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    pRenderer->render(0, kSPP);
    pRenderer->waitForTask();

    size_t targetStride = 0;
    const float* pTargetRaw =
        reinterpret_cast<const float*>(pTargetBuf->data(targetStride, true));
    ASSERT_NE(pTargetRaw, nullptr);
    vector<float> targetPixels(pTargetRaw, pTargetRaw + kWidth * kHeight * 4);
    saveFloatRGBAasPNG(kDebugDir + "/target.png", targetPixels.data(), kWidth, kHeight);

    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "CornellTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    auto computeLoss = [&](const float* pRendered, const float* pTarget,
                            int width, int height) -> double
    {
        double sum   = 0.0;
        int    count = 0;
        for (int i = 0; i < width * height; i++)
        {
            if (pRendered[i * 4 + 3] < 0.5f)
                continue;
            count++;
            for (int c = 0; c < 3; c++)
            {
                float d = pRendered[i * 4 + c] - pTarget[i * 4 + c];
                sum += d * d;
            }
        }
        return count > 0 ? sum / count : 0.0;
    };

    // ---- Optimization: left wall RED → GREEN ----
    vec3 leftWallColor = vec3(0.8f, 0.1f, 0.1f);
    float gradScale = -1.0f; // captured at step 0, reused for consistent color mapping

    vector<double> losses;
    losses.reserve(kNumSteps);

    for (int step = 0; step < kNumSteps; step++)
    {
        {
            Properties p;
            p["base_color"] = leftWallColor;
            p["emission"]   = 0.0f;
            pScene->setMaterialProperties(kLeftWallMtl, p);
        }

        IRenderBufferPtr pOptBuf =
            pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
        pRenderer->setTargets({ { AOV::kFinal, pOptBuf } });
        pRenderer->render(0, kSPP);
        pRenderer->waitForTask();

        size_t optStride = 0;
        const float* pOptRaw =
            reinterpret_cast<const float*>(pOptBuf->data(optStride, true));
        ASSERT_NE(pOptRaw, nullptr);
        double loss = computeLoss(pOptRaw, targetPixels.data(), kWidth, kHeight);
        losses.push_back(loss);

        {
            char fname[64];
            snprintf(fname, sizeof(fname), "/step_%03d.png", step);
            saveFloatRGBAasPNG(kDebugDir + fname, pOptRaw, kWidth, kHeight);

            snprintf(fname, sizeof(fname), "/grad_%03d.png", step);
            float frameMax = 0.0f;
            saveGradientVisPNG(kDebugDir + fname,
                pOptRaw, targetPixels.data(), kWidth, kHeight,
                gradScale, &frameMax);
            if (step == 0)
                gradScale = frameMax;
            snprintf(fname, sizeof(fname), "/diff_%03d.png", step);
            saveDiffPNG(kDebugDir + fname,
                targetPixels.data(), pOptRaw, kWidth, kHeight);
        }

        vector<float> grads = pRenderer->getMaterialGradients();
        ASSERT_EQ(grads.size(), 15u) << "No valid surface hits at step " << step;

        AU_INFO("  Step %3d: loss=%.6f  leftWall=(%.3f, %.3f, %.3f)  "
            "grad=(%.4f, %.4f, %.4f)",
            step, loss,
            leftWallColor.x, leftWallColor.y, leftWallColor.z,
            grads[0], grads[1], grads[2]);

        constexpr float kEps = 0.01f;
        leftWallColor.x -= kLR * grads[0];
        leftWallColor.y -= kLR * grads[1];
        leftWallColor.z -= kLR * grads[2];
        leftWallColor.x = std::max(kEps, std::min(1.0f, leftWallColor.x));
        leftWallColor.y = std::max(kEps, std::min(1.0f, leftWallColor.y));
        leftWallColor.z = std::max(kEps, std::min(1.0f, leftWallColor.z));
    }

    // ---- Assertions ----
    int maxMonoRun = 1, currentRun = 1;
    for (int i = 1; i < (int)losses.size(); i++)
    {
        if (losses[i] <= losses[i - 1])
        {
            currentRun++;
            maxMonoRun = std::max(maxMonoRun, currentRun);
        }
        else
        {
            currentRun = 1;
        }
    }
    AU_INFO("Cornell Box — longest monotone decrease: %d steps (need %d)",
        maxMonoRun, kMinMonoSteps);
    EXPECT_GE(maxMonoRun, kMinMonoSteps)
        << "Loss did not decrease monotonically for " << kMinMonoSteps
        << " consecutive steps. Longest run: " << maxMonoRun;

    double initialLoss = losses.front();
    double finalLoss   = losses.back();
    AU_INFO("Cornell Box — initial loss=%.6f  final loss=%.6f  ratio=%.4f",
        initialLoss, finalLoss, finalLoss / initialLoss);
    EXPECT_LT(finalLoss, initialLoss * kLossReductionFactor)
        << "Final loss (" << finalLoss << ") is not < "
        << kLossReductionFactor << " * initial loss (" << initialLoss << ")";

    AU_INFO("Cornell Box — gradient images saved to %s", kDebugDir.c_str());
}

// ============================================================
// Test A — Multi-bounce gradient sign (color bleeding)
//
// Scene: Two planes forming an L-shape (floor + wall). A bright emissive light
// panel illuminates the floor from above. The floor is RED diffuse; the wall is
// WHITE diffuse. The camera looks at the wall, which receives indirect red light
// bounced off the floor.
//
// Target: same scene but the floor is GREEN. The wall therefore receives green
// indirect light.
//
// The gradient on the floor's base_color must drive it from red toward green:
//   grad[floor.baseColor.r] > 0  (reduce red contribution on wall)
//   grad[floor.baseColor.g] < 0  (increase green contribution on wall)
//
// This can only succeed if bounce 1 (the floor hit that provides indirect
// illumination to the wall) is recorded and differentiated.
// ============================================================
TEST_P(DiffRenderingTest, TestMultiBounceGradientSign)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }


    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/MultiBounce";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    pRenderer->options().setInt("traceDepth", 5);

    // Camera looks at the wall (along +Z), positioned so the wall fills the frame.
    setDefaultRendererCamera(vec3(0, 0.5f, -3.0f), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // NOTE: createPlaneGeometry() produces a 2x2 quad in the XY plane at Z=0
    // with normal (0,0,-1). Transforms below orient each surface appropriately.

    // Floor plane — rotate +90° around X to get normal +Y, then scale.
    Path floorGeom = createPlaneGeometry(*pScene);
    const Path kFloorMaterial = "FloorMaterial";
    pScene->setMaterialType(kFloorMaterial);

    // Wall plane — default normal is already -Z (faces camera). Just translate & scale.
    Path wallGeom = createPlaneGeometry(*pScene);
    const Path kWallMaterial = "WallMaterial";
    pScene->setMaterialType(kWallMaterial);

    // Light panel — rotate -90° around X to get normal -Y (faces down).
    Path lightGeom = createPlaneGeometry(*pScene);
    const Path kLightMaterial = "LightMaterial";
    pScene->setMaterialType(kLightMaterial);

    // Floor: large horizontal surface at Y=0, normal +Y.
    // Rotate +90° around X swings -Z normal to +Y; scale 4x in XZ after rotation.
    mat4 floorXform = glm::rotate(glm::radians(90.0f), vec3(1, 0, 0))
        * glm::scale(vec3(4.0f, 4.0f, 1.0f));
    Properties floorInstProps;
    floorInstProps[Names::InstanceProperties::kMaterial]  = kFloorMaterial;
    floorInstProps[Names::InstanceProperties::kTransform] = floorXform;
    ASSERT_TRUE(pScene->addInstance("FloorInstance", floorGeom, floorInstProps));

    // Wall: vertical surface at Z=2, normal -Z (faces camera).
    // No rotation — default -Z normal is correct.
    mat4 wallXform = glm::translate(vec3(0, 1.0f, 2.0f))
        * glm::scale(vec3(4.0f, 2.0f, 1.0f));
    Properties wallInstProps;
    wallInstProps[Names::InstanceProperties::kMaterial]  = kWallMaterial;
    wallInstProps[Names::InstanceProperties::kTransform] = wallXform;
    ASSERT_TRUE(pScene->addInstance("WallInstance", wallGeom, wallInstProps));

    // Light panel: emissive surface at Y=3 above the floor, normal -Y (faces down).
    // Rotate -90° around X swings -Z normal to -Y.
    mat4 lightXform = glm::translate(vec3(0, 3.0f, 1.0f))
        * glm::rotate(glm::radians(-90.0f), vec3(1, 0, 0))
        * glm::scale(vec3(3.0f, 3.0f, 1.0f));
    Properties lightInstProps;
    lightInstProps[Names::InstanceProperties::kMaterial]  = kLightMaterial;
    lightInstProps[Names::InstanceProperties::kTransform] = lightXform;
    ASSERT_TRUE(pScene->addInstance("LightInstance", lightGeom, lightInstProps));

    // Light panel material: bright emissive white.
    {
        Properties p;
        p["emission_color"] = vec3(1.0f, 1.0f, 1.0f);
        p["emission"]       = 10.0f;
        p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kLightMaterial, p);
    }

    // Wall material: white diffuse.
    {
        Properties p;
        p["base_color"] = vec3(0.8f, 0.8f, 0.8f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kWallMaterial, p);
    }

    // ---- Render TARGET: floor is GREEN ----
    {
        Properties p;
        p["base_color"] = vec3(0.0f, 0.8f, 0.0f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kFloorMaterial, p);
    }

    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    size_t targetStride = 0;
    const float* pTargetPixels =
        reinterpret_cast<const float*>(pTargetBuf->data(targetStride, true));
    ASSERT_NE(pTargetPixels, nullptr);
    saveFloatRGBAasPNG(kDebugDir + "/target_green_floor.png", pTargetPixels, kWidth, kHeight);

    vector<float> targetPixels(pTargetPixels, pTargetPixels + kWidth * kHeight * 4);

    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "MultiBounceTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    // ---- Render with RED floor (the parameter we want gradients for) ----
    {
        Properties p;
        p["base_color"] = vec3(0.8f, 0.0f, 0.0f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kFloorMaterial, p);
    }

    IRenderBufferPtr pRedBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pRedBuf } });
    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    size_t redStride = 0;
    const float* pRedPixels =
        reinterpret_cast<const float*>(pRedBuf->data(redStride, true));
    ASSERT_NE(pRedPixels, nullptr);
    saveFloatRGBAasPNG(kDebugDir + "/rendered_red_floor.png", pRedPixels, kWidth, kHeight);

    vector<float> grads = pRenderer->getMaterialGradients();
    ASSERT_EQ(grads.size(), 15u) << "No valid surface hits.";

    AU_INFO("MultiBounce Test A — mean material gradients:");
    AU_INFO("  baseColor:      (%.6f, %.6f, %.6f)", grads[0], grads[1], grads[2]);
    AU_INFO("  roughness:      %.6f", grads[3]);
    AU_INFO("  metalness:      %.6f", grads[4]);
    AU_INFO("  emissionColor:  (%.6f, %.6f, %.6f)", grads[5], grads[6], grads[7]);
    AU_INFO("  emission:       %.6f", grads[8]);
    AU_INFO("  specular:       %.6f", grads[9]);
    AU_INFO("  specularColor:  (%.6f, %.6f, %.6f)", grads[10], grads[11], grads[12]);
    AU_INFO("  specularIOR:    %.6f", grads[13]);
    AU_INFO("  specularAniso:  %.6f", grads[14]);

    // With multi-bounce, the gradients from the floor's indirect contribution
    // to the wall should produce non-zero BRDF-related gradients (baseColor,
    // roughness, metalness, specular, etc.) via bwd_diff(evaluateMaterial).
    float brdfGradMag = 0.0f;
    for (int i = 0; i < 15; i++)
        brdfGradMag += std::abs(grads[i]);
    EXPECT_GT(brdfGradMag, 0.0f)
        << "Expected non-zero material gradients from multi-bounce indirect illumination.";

    float baseColorMag = std::abs(grads[0]) + std::abs(grads[1]) + std::abs(grads[2]);
    AU_INFO("  baseColor magnitude: %.8f, total gradient magnitude: %.8f",
            baseColorMag, brdfGradMag);
}

// ============================================================
// Test B — Mirror reflection gradient
//
// Scene: A teapot (glossy/metallic) in front of an emissive backdrop plane.
// Camera sees the teapot, which reflects the backdrop color.
// Target: same scene but backdrop is BLUE emissive.
// Rendered: backdrop is RED emissive.
//
// Gradients on the backdrop's emissionColor should be non-zero, because the
// reflected color on the teapot comes from bounce 1 (backdrop surface hit).
// ============================================================
TEST_P(DiffRenderingTest, TestMirrorReflectionGradient)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }


    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/Mirror";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    pRenderer->options().setInt("traceDepth", 5);

    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // Metallic teapot (mirror-like).
    Path teapotGeom = createTeapotGeometry(*pScene);
    const Path kTeapotMaterial = "MirrorTeapotMaterial";
    pScene->setMaterialType(kTeapotMaterial);
    {
        Properties p;
        p["base_color"] = vec3(0.9f, 0.9f, 0.9f);
        p["metalness"]  = 1.0f;
        p["specular_roughness"] = 0.01f;
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kTeapotMaterial, p);
    }
    ASSERT_TRUE(pScene->addInstance("TeapotInstance", teapotGeom,
        { { Names::InstanceProperties::kMaterial, kTeapotMaterial } }));

    // Backdrop plane behind the teapot. Default plane normal is -Z (faces camera),
    // so no rotation needed — just translate behind the teapot and scale.
    Path backdropGeom = createPlaneGeometry(*pScene);
    const Path kBackdropMaterial = "BackdropMaterial";
    pScene->setMaterialType(kBackdropMaterial);

    mat4 backdropXform = glm::translate(vec3(0, 1.0f, 3.0f))
        * glm::scale(vec3(5.0f, 5.0f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("BackdropInstance", backdropGeom,
        { { Names::InstanceProperties::kMaterial, kBackdropMaterial },
          { Names::InstanceProperties::kTransform, backdropXform } }));

    // ---- Render TARGET: blue backdrop ----
    {
        Properties p;
        p["emission_color"] = vec3(0.0f, 0.0f, 1.0f);
        p["emission"]       = 3.0f;
        p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kBackdropMaterial, p);
    }

    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    size_t targetStride = 0;
    const float* pTargetPixels =
        reinterpret_cast<const float*>(pTargetBuf->data(targetStride, true));
    ASSERT_NE(pTargetPixels, nullptr);
    saveFloatRGBAasPNG(kDebugDir + "/target_blue_backdrop.png", pTargetPixels, kWidth, kHeight);

    vector<float> targetPixels(pTargetPixels, pTargetPixels + kWidth * kHeight * 4);

    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "MirrorTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    // ---- Render with RED backdrop ----
    {
        Properties p;
        p["emission_color"] = vec3(1.0f, 0.0f, 0.0f);
        p["emission"]       = 3.0f;
        p["base_color"]     = vec3(0.0f, 0.0f, 0.0f);
        pScene->setMaterialProperties(kBackdropMaterial, p);
    }

    IRenderBufferPtr pRedBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pRedBuf } });
    pRenderer->render(0, 1);
    pRenderer->waitForTask();

    size_t redStride = 0;
    const float* pRedPixels =
        reinterpret_cast<const float*>(pRedBuf->data(redStride, true));
    ASSERT_NE(pRedPixels, nullptr);
    saveFloatRGBAasPNG(kDebugDir + "/rendered_red_backdrop.png", pRedPixels, kWidth, kHeight);

    vector<float> grads = pRenderer->getMaterialGradients();
    ASSERT_EQ(grads.size(), 15u) << "No valid surface hits.";

    AU_INFO("Mirror Test B — mean material gradients:");
    AU_INFO("  emissionColor: (%.6f, %.6f, %.6f)", grads[5], grads[6], grads[7]);
    AU_INFO("  emission: %.6f", grads[8]);

    // The backdrop emissionColor gradients should be non-zero: the reflected color
    // on the teapot comes from the backdrop (bounce 1). This only works with
    // multi-bounce path recording.
    float emitGradMag = std::abs(grads[5]) + std::abs(grads[6]) + std::abs(grads[7]);
    EXPECT_GT(emitGradMag, 0.0f)
        << "Expected non-zero emissionColor gradients from reflected backdrop (bounce 1).";

    // Stronger directional check: red is over-represented, so grad[emissionColor.r] > 0.
    EXPECT_GT(grads[5], 0.0f)
        << "Expected grad[emissionColor.r] > 0 (rendered red > target red).";
    // Blue is under-represented, so grad[emissionColor.b] < 0.
    EXPECT_LT(grads[7], 0.0f)
        << "Expected grad[emissionColor.b] < 0 (rendered blue < target blue).";
}

// ============================================================
// Test C — Numerical gradient check with multi-bounce
//
// Uses the mirror reflection scene (Test B) to verify that the AD gradient
// from the backdrop's emissionColor (which is only visible via reflection
// on the teapot at bounce 1) matches finite differences.
// ============================================================
TEST_P(DiffRenderingTest, TestMultiBounceNumericalGradCheck)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int   kSPP    = kTestSpp;
    constexpr float kEps    = 0.01f;
    constexpr float kRelTol = 0.20f; // 20% tolerance: detached sampling (MIS weights not
                                     // differentiated) introduces ~15% systematic AD bias

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/MultiBounceNumerical";
    std::filesystem::create_directories(kDebugDir);

    // Surface-hit pixel mask for consistent FD/AD comparison.
    vector<float> basePixelMask;
    auto computeMeanLoss = [&](const float* pRendered, const float* pTarget,
                                int width, int height) -> double
    {
        double sum   = 0.0;
        int    count = 0;
        for (int i = 0; i < width * height; i++)
        {
            float mask = basePixelMask.empty() ? pRendered[i * 4 + 3] : basePixelMask[i * 4 + 3];
            if (mask < 0.5f)
                continue;
            count++;
            for (int c = 0; c < 3; c++)
            {
                float diff = pRendered[i * 4 + c] - pTarget[i * 4 + c];
                sum += diff * diff;
            }
        }
        return count > 0 ? sum / count : 0.0;
    };

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    pRenderer->options().setInt("traceDepth", 5);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // Teapot: metallic mirror.
    Path teapotGeom = createTeapotGeometry(*pScene);
    const Path kTeapotMaterial = "NumCheckTeapotMtl";
    pScene->setMaterialType(kTeapotMaterial);
    {
        Properties p;
        p["base_color"] = vec3(0.9f, 0.9f, 0.9f);
        p["metalness"]  = 1.0f;
        p["specular_roughness"] = 0.01f;
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kTeapotMaterial, p);
    }
    ASSERT_TRUE(pScene->addInstance("NumCheckTeapot", teapotGeom,
        { { Names::InstanceProperties::kMaterial, kTeapotMaterial } }));

    // Backdrop: emissive plane behind teapot. Default -Z normal faces camera.
    Path backdropGeom = createPlaneGeometry(*pScene);
    const Path kBackdropMaterial = "NumCheckBackdropMtl";
    pScene->setMaterialType(kBackdropMaterial);
    mat4 backdropXform = glm::translate(vec3(0, 1.0f, 3.0f))
        * glm::scale(vec3(5.0f, 5.0f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("NumCheckBackdrop", backdropGeom,
        { { Names::InstanceProperties::kMaterial, kBackdropMaterial },
          { Names::InstanceProperties::kTransform, backdropXform } }));

    const vec3  kBaseEmitColor  = vec3(1.0f, 0.5f, 0.2f);
    const float kBaseEmission   = 3.0f;
    const vec3  kTargetEmitColor = vec3(0.0f, 0.0f, 1.0f);

    // ---- Render target ----
    {
        Properties p;
        p["emission_color"] = kTargetEmitColor;
        p["emission"]       = kBaseEmission;
        p["base_color"]     = vec3(0.0f);
        pScene->setMaterialProperties(kBackdropMaterial, p);
    }
    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    pRenderer->render(0, kSPP);
    pRenderer->waitForTask();
    size_t ts = 0;
    const float* pTP = reinterpret_cast<const float*>(pTargetBuf->data(ts, true));
    ASSERT_NE(pTP, nullptr);
    saveFloatRGBAasPNG(kDebugDir + "/target.png", pTP, kWidth, kHeight);
    vector<float> targetPixels(pTP, pTP + kWidth * kHeight * 4);

    IImage::InitData tgt;
    tgt.pImageData = targetPixels.data();
    tgt.format     = ImageFormat::Float_RGBA;
    tgt.linearize  = false;
    tgt.width      = kWidth;
    tgt.height     = kHeight;
    tgt.name       = "NumCheckTarget";
    pRenderer->setDiffTargetImage(tgt);

    // Parameters to test (backdrop emissionColor channels).
    struct ParamTest { const char* name; int gradIdx; vec3 emitColor; float emission; };
    vector<ParamTest> params = {
        { "emissionColor.r", 5, vec3(kBaseEmitColor.x + kEps, kBaseEmitColor.y, kBaseEmitColor.z), kBaseEmission },
        { "emissionColor.g", 6, vec3(kBaseEmitColor.x, kBaseEmitColor.y + kEps, kBaseEmitColor.z), kBaseEmission },
        { "emissionColor.b", 7, vec3(kBaseEmitColor.x, kBaseEmitColor.y, kBaseEmitColor.z + kEps), kBaseEmission },
    };

    // ---- Base render + AD gradients ----
    {
        Properties p;
        p["emission_color"] = kBaseEmitColor;
        p["emission"]       = kBaseEmission;
        p["base_color"]     = vec3(0.0f);
        pScene->setMaterialProperties(kBackdropMaterial, p);
    }
    IRenderBufferPtr pBaseBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pBaseBuf } });
    pRenderer->render(0, kSPP);
    pRenderer->waitForTask();
    size_t bs = 0;
    const float* pBP = reinterpret_cast<const float*>(pBaseBuf->data(bs, true));
    ASSERT_NE(pBP, nullptr);
    saveFloatRGBAasPNG(kDebugDir + "/base.png", pBP, kWidth, kHeight);
    vector<float> basePixels(pBP, pBP + kWidth * kHeight * 4);
    basePixelMask = basePixels;
    double lossBase = computeMeanLoss(basePixels.data(), targetPixels.data(), kWidth, kHeight);

    vector<float> adGrads = pRenderer->getMaterialGradients();
    ASSERT_EQ(adGrads.size(), 15u);

    AU_INFO("MultiBounce Numerical — base loss = %.6f (SPP=%d)", lossBase, kSPP);

    bool allPassed = true;
    for (const auto& pt : params)
    {
        {
            Properties p;
            p["emission_color"] = pt.emitColor;
            p["emission"]       = pt.emission;
            p["base_color"]     = vec3(0.0f);
            pScene->setMaterialProperties(kBackdropMaterial, p);
        }
        IRenderBufferPtr pPertBuf =
            pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
        pRenderer->setTargets({ { AOV::kFinal, pPertBuf } });
        pRenderer->render(0, kSPP);
        pRenderer->waitForTask();
        size_t ps = 0;
        const float* pPP = reinterpret_cast<const float*>(pPertBuf->data(ps, true));
        ASSERT_NE(pPP, nullptr);
        saveFloatRGBAasPNG(kDebugDir + "/perturbed_" + pt.name + ".png", pPP, kWidth, kHeight);
        vector<float> pertPixels(pPP, pPP + kWidth * kHeight * 4);

        double lossPert = computeMeanLoss(pertPixels.data(), targetPixels.data(), kWidth, kHeight);
        double fd       = (lossPert - lossBase) / kEps;
        double ad       = adGrads[pt.gradIdx];
        double relErr   = (std::abs(fd) > 1e-8) ? std::abs(ad - fd) / std::abs(fd) : std::abs(ad - fd);

        AU_INFO("  %-20s  AD=%+.6f  FD=%+.6f  relErr=%.4f  %s",
            pt.name, ad, fd, relErr, relErr < kRelTol ? "PASS" : "FAIL");

        EXPECT_LT(relErr, kRelTol)
            << "Numerical gradient check FAILED for " << pt.name
            << ": AD=" << ad << " FD=" << fd << " relErr=" << relErr;
        if (relErr >= kRelTol)
            allPassed = false;
    }

    AU_INFO("MultiBounce Numerical — %s", allPassed ? "ALL PASSED" : "SOME FAILED");
}

// ============================================================
// Test D — Optimization loop with indirect illumination (baseColor)
//
// Uses the mirror reflection scene: optimizes the backdrop base_color
// (visible only via reflection on the teapot at bounce 1) to match a target.
// A front-facing distant light illuminates the backdrop so its BRDF
// (which depends on base_color) modulates the reflected radiance.
// Starting from RED backdrop, target is BLUE backdrop.
//
// Pass criteria:
//   - Loss decreases over iterations.
//   - Final loss < 20% of initial (looser tolerance for indirect paths).
// ============================================================
TEST_P(DiffRenderingTest, TestMultiBounceOptimizationLoop)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int   kSPP                = kTestSpp;
    constexpr int   kNumSteps           = 50;
    constexpr float kLR                 = 0.50f;
    constexpr int   kMinMonoSteps       = 10;
    constexpr float kLossReductionFactor = 0.30f;

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/MultiBounceOptim";
    std::filesystem::create_directories(kDebugDir);

    IRendererPtr pRenderer = createDefaultRenderer(kWidth, kHeight);
    ASSERT_NE(pRenderer, nullptr);
    pRenderer->options().setBoolean("isGammaCorrectionEnabled", false);
    pRenderer->options().setBoolean("alphaEnabled", true);
    pRenderer->options().setInt("traceDepth", 5);
    setDefaultRendererCamera(vec3(0, 1, -5), vec3(0, 0.5f, 0));

    IScenePtr pScene = createDefaultScene();
    ASSERT_NE(pScene, nullptr);

    // Point the default distant light at the backdrop (toward +Z) so the
    // backdrop's BRDF / base_color actually modulates visible radiance.
    defaultDistantLight()->values().setFloat3(
        Names::LightProperties::kDirection, value_ptr(vec3(0, 0, -1)));
    defaultDistantLight()->values().setFloat(
        Names::LightProperties::kIntensity, 3.0f);

    Path teapotGeom = createTeapotGeometry(*pScene);
    const Path kTeapotMaterial = "OptTeapotMtl";
    pScene->setMaterialType(kTeapotMaterial);
    {
        Properties p;
        p["base_color"]  = vec3(0.9f, 0.9f, 0.9f);
        p["metalness"]   = 1.0f;
        p["specular_roughness"] = 0.01f;
        p["emission"]    = 0.0f;
        pScene->setMaterialProperties(kTeapotMaterial, p);
    }
    ASSERT_TRUE(pScene->addInstance("OptTeapot", teapotGeom,
        { { Names::InstanceProperties::kMaterial, kTeapotMaterial } }));

    Path backdropGeom = createPlaneGeometry(*pScene);
    const Path kBackdropMaterial = "OptBackdropMtl";
    pScene->setMaterialType(kBackdropMaterial);
    mat4 backdropXform = glm::translate(vec3(0, 1.0f, 3.0f))
        * glm::scale(vec3(5.0f, 5.0f, 1.0f));
    ASSERT_TRUE(pScene->addInstance("OptBackdrop", backdropGeom,
        { { Names::InstanceProperties::kMaterial, kBackdropMaterial },
          { Names::InstanceProperties::kTransform, backdropXform } }));

    // ---- Target: cool blue diffuse backdrop ----
    // Avoid exact zeros in base_color components: pow(0, n) has an AD singularity
    // in Slang where the derivative is computed as pow(x,n)/x*n → 0/0 = NaN.
    {
        Properties p;
        p["base_color"] = vec3(0.05f, 0.05f, 0.9f);
        p["emission"]   = 0.0f;
        pScene->setMaterialProperties(kBackdropMaterial, p);
    }
    IRenderBufferPtr pTargetBuf =
        pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
    pRenderer->setTargets({ { AOV::kFinal, pTargetBuf } });
    pRenderer->render(0, kSPP);
    pRenderer->waitForTask();
    size_t ts = 0;
    const float* pTP = reinterpret_cast<const float*>(pTargetBuf->data(ts, true));
    ASSERT_NE(pTP, nullptr);
    vector<float> targetPixels(pTP, pTP + kWidth * kHeight * 4);
    saveFloatRGBAasPNG(kDebugDir + "/target.png", targetPixels.data(), kWidth, kHeight);

    IImage::InitData targetInitData;
    targetInitData.pImageData = targetPixels.data();
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "OptTarget";
    pRenderer->setDiffTargetImage(targetInitData);

    auto computeLoss = [&](const float* pRendered, const float* pTarget,
                            int width, int height) -> double
    {
        double sum   = 0.0;
        int    count = 0;
        for (int i = 0; i < width * height; i++)
        {
            if (pRendered[i * 4 + 3] < 0.5f)
                continue;
            count++;
            for (int c = 0; c < 3; c++)
            {
                float d = pRendered[i * 4 + c] - pTarget[i * 4 + c];
                sum += d * d;
            }
        }
        return count > 0 ? sum / count : 0.0;
    };

    // Start from warm red diffuse backdrop.
    // Avoid exact zeros: pow(0, n) has an AD singularity in Slang.
    vec3 baseColor = vec3(0.9f, 0.05f, 0.05f);
    float gradScale = -1.0f;

    vector<double> losses;
    losses.reserve(kNumSteps);

    for (int step = 0; step < kNumSteps; step++)
    {
        {
            Properties p;
            p["base_color"] = baseColor;
            p["emission"]   = 0.0f;
            pScene->setMaterialProperties(kBackdropMaterial, p);
        }

        IRenderBufferPtr pOptBuf =
            pRenderer->createRenderBuffer(kWidth, kHeight, ImageFormat::Float_RGBA);
        pRenderer->setTargets({ { AOV::kFinal, pOptBuf } });
        pRenderer->render(0, kSPP);
        pRenderer->waitForTask();

        size_t os = 0;
        const float* pOP = reinterpret_cast<const float*>(pOptBuf->data(os, true));
        ASSERT_NE(pOP, nullptr);
        double loss = computeLoss(pOP, targetPixels.data(), kWidth, kHeight);
        losses.push_back(loss);

        {
            char fname[64];
            snprintf(fname, sizeof(fname), "/step_%03d.png", step);
            saveFloatRGBAasPNG(kDebugDir + fname, pOP, kWidth, kHeight);

            if (step == 0 || step == kNumSteps / 2 || step == kNumSteps - 1)
            {
                snprintf(fname, sizeof(fname), "/grad_%03d.png", step);
                float frameMax = 0.0f;
                saveGradientVisPNG(kDebugDir + fname,
                    pOP, targetPixels.data(), kWidth, kHeight,
                    gradScale, &frameMax);
                if (step == 0)
                    gradScale = frameMax;
                snprintf(fname, sizeof(fname), "/diff_%03d.png", step);
                saveDiffPNG(kDebugDir + fname,
                    targetPixels.data(), pOP, kWidth, kHeight);
            }
        }

        vector<float> grads = pRenderer->getMaterialGradients();
        ASSERT_EQ(grads.size(), 15u) << "No valid surface hits at step " << step;

        AU_INFO("  Step %3d: loss=%.6f  baseColor=(%.3f, %.3f, %.3f)  grad=(%.4f, %.4f, %.4f)",
            step, loss, baseColor.x, baseColor.y, baseColor.z,
            grads[0], grads[1], grads[2]);

        // SGD on baseColor (indices 0, 1, 2).
        // Clamp to [eps, 1] to avoid pow(0, n) AD singularity in Slang.
        constexpr float kEps = 0.01f;
        baseColor.x -= kLR * grads[0];
        baseColor.y -= kLR * grads[1];
        baseColor.z -= kLR * grads[2];
        baseColor.x = std::max(kEps, std::min(1.0f, baseColor.x));
        baseColor.y = std::max(kEps, std::min(1.0f, baseColor.y));
        baseColor.z = std::max(kEps, std::min(1.0f, baseColor.z));
    }

    // Check 1: monotone decrease for at least kMinMonoSteps consecutive steps.
    int maxMonoRun = 1, currentRun = 1;
    for (int i = 1; i < (int)losses.size(); i++)
    {
        if (losses[i] <= losses[i - 1])
        {
            currentRun++;
            maxMonoRun = std::max(maxMonoRun, currentRun);
        }
        else
        {
            currentRun = 1;
        }
    }
    AU_INFO("MultiBounce Optim — longest monotone decrease: %d steps (need %d)",
        maxMonoRun, kMinMonoSteps);
    EXPECT_GE(maxMonoRun, kMinMonoSteps)
        << "Loss did not decrease monotonically for " << kMinMonoSteps
        << " consecutive steps. Longest run: " << maxMonoRun;

    // Check 2: final loss < kLossReductionFactor * initial loss.
    double initialLoss = losses.front();
    double finalLoss   = losses.back();
    AU_INFO("MultiBounce Optim — initial loss=%.6f  final loss=%.6f  ratio=%.4f",
        initialLoss, finalLoss, finalLoss / initialLoss);
    EXPECT_LT(finalLoss, initialLoss * kLossReductionFactor)
        << "Final loss (" << finalLoss << ") is not < "
        << kLossReductionFactor << " * initial loss (" << initialLoss << ")";
}

INSTANTIATE_TEST_SUITE_P(DiffRendering, DiffRenderingTest, TEST_SUITE_RENDERER_TYPES());

} // namespace

#endif // ENABLE_DIFFERENTIABLE_RENDERING
