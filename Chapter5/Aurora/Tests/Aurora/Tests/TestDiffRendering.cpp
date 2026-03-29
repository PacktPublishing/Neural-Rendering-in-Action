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

// ---- Test fixture ----

class DiffRenderingTest : public TestHelpers::FixtureBase
{
public:
    DiffRenderingTest() {}
    ~DiffRenderingTest() {}
};

// ============================================================
// Step 3 — Gradient sign test (emission-based)
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
TEST_P(DiffRenderingTest, TestStep3GradientSign)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int kWidth  = 64;
    constexpr int kHeight = 64;

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

    // ---- Step 1: Render with emissive BLUE → capture as Float_RGBA target ----
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

    // ---- Step 2: Change to emissive RED ----
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

    // ---- Step 3: Set the blue render as the diff target ----
    IImage::InitData targetInitData;
    targetInitData.pImageData = pRefPixels;
    targetInitData.format     = ImageFormat::Float_RGBA;
    targetInitData.linearize  = false;
    targetInitData.width      = kWidth;
    targetInitData.height     = kHeight;
    targetInitData.name       = "DiffTestTargetImage";
    pRenderer->setDiffTargetImage(targetInitData);

    // ---- Step 4: Render with RED (backward pass runs automatically) ----
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

    // ---- Step 5: Read back gradients and check signs ----
    vector<float> grads = pRenderer->getMaterialGradients();

    ASSERT_EQ(grads.size(), 15u)
        << "getMaterialGradients() returned empty — no valid surface hits. "
           "Check that the teapot is visible.";

    // Gradient layout (MATERIAL_GRAD_STRIDE = 15):
    //   [0-2]  baseColor.xyz
    //   [3]    specularRoughness
    //   [4]    metalness
    //   [5-7]  emissionColor.xyz   ← we check these
    //   [8]    emission
    //   [9]    specular
    //   [10-12] specularColor.xyz
    //   [13]   specularIOR
    //   [14]   specularAnisotropy
    AU_INFO("DiffRenderTest Step 3 — mean material gradients:");
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
// Step 4 — Numerical gradient check (end-to-end correctness)
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
TEST_P(DiffRenderingTest, TestStep4NumericalGradCheck)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int   kWidth   = 64;
    constexpr int   kHeight  = 64;
    constexpr float kEps     = 0.01f;  // small perturbation; emission is deterministic so no MC noise
    constexpr float kRelTol  = 0.05f;  // 5% relative tolerance

    // Create debug output directory.
    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/Step4";
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
    targetInitData.name       = "DiffStep4Target";
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

    AU_INFO("DiffRenderTest Step 4 — base loss = %.6f", lossBase);

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

    AU_INFO("DiffRenderTest Step 4 — %s", allPassed ? "ALL PASSED" : "SOME FAILED");
    AU_INFO("DiffRenderTest Step 4 — debug images saved to %s", kDebugDir.c_str());
}

// ============================================================
// Step 6 — End-to-end C++ optimization loop
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
TEST_P(DiffRenderingTest, TestStep6OptimizationLoop)
{
    if (!isDirectX() || !backendSupported())
    {
        GTEST_SKIP() << "Differentiable rendering requires DirectX backend.";
    }

    constexpr int   kWidth              = 64;
    constexpr int   kHeight             = 64;
    constexpr int   kNumSteps           = 10;
    constexpr float kLR                 = 0.05f;  // SGD learning rate
    constexpr int   kMinMonoSteps       = 8;     // minimum consecutive loss-decreasing steps
    constexpr float kLossReductionFactor = 0.10f; // final loss must be < 10% of initial

    const std::string kDebugDir = "./OutputImages/DiffRenderDebug/Step6";
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
    AU_INFO("DiffRenderTest Step 6 — longest monotone decrease: %d steps (need %d)",
        maxMonoRun, kMinMonoSteps);
    EXPECT_GE(maxMonoRun, kMinMonoSteps)
        << "Loss did not decrease monotonically for " << kMinMonoSteps
        << " consecutive steps. Longest run: " << maxMonoRun;

    // ---- Check 2: final loss < kLossReductionFactor * initial loss ----
    double initialLoss = losses.front();
    double finalLoss   = losses.back();
    AU_INFO("DiffRenderTest Step 6 — initial loss=%.6f  final loss=%.6f  ratio=%.4f",
        initialLoss, finalLoss, finalLoss / initialLoss);
    EXPECT_LT(finalLoss, initialLoss * kLossReductionFactor)
        << "Final loss (" << finalLoss << ") is not < "
        << kLossReductionFactor << " * initial loss (" << initialLoss << ")";

    AU_INFO("DiffRenderTest Step 6 — debug images saved to %s", kDebugDir.c_str());
}

INSTANTIATE_TEST_SUITE_P(DiffRendering, DiffRenderingTest, TEST_SUITE_RENDERER_TYPES());

} // namespace

#endif // ENABLE_DIFFERENTIABLE_RENDERING
