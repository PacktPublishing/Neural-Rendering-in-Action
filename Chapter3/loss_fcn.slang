// Image-Based Loss for Circle Learning
// Step 3B: Building toward full image-based autodiff
// Matching circle_focused_learning.py exactly

// ============================================================================
// Helper Functions
// ============================================================================

// Sigmoid activation function (differentiable)
// MUST be [Differentiable] because it's called inside renderCircleAtPixel,
// which is called inside computeCircleLossDifferentiable.
// Without this, gradient chain breaks: ∂loss/∂circle_x cannot be computed!
[Differentiable]
float sigmoid(float x) {
    float clamped_x = clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + exp(-clamped_x));
}

// ============================================================================
// Step 3B.1: Extract Circle Mask from Image
// ============================================================================

// Extract circle mask at a single pixel
// Detects red circle pixels (high red, low green/blue)
// Returns value between 0 (background) and 1 (circle)
float extractCircleMaskAtPixel(float red, float green, float blue) {
    float red_strength = sigmoid((red - 0.5) * 20.0);
    float green_penalty = sigmoid((0.5 - green) * 20.0);
    float blue_penalty = sigmoid((0.5 - blue) * 20.0);
    
    return red_strength * green_penalty * blue_penalty;
}

// Extract mask for entire image
// Input: RGBA image buffer (flattened)
// Output: Mask buffer (grayscale, 0-1)
void extractCircleMask(
    RWStructuredBuffer<float> image,
    uint2 image_size,
    RWStructuredBuffer<float> mask
) {
    for (uint y = 0; y < image_size.y; y++) {
        for (uint x = 0; x < image_size.x; x++) {
            uint idx = y * image_size.x + x;
            uint img_idx = idx * 4;  // RGBA
            
            float red = image[img_idx + 0];
            float green = image[img_idx + 1];
            float blue = image[img_idx + 2];
            
            mask[idx] = extractCircleMaskAtPixel(red, green, blue);
        }
    }
}

// ============================================================================
// Step 3B.2: Compute Image Centroid
// ============================================================================

// Compute centroid (center of mass) of a mask
// Returns (centroid_x, centroid_y)
void computeImageCentroid(
    RWStructuredBuffer<float> mask,
    uint2 image_size,
    out float centroid_x,
    out float centroid_y
) {
    float total_weight = 0;
    float weighted_x = 0;
    float weighted_y = 0;
    
    for (uint y = 0; y < image_size.y; y++) {
        for (uint x = 0; x < image_size.x; x++) {
            uint idx = y * image_size.x + x;
            float m = mask[idx];
            
            total_weight += m;
            weighted_x += x * m;
            weighted_y += y * m;
        }
    }
    
    // Avoid division by zero
    if (total_weight > 0) {
        centroid_x = weighted_x / total_weight;
        centroid_y = weighted_y / total_weight;
    } else {
        // Default to center if no mask
        centroid_x = image_size.x / 2.0;
        centroid_y = image_size.y / 2.0;
    }
}

// ============================================================================
// Step 3B.3: Compute IoU (Intersection over Union)
// ============================================================================

// Result structure for IoU computation
struct IoUResult {
    float intersection_pixels;
    float union_pixels;
    float target_pixels;
    float rendered_pixels;
    float iou;
    float iou_loss;
};

// Compute IoU between target and rendered masks
// Matches PyTorch: intersection = target * rendered, union = target + rendered - intersection
IoUResult computeIoU(
    RWStructuredBuffer<float> target_mask,
    RWStructuredBuffer<float> rendered_mask,
    uint2 image_size
) {
    IoUResult result;
    
    float intersection_sum = 0;
    float target_sum = 0;
    float rendered_sum = 0;
    
    // Sum over all pixels
    for (uint y = 0; y < image_size.y; y++) {
        for (uint x = 0; x < image_size.x; x++) {
            uint idx = y * image_size.x + x;
            float t = target_mask[idx];
            float r = rendered_mask[idx];
            
            float intersect = t * r;
            
            intersection_sum += intersect;
            target_sum += t;
            rendered_sum += r;
        }
    }
    
    // Union = target + rendered - intersection
    float union_sum = target_sum + rendered_sum - intersection_sum;
    
    // Store pixel counts
    result.intersection_pixels = intersection_sum;
    result.union_pixels = union_sum;
    result.target_pixels = target_sum;
    result.rendered_pixels = rendered_sum;
    
    // Compute IoU and loss
    if (union_sum > 0) {
        result.iou = intersection_sum / union_sum;
        result.iou_loss = 1.0 - result.iou;
    } else {
        result.iou = 0;
        result.iou_loss = 1.0;
    }
    
    return result;
}

// ============================================================================
// Step 3B.4: Compute Distance Loss
// ============================================================================

// Compute distance between two centroids, normalized by image size (differentiable)
// Matches PyTorch: centroid_distance = sqrt(sum((target - rendered)^2))
//                  distance_loss = centroid_distance / image_size
[Differentiable]
float computeDistanceLoss(
    no_diff float target_cx, no_diff float target_cy,  // ← Target centroid is constant
    float rendered_cx, float rendered_cy,               // ← Rendered centroid depends on circle params
    no_diff uint2 image_size                            // ← Image size is constant
) {
    float dx = target_cx - rendered_cx;
    float dy = target_cy - rendered_cy;
    float centroid_distance = sqrt(dx * dx + dy * dy);
    
    // Normalize by image size
    float distance_loss = centroid_distance / image_size.x;
    
    return distance_loss;
}

// ============================================================================
// Step 3B.5: Compute Size Loss and Radius Growth Incentive
// ============================================================================

// Result structure for size loss computation
struct SizeLossResult {
    float target_radius_estimate;
    float rendered_radius_estimate;
    float size_ratio;
    float size_penalty;
    float radius_growth_incentive;
};

// Compute size-based losses from mask pixel counts (differentiable)
// Matches PyTorch:
//   size_ratio = min(target_pixels, rendered_pixels) / max(target_pixels, rendered_pixels)
//   size_penalty = 1.0 - size_ratio
//   target_radius_estimate = sqrt(target_pixels / pi)
//   rendered_radius_estimate = sqrt(rendered_pixels / pi)
//   radius_growth_incentive = relu(target_radius - rendered_radius) / image_size
[Differentiable]
SizeLossResult computeSizeLoss(
    no_diff float target_pixels,     // ← Target pixel count is constant
    float rendered_pixels,            // ← Rendered pixel count depends on circle params
    no_diff uint2 image_size          // ← Image size is constant
) {
    SizeLossResult result;
    
    // Estimate radii from pixel counts (Area = pi * r^2, so r = sqrt(Area / pi))
    const float PI = 3.14159;
    result.target_radius_estimate = sqrt(target_pixels / PI);
    result.rendered_radius_estimate = sqrt(rendered_pixels / PI);
    
    // Size ratio (smaller / larger)
    float min_pixels = min(target_pixels, rendered_pixels);
    float max_pixels = max(target_pixels, rendered_pixels);
    
    result.size_ratio = (max_pixels > 0) ? (min_pixels / max_pixels) : 1.0;
    result.size_penalty = 1.0 - result.size_ratio;
    
    // Radius growth incentive (only if rendered is smaller than target)
    // Use max(0, x) instead of if/else for better autodiff (equivalent to ReLU)
    float radius_diff = result.target_radius_estimate - result.rendered_radius_estimate;
    result.radius_growth_incentive = max(0.0, radius_diff) / image_size.x;
    
    return result;
}

// ============================================================================
// Step 3B.6: Unified Loss Computation (All Components Combined)
// ============================================================================

// Render circle at a single pixel (differentiable)
// Returns alpha value [0, 1] for smooth circle rendering
// MUST be [Differentiable] because:
// 1. Called inside computeCircleLossDifferentiable (line 286)
// 2. Takes circle_x, circle_y, circle_radius as parameters (the vars we optimize!)
// 3. Without this, ∂loss/∂circle_x, ∂loss/∂circle_y, ∂loss/∂circle_radius cannot be computed
[Differentiable]
float renderCircleAtPixel(no_diff float px, no_diff float py, float circle_x, float circle_y, float circle_radius) {
    float dx = px - circle_x;
    float dy = py - circle_y;
    float dist_sq = dx * dx + dy * dy;
    float radius_sq = circle_radius * circle_radius + 1e-6;
    
    // Smooth circle edge using sigmoid
    float alpha = sigmoid((radius_sq - dist_sq) * 10.0);
    return alpha;
}

// Extract circle mask from rendered pixel color (differentiable)
[Differentiable]
float extractCircleMaskValue(float red, float green, float blue) {
    float red_strength = sigmoid((red - 0.5) * 20.0);
    float green_penalty = sigmoid((0.5 - green) * 20.0);
    float blue_penalty = sigmoid((0.5 - blue) * 20.0);
    return red_strength * green_penalty * blue_penalty;
}

// MAIN DIFFERENTIABLE FUNCTION: Compute loss from circle parameters
// This function loops over pixels to compute statistics, then computes loss
[Differentiable]
float computeCircleLossDifferentiable(
    float circle_x,              // ← We want gradients for these 3 parameters
    float circle_y,
    float circle_radius,
    RWStructuredBuffer<float> target_mask,  // Pre-computed target mask (constant)
    no_diff uint2 image_size,    // ← Constants: no_diff = don't compute gradients
    no_diff float target_cx,     // ← Pre-computed target centroid (constant)
    no_diff float target_cy,
    no_diff float target_pixels  // ← Pre-computed target pixel count (constant)
) {
    // Step 1: Render circle and compute statistics
    float rendered_sum = 0;
    float rendered_cx_sum = 0;
    float rendered_cy_sum = 0;
    float intersection_sum = 0;
    
    // Loop over all pixels (forward pass only - Slang CAN differentiate through this!)
    // [MaxIters] tells Slang the maximum iteration count for autodiff
    // NOTE: Using smaller max to reduce GPU timeout risk
    [MaxIters(128)]  // Support up to 64x64 images (larger causes GPU timeout)
    for (uint y = 0; y < image_size.y; y++) {
        [MaxIters(128)]
        for (uint x = 0; x < image_size.x; x++) {
            // Render at this pixel
            float alpha = renderCircleAtPixel(float(x), float(y), circle_x, circle_y, circle_radius);
            
            // Convert alpha to mask value (simulating red circle)
            float red = 1.0;
            float green = 1.0 - alpha;
            float blue = 1.0 - alpha;
            float rendered_mask_val = extractCircleMaskValue(red, green, blue);
            
            // Accumulate statistics
            rendered_sum += rendered_mask_val;
            rendered_cx_sum += x * rendered_mask_val;
            rendered_cy_sum += y * rendered_mask_val;
            
            // Compute intersection with target
            uint idx = y * image_size.x + x;
            intersection_sum += target_mask[idx] * rendered_mask_val;
        }
    }
    
    // Step 2: Compute rendered centroid
    float rendered_cx = (rendered_sum > 0) ? (rendered_cx_sum / rendered_sum) : (image_size.x / 2.0);
    float rendered_cy = (rendered_sum > 0) ? (rendered_cy_sum / rendered_sum) : (image_size.y / 2.0);
    
    // Step 3: Compute union
    float union_sum = target_pixels + rendered_sum - intersection_sum;
    
    // Step 4: Use our differentiable loss formula
    return computeLossFromStatistics(
        intersection_sum, union_sum, target_pixels, rendered_sum,
        target_cx, target_cy, rendered_cx, rendered_cy,
        image_size
    );
}

// Differentiable loss formula (takes pre-computed statistics)
// This is used by the main differentiable function above
[Differentiable]
float computeLossFromStatistics(
    float intersection_pixels,
    float union_pixels,
    no_diff float target_pixels,    // ← Constant: target never changes
    float rendered_pixels,           // ← Variable: depends on circle params
    no_diff float target_cx,         // ← Constants: target centroid never changes
    no_diff float target_cy,
    float rendered_cx,               // ← Variables: rendered centroid depends on circle params
    float rendered_cy,
    no_diff uint2 image_size         // ← Constant: image size never changes
) {
    // Step 1: Compute distance loss (always needed)
    float distance_loss = computeDistanceLoss(target_cx, target_cy, rendered_cx, rendered_cy, image_size);
    
    // Step 2: Compute size loss components (inlined for better autodiff)
    const float PI = 3.14159;
    float target_radius_estimate = sqrt(target_pixels / PI);
    float rendered_radius_estimate = sqrt(rendered_pixels / PI);
    
    float min_pixels = min(target_pixels, rendered_pixels);
    float max_pixels = max(target_pixels, rendered_pixels);
    float size_ratio = (max_pixels > 0) ? (min_pixels / max_pixels) : 1.0;
    float size_penalty = 1.0 - size_ratio;
    
    float radius_diff = target_radius_estimate - rendered_radius_estimate;
    float radius_growth_incentive = max(0.0, radius_diff) / float(image_size.x);
    
    // Step 3: Compute total loss with conditional logic
    float total_loss;
    
    if (union_pixels > 0) {
        if (intersection_pixels < 1.0) {
            // No overlap: use distance loss
            total_loss = distance_loss + 0.5 * size_penalty;
        } else {
            // Has overlap: use IoU loss
            float iou = intersection_pixels / union_pixels;
            float iou_loss = 1.0 - iou;
            total_loss = iou_loss + 0.5 * size_penalty + 0.3 * radius_growth_incentive;
        }
    } else {
        // No circles: high penalty
        total_loss = distance_loss + 1.0;
    }
    
    return total_loss;
}

// ============================================================================
// Step 3B.6 Original: Non-Differentiable Version (for testing/comparison)
// ============================================================================

// Complete loss result structure
struct CompleteLossResult {
    // Mask statistics
    float intersection_pixels;
    float union_pixels;
    float target_pixels;
    float rendered_pixels;
    
    // IoU components
    float iou;
    float iou_loss;
    
    // Distance components
    float target_cx;
    float target_cy;
    float rendered_cx;
    float rendered_cy;
    float centroid_distance;
    float distance_loss;
    
    // Size components
    float target_radius_estimate;
    float rendered_radius_estimate;
    float size_ratio;
    float size_penalty;
    float radius_growth_incentive;
    
    // Final loss
    float total_loss;
    bool used_distance_loss;  // true if intersection < 1.0
};

// Compute complete loss matching circle_focused_learning.py exactly
// This combines all loss components with the same conditional logic
CompleteLossResult computeCompleteLoss(
    RWStructuredBuffer<float> target_mask,
    RWStructuredBuffer<float> rendered_mask,
    uint2 image_size
) {
    CompleteLossResult result;
    
    // Step 1: Compute IoU and pixel counts
    IoUResult iou_result = computeIoU(target_mask, rendered_mask, image_size);
    result.intersection_pixels = iou_result.intersection_pixels;
    result.union_pixels = iou_result.union_pixels;
    result.target_pixels = iou_result.target_pixels;
    result.rendered_pixels = iou_result.rendered_pixels;
    result.iou = iou_result.iou;
    result.iou_loss = iou_result.iou_loss;
    
    // Step 2: Compute centroids and distance loss
    float target_cx, target_cy;
    computeImageCentroid(target_mask, image_size, target_cx, target_cy);
    result.target_cx = target_cx;
    result.target_cy = target_cy;
    
    float rendered_cx, rendered_cy;
    computeImageCentroid(rendered_mask, image_size, rendered_cx, rendered_cy);
    result.rendered_cx = rendered_cx;
    result.rendered_cy = rendered_cy;
    
    float dx = target_cx - rendered_cx;
    float dy = target_cy - rendered_cy;
    result.centroid_distance = sqrt(dx * dx + dy * dy);
    result.distance_loss = result.centroid_distance / image_size.x;
    
    // Step 3: Compute size loss components
    SizeLossResult size_result = computeSizeLoss(result.target_pixels, result.rendered_pixels, image_size);
    result.target_radius_estimate = size_result.target_radius_estimate;
    result.rendered_radius_estimate = size_result.rendered_radius_estimate;
    result.size_ratio = size_result.size_ratio;
    result.size_penalty = size_result.size_penalty;
    result.radius_growth_incentive = size_result.radius_growth_incentive;
    
    // Step 4: Compute total loss with conditional logic
    // Matches PyTorch exactly:
    // if union_pixels > 0:
    //     if intersection_pixels < 1.0:
    //         total_loss = distance_loss + 0.5 * size_penalty + 0.3 * radius_growth_incentive
    //     else:
    //         total_loss = iou_loss + 0.5 * size_penalty + 0.3 * radius_growth_incentive
    // else:
    //     total_loss = distance_loss + 1.0
    
    if (result.union_pixels > 0) {
        if (result.intersection_pixels < 1.0) {
            // No overlap: use distance loss
            result.total_loss = result.distance_loss + 
                              0.5 * result.size_penalty + 
                              0.3 * result.radius_growth_incentive;
            result.used_distance_loss = true;
        } else {
            // Has overlap: use IoU loss
            result.total_loss = result.iou_loss + 
                              0.5 * result.size_penalty + 
                              0.3 * result.radius_growth_incentive;
            result.used_distance_loss = false;
        }
    } else {
        // No circles at all: high penalty + distance loss
        result.total_loss = result.distance_loss + 1.0;
        result.used_distance_loss = true;
    }
    
    return result;
}

// ============================================================================
// Compute Shaders for Testing
// ============================================================================

// Compute shader for testing mask extraction
// Process one pixel per thread
[shader("compute")]
[numthreads(8, 8, 1)]
void extractMaskShader(
    uint3 id : SV_DispatchThreadID,
    uniform uint2 image_size,
    RWStructuredBuffer<float> input_image,
    RWStructuredBuffer<float> output_mask
) {
    uint x = id.x;
    uint y = id.y;
    
    if (x >= image_size.x || y >= image_size.y) return;
    
    uint idx = y * image_size.x + x;
    uint img_idx = idx * 4;  // RGBA
    
    float red = input_image[img_idx + 0];
    float green = input_image[img_idx + 1];
    float blue = input_image[img_idx + 2];
    
    output_mask[idx] = extractCircleMaskAtPixel(red, green, blue);
}

// Compute shader for testing centroid computation
[shader("compute")]
[numthreads(1, 1, 1)]
void computeCentroidShader(
    uniform uint2 image_size,
    RWStructuredBuffer<float> input_mask,
    RWStructuredBuffer<float> output_centroid  // [cx, cy]
) {
    float cx, cy;
    computeImageCentroid(input_mask, image_size, cx, cy);
    
    output_centroid[0] = cx;
    output_centroid[1] = cy;
}

// Compute shader for testing IoU computation
[shader("compute")]
[numthreads(1, 1, 1)]
void computeIoUShader(
    uniform uint2 image_size,
    RWStructuredBuffer<float> target_mask,
    RWStructuredBuffer<float> rendered_mask,
    RWStructuredBuffer<float> output_iou  // [intersection, union, target_pixels, rendered_pixels, iou, iou_loss]
) {
    IoUResult result = computeIoU(target_mask, rendered_mask, image_size);
    
    output_iou[0] = result.intersection_pixels;
    output_iou[1] = result.union_pixels;
    output_iou[2] = result.target_pixels;
    output_iou[3] = result.rendered_pixels;
    output_iou[4] = result.iou;
    output_iou[5] = result.iou_loss;
}

// Compute shader for testing distance loss computation
[shader("compute")]
[numthreads(1, 1, 1)]
void computeDistanceLossShader(
    uniform uint2 image_size,
    RWStructuredBuffer<float> target_mask,
    RWStructuredBuffer<float> rendered_mask,
    RWStructuredBuffer<float> output_distance  // [target_cx, target_cy, rendered_cx, rendered_cy, centroid_distance, distance_loss]
) {
    // Compute centroids
    float target_cx, target_cy;
    computeImageCentroid(target_mask, image_size, target_cx, target_cy);
    
    float rendered_cx, rendered_cy;
    computeImageCentroid(rendered_mask, image_size, rendered_cx, rendered_cy);
    
    // Compute distance loss
    float distance_loss = computeDistanceLoss(target_cx, target_cy, rendered_cx, rendered_cy, image_size);
    
    // Compute raw centroid distance for verification
    float dx = target_cx - rendered_cx;
    float dy = target_cy - rendered_cy;
    float centroid_distance = sqrt(dx * dx + dy * dy);
    
    // Output results
    output_distance[0] = target_cx;
    output_distance[1] = target_cy;
    output_distance[2] = rendered_cx;
    output_distance[3] = rendered_cy;
    output_distance[4] = centroid_distance;
    output_distance[5] = distance_loss;
}

// Compute shader for testing size loss computation
[shader("compute")]
[numthreads(1, 1, 1)]
void computeSizeLossShader(
    uniform uint2 image_size,
    RWStructuredBuffer<float> target_mask,
    RWStructuredBuffer<float> rendered_mask,
    RWStructuredBuffer<float> output_size  // [target_pixels, rendered_pixels, target_radius, rendered_radius, size_ratio, size_penalty, radius_growth_incentive]
) {
    // First compute IoU to get pixel counts
    IoUResult iou_result = computeIoU(target_mask, rendered_mask, image_size);
    
    // Compute size loss
    SizeLossResult size_result = computeSizeLoss(iou_result.target_pixels, iou_result.rendered_pixels, image_size);
    
    // Output results
    output_size[0] = iou_result.target_pixels;
    output_size[1] = iou_result.rendered_pixels;
    output_size[2] = size_result.target_radius_estimate;
    output_size[3] = size_result.rendered_radius_estimate;
    output_size[4] = size_result.size_ratio;
    output_size[5] = size_result.size_penalty;
    output_size[6] = size_result.radius_growth_incentive;
}

// Compute shader for complete loss computation (Step 3B.6 - Original Non-Differentiable)
[shader("compute")]
[numthreads(1, 1, 1)]
void computeCompleteLossShader(
    uniform uint2 image_size,
    RWStructuredBuffer<float> target_mask,
    RWStructuredBuffer<float> rendered_mask,
    RWStructuredBuffer<float> output_loss  // [total_loss, iou_loss, distance_loss, size_penalty, radius_growth_incentive, intersection_pixels, used_distance_loss]
) {
    // Compute complete loss
    CompleteLossResult result = computeCompleteLoss(target_mask, rendered_mask, image_size);
    
    // Output main results
    output_loss[0] = result.total_loss;
    output_loss[1] = result.iou_loss;
    output_loss[2] = result.distance_loss;
    output_loss[3] = result.size_penalty;
    output_loss[4] = result.radius_growth_incentive;
    output_loss[5] = result.intersection_pixels;
    output_loss[6] = result.used_distance_loss ? 1.0 : 0.0;
}

// ============================================================================
// AUTODIFF: Compute Gradients using Slang's bwd_diff
// ============================================================================

// Compute gradients of loss w.r.t. circle parameters using autodiff
// This is the main training shader!
// Uses BACKWARD-mode autodiff (bwd_diff) to match PyTorch's loss.backward()
// This is MORE EFFICIENT: 1 backward pass gets all 3 gradients!
[shader("compute")]
[numthreads(1, 1, 1)]
void computeGradientsAutodiff(
    uniform float circle_x,
    uniform float circle_y,
    uniform float circle_radius,
    uniform uint2 image_size,
    uniform float target_cx,
    uniform float target_cy,
    uniform float target_pixels,
    RWStructuredBuffer<float> target_mask,
    RWStructuredBuffer<float> output_grads  // [loss, grad_x, grad_y, grad_radius, rendered_sum, intersection_sum, iou, size_penalty, radius_growth]
) {
    // Create differential pairs
    var dp_x = diffPair(circle_x);
    var dp_y = diffPair(circle_y);
    var dp_r = diffPair(circle_radius);
    
    // Forward pass: compute loss
    float loss = computeCircleLossDifferentiable(
        circle_x, circle_y, circle_radius,
        target_mask, image_size,
        target_cx, target_cy, target_pixels
    );
    
    // Backward pass
    bwd_diff(computeCircleLossDifferentiable)(
        dp_x, dp_y, dp_r,                      // Parameters to differentiate (diffPairs)
        target_mask, image_size,                // Buffers and constants (regular types)
        target_cx, target_cy, target_pixels,    // no_diff constants (regular floats)
        1.0  // Seed gradient: ∂loss/∂loss = 1.0
    );
    
    output_grads[0] = loss;
    output_grads[1] = dp_x.d;
    output_grads[2] = dp_y.d;
    output_grads[3] = dp_r.d;    
}

float renderCircle(float2 coord, CircleParams circle) {
    float2 center = float2(circle.x, circle.y);
    float dx = coord.x - center.x;
    float dy = coord.y - center.y;
    float dist_sq = dx * dx + dy * dy;
    float radius_sq = circle.radius * circle.radius + 1e-6;
    
    // Use sigmoid function (matches image_based_loss.slang renderCircleAtPixel)
    float alpha = sigmoid((radius_sq - dist_sq) * 10.0);
    return alpha;
}

struct CircleParams {
    float x;
    float y;
    float radius;
}

// Compute shader for rendering circles to buffer
[shader("compute")]
[numthreads(8, 8, 1)]
void renderCirclesToBuffer(
    uint3 id : SV_DispatchThreadID,
    uniform uint2 image_size,
    uniform CircleParams circle,
    RWStructuredBuffer<float> output
) {
    uint2 pixelCoord = id.xy;
    if (pixelCoord.x >= image_size.x || pixelCoord.y >= image_size.y) return;
    
    float2 coord = float2(pixelCoord);
    float alpha = renderCircle(coord, circle);
    float3 circle_color = float3(1.0, 0.0, 0.0); // Red color
    float3 background_color = float3(1.0, 1.0, 1.0); // White background
    
    // Proper alpha blending: circle_color * alpha + background_color * (1 - alpha)
    float3 final_color = circle_color * alpha + background_color * (1.0 - alpha);
    float final_alpha = 1.0; // Always fully opaque for final image
    
    // Write RGBA values to buffer
    uint idx = (pixelCoord.y * image_size.x + pixelCoord.x) * 4;
    output[idx + 0] = final_color.r;
    output[idx + 1] = final_color.g;
    output[idx + 2] = final_color.b;
    output[idx + 3] = final_alpha;
}