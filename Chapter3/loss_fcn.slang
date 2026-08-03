// Image-Based Loss for Circle Learning

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
// Step 3.1: Extract Circle Mask from Image
// ============================================================================

// Extract circle mask at a single pixel
// Detects red circle pixels (high red, low green/blue)
// Returns value between 0 (background) and 1 (circle)
// This is only needed for the TARGET, where all we have is an RGB image.
// The rendered side gets its coverage straight out of renderCircleAtPixel and
// deliberately does not round-trip through colour - see the note there.
float extractCircleMaskValue(float red, float green, float blue) {
    float red_strength = sigmoid((red - 0.5) * 20.0);
    float green_penalty = sigmoid((0.5 - green) * 20.0);
    float blue_penalty = sigmoid((0.5 - blue) * 20.0);

    return red_strength * green_penalty * blue_penalty;
}

// ============================================================================
// Step 3.2: Compute Image Centroid
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
// Step 3.4: Compute Distance Loss
// ============================================================================

// Compute distance between two centroids, normalized by image size (differentiable)
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
// Step 3.5: Compute Size Loss and Radius Growth Incentive
// ============================================================================

// Result structure for size loss computation
// Must conform to IDifferentiable: computeLossFromStatistics returns a value
// derived from this struct, and without the conformance Slang silently drops
// the derivative w.r.t. circle_radius (dr collapses to ~0 in the no-overlap
// branch, where size_penalty is the only radius-dependent term).
struct SizeLossResult : IDifferentiable {
    float target_radius_estimate;
    float rendered_radius_estimate;
    float size_ratio;
    float size_penalty;
    float radius_growth_incentive;
};

// Compute size-based losses from mask pixel counts (differentiable)
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
// Step 3.6: Unified Loss Computation (All Components Combined)
// ============================================================================

// Controls how wide the soft edge of the circle is, in pixels.
// This is the single most important number in this file. All of the gradient
// that reaches circle_x/y/radius comes from the band of pixels where this
// sigmoid is actually transitioning, so that band has to be at least a pixel
// or two wide. If it is narrower than the pixel spacing, the grid steps clean
// over it and the measured gradient is alternately enormous and exactly zero.
// 2.0 gives a ~2.2 pixel 10%-90% transition: wide enough that several pixels
// always sample it, narrow enough that the rendered disc still looks crisp.
static const float EDGE_SHARPNESS = 2.0;

// Render circle at a single pixel (differentiable)
// Returns alpha value [0, 1] for smooth circle rendering
// MUST be [Differentiable] because:
// 1. Called inside computeCircleLossDifferentiable
// 2. Takes circle_x, circle_y, circle_radius as parameters (the vars we optimize!)
// 3. Without this, ∂loss/∂circle_x, ∂loss/∂circle_y, ∂loss/∂circle_radius cannot be computed
[Differentiable]
float renderCircleAtPixel(no_diff float px, no_diff float py, float circle_x, float circle_y, float circle_radius) {
    float dx = px - circle_x;
    float dy = py - circle_y;

    // Distance to the centre, NOT squared distance. With squared distance the
    // width of the transition shrinks as 1/(2r), so it falls below one pixel
    // as soon as the circle is of any size and the gradient stops being
    // measurable. The epsilon keeps d(sqrt)/dx finite at the exact centre.
    float dist = sqrt(dx * dx + dy * dy + 1e-8);

    // Smooth circle edge using sigmoid: positive inside, negative outside.
    float alpha = sigmoid((circle_radius - dist) * EDGE_SHARPNESS);
    return alpha;
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
            
            float rendered_mask_val = alpha;
            
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
    
    // Step 2: Compute size loss components (single definition)
    SizeLossResult size = computeSizeLoss(target_pixels, rendered_pixels, image_size);
    float size_penalty = size.size_penalty;
    float radius_growth_incentive = size.radius_growth_incentive;

    // Step 3: Compute total loss with conditional logic
    float total_loss;
    
    if (union_pixels > 0) {
        if (intersection_pixels < 1.0) {
            // No overlap: use distance loss.
            // The +1.0 puts this branch above the overlap branch, whose
            // iou_loss can never exceed 1.0. 
            total_loss = 1.0 + distance_loss + 0.5 * size_penalty;
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
    
    output_mask[idx] = extractCircleMaskValue(red, green, blue);
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

// ============================================================================
// AUTODIFF: Compute Gradients using Slang's bwd_diff
// ============================================================================

// Compute gradients of loss w.r.t. circle parameters using autodiff
// This is the main training shader!
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

// Thin adapter so the displayed image and the optimized mask can never drift
// apart: both go through renderCircleAtPixel, which is the single definition
// of the circle's shape.
float renderCircle(float2 coord, CircleParams circle) {
    return renderCircleAtPixel(coord.x, coord.y,
                               circle.x, circle.y, circle.radius);
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