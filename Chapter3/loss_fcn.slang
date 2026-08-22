// Image-Based Loss for Circle Learning

// ============================================================================
// Helper Functions
// ============================================================================

// Sigmoid activation function (differentiable)
// MUST be [Differentiable] because it's called inside renderCircleAtPixel,
// which computeGradientsAutodiff differentiates.
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
    uint2 image_size                            // ← Image size is constant
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
    uint2 image_size          // ← Image size is constant
) {
    SizeLossResult result;
    
    // Estimate radii from pixel counts (Area = pi * r^2, so r = sqrt(Area / pi))
    const float PI = 3.14159;
    result.target_radius_estimate = sqrt(target_pixels / PI);
    result.rendered_radius_estimate = sqrt(rendered_pixels / PI);
    
    // Size agreement, as a smooth symmetric ratio rather than min/max.
    //   2*a*b / (a^2 + b^2)  is 1 exactly when a == b, falls off either side,
    //   and is symmetric in a and b, so it behaves like min/max - but min/max
    // has a corner precisely where the two counts are equal, which is where
    // the optimizer is trying to come to rest. The corner makes dLoss/dradius
    // jump sign across the solution and the radius chatters instead of
    // settling. This form is differentiable there, with zero slope at the
    // match, so the radius can actually land on it.
    float sum_squares = target_pixels * target_pixels + rendered_pixels * rendered_pixels;
    result.size_ratio = (sum_squares > 0) ? (2.0 * target_pixels * rendered_pixels / sum_squares) : 1.0;

    // Equivalently (target - rendered)^2 / (target^2 + rendered^2).
    result.size_penalty = 1.0 - result.size_ratio;

    // Radius growth incentive (only if rendered is smaller than target).
    // Squared ReLU, not plain max(0, x), for the same reason: t^2 has slope
    // 2t, which is continuous through t == 0, whereas max(0, t) hands the
    // radius a slope of 1 on one side and 0 on the other at the exact point
    // the sizes agree. Normalized by image_size^2 to stay dimensionless.
    float radius_diff = max(0.0, result.target_radius_estimate - result.rendered_radius_estimate);
    result.radius_growth_incentive = (radius_diff * radius_diff)
                                   / float(image_size.x * image_size.x);
    
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
// 1. computeGradientsAutodiff calls bwd_diff on it, once per pixel
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

// Differentiable loss formula, in terms of image statistics rather than pixels.
// computeGradientsAutodiff differentiates this to get dLoss/d(statistics), then
// pushes those derivatives back to the circle parameters one pixel at a time.
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
    uint2 image_size         // ← Constant: image size never changes
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
//
// Differentiating in two phases costs one hand-written chain rule:
//   phase 1  - accumulate image statistics with no autodiff at all
//   phase 2a - bwd_diff the *statistics* -> dLoss/d{I, U, R, rcx, rcy}
//   phase 2b - bwd_diff one pixel at a time, seeded with dLoss/dalpha_i
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
    RWStructuredBuffer<float> output_grads  // [loss, grad_x, grad_y, grad_radius]
) {
    // ---- Phase 1: forward statistics (plain code, no autodiff) ----
    float rendered_sum = 0;
    float cx_sum = 0;
    float cy_sum = 0;
    float intersection_sum = 0;

    for (uint y = 0; y < image_size.y; y++) {
        for (uint x = 0; x < image_size.x; x++) {
            float alpha = renderCircleAtPixel(float(x), float(y), circle_x, circle_y, circle_radius);

            rendered_sum += alpha;
            cx_sum += x * alpha;
            cy_sum += y * alpha;
            intersection_sum += target_mask[y * image_size.x + x] * alpha;
        }
    }

    float rendered_cx = (rendered_sum > 0) ? (cx_sum / rendered_sum) : (image_size.x / 2.0);
    float rendered_cy = (rendered_sum > 0) ? (cy_sum / rendered_sum) : (image_size.y / 2.0);
    float union_sum = target_pixels + rendered_sum - intersection_sum;

    float loss = computeLossFromStatistics(
        intersection_sum, union_sum, target_pixels, rendered_sum,
        target_cx, target_cy, rendered_cx, rendered_cy,
        image_size
    );

    // ---- Phase 2a: dLoss/d(statistics) ----
    var dp_I = diffPair(intersection_sum);
    var dp_U = diffPair(union_sum);
    var dp_R = diffPair(rendered_sum);
    var dp_cx = diffPair(rendered_cx);
    var dp_cy = diffPair(rendered_cy);

    bwd_diff(computeLossFromStatistics)(
        dp_I, dp_U, target_pixels, dp_R,
        target_cx, target_cy, dp_cx, dp_cy,
        image_size,
        1.0  // Seed gradient: dloss/dloss = 1.0
    );

    // ---- Phase 2b: dLoss/d(circle params), one differentiated pixel at a time ----
    // Chain rule from the statistics down to each pixel's alpha:
    //   dI/dalpha_i   = mask_i
    //   dU/dalpha_i   = 1 - mask_i      (U = target_pixels + R - I)
    //   dR/dalpha_i   = 1
    //   drcx/dalpha_i = (x_i - rcx) / R (rcx = sum(x*alpha) / R, quotient rule)
    //   drcy/dalpha_i = (y_i - rcy) / R
    float inv_R = (rendered_sum > 0) ? (1.0 / rendered_sum) : 0.0;

    float grad_x = 0;
    float grad_y = 0;
    float grad_r = 0;

    for (uint y = 0; y < image_size.y; y++) {
        for (uint x = 0; x < image_size.x; x++) {
            float m = target_mask[y * image_size.x + x];

            float dLoss_dAlpha = dp_I.d * m
                               + dp_U.d * (1.0 - m)
                               + dp_R.d
                               + dp_cx.d * (float(x) - rendered_cx) * inv_R
                               + dp_cy.d * (float(y) - rendered_cy) * inv_R;

            var dp_x = diffPair(circle_x);
            var dp_y = diffPair(circle_y);
            var dp_r = diffPair(circle_radius);

            bwd_diff(renderCircleAtPixel)(float(x), float(y), dp_x, dp_y, dp_r, dLoss_dAlpha);

            grad_x += dp_x.d;
            grad_y += dp_y.d;
            grad_r += dp_r.d;
        }
    }

    output_grads[0] = loss;
    output_grads[1] = grad_x;
    output_grads[2] = grad_y;
    output_grads[3] = grad_r;
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