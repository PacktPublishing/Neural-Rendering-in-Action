import "slangpy";

[Differentiable]
float neuralStyleFunction(float x, float y, float w1, float w2, float w3)
{
    // Simple function with multiple parameters
    float h = w1 * x + w2 * y;  // Linear layer
    float activated = h * h;     // Quadratic activation
    float output = w3 * activated; // Output layer
    return output;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeBackwardGradients(uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform float x,
    uniform float y,
    uniform float w1,
    uniform float w2,
    uniform float w3,
    RWStructuredBuffer<float> output)
{
    // In backward mode, we propagate derivatives from output to inputs
    // This is more efficient when we have many parameters
    
    // Initialize all inputs as differential pairs
    var dpair_x = diffPair(x, 0.0f);
    var dpair_y = diffPair(y, 0.0f);
    var dpair_w1 = diffPair(w1, 0.0f);
    var dpair_w2 = diffPair(w2, 0.0f);
    var dpair_w3 = diffPair(w3, 0.0f);
    
    // Backward pass: compute gradients
    // bwd_diff propagates gradient from output (1.0) back to all differentiable inputs
    bwd_diff(neuralStyleFunction)(
        dpair_x, dpair_y, 
        dpair_w1, dpair_w2, dpair_w3,
        1.0f  // Output gradient (dLoss/dOutput = 1.0)
    );
    
    // Compute forward value separately
    let fwd_result = neuralStyleFunction(x, y, w1, w2, w3);
    
    output[0] = fwd_result;        // Function value
    output[1] = dpair_w1.d;        // df/dw1
    output[2] = dpair_w2.d;        // df/dw2
    output[3] = dpair_w3.d;        // df/dw3
}