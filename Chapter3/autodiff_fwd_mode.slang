import "slangpy";

[Differentiable]
float simpleFunction(float x, float y)
{
    // f(x,y) = x² + 2xy + y²
    return x * x + 2.0f * x * y + y * y;
}

// Compute function value and derivative w.r.t. x
[shader("compute")]
[numthreads(1, 1, 1)]
void computeForwardDerivative(uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform float x,
    uniform float y,
    RWStructuredBuffer<float> output)
{
    // Forward mode: compute derivative w.r.t. x
    // Create differential pair: (value, derivative)
    let x_pair = diffPair(x, 1.0f);  // dx/dx = 1.0 (we're differentiating w.r.t. x)
    let y_pair = diffPair(y, 0.0f);  // dy/dx = 0.0 (y is constant w.r.t. x)
    
    // Use fwd_diff to compute forward-mode derivative
    // This automatically applies the chain rule!
    let result = fwd_diff(simpleFunction)(x_pair, y_pair);
    
    output[0] = result.p;  // Primal value (function value)
    output[1] = result.d;  // Differential value (df/dx)
    
    // If we wanted df/dy instead, we'd swap the differential parts:
    // let x_pair = diffPair(x, 0.0f);
    // let y_pair = diffPair(y, 1.0f);
}