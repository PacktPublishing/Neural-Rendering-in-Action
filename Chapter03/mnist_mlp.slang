// SPDX-License-Identifier: Apache-2.0
// MNIST MLP in Slang - Recipe for Chapter 3: Neural Rendering Toolkit
// Architecture: 784 → 128 → 64 → 10

import slangpy;

// Network configuration
public static const int input_size = 784;   // 28x28 flattened
public static const int hidden1_size = 128;
public static const int hidden2_size = 64;
public static const int output_size = 10;   // 10 digits

// ============================================================================
// Layer Implementation
// ============================================================================

public struct Layer<int input_size, int output_size>
{
    internal float* m_weights;
    internal float* m_biases;
    internal float* m_weights_grad;
    internal float* m_biases_grad;
    
    static const float LEAKY_RELU_ALPHA = 0.01f;
    
    internal __init(
        float* inWeights, 
        float* inBiases, 
        float* inWeightsGrad, 
        float* inBiasesGrad
    )
    {
        m_weights = inWeights;
        m_biases = inBiases;
        m_weights_grad = inWeightsGrad;
        m_biases_grad = inBiasesGrad;
    }
    
    // Forward pass: output = weights * input + biases
    // Applies LeakyReLU activation
    [BackwardDerivative(backward)]
    [NoDiffThis]
    public float[output_size] forward(float[input_size] input)
    {
        float[output_size] output;
        
        [MaxIters(256)]  // Max output_size is 128
        for (int i = 0; i < output_size; i++) {
            float sum = 0.0f;
            
            [MaxIters(1024)]  // Max input_size is 784
            for (int j = 0; j < input_size; j++) {
                sum += m_weights[i * input_size + j] * input[j];
            }
            
            output[i] = sum + m_biases[i];
            
            // Apply LeakyReLU activation
            if (output[i] < 0.0f) {
                output[i] *= LEAKY_RELU_ALPHA;
            }
        }
        
        return output;
    }
    
    // Backward pass - manually compute gradients
    public void backward(
        inout DifferentialPair<float[input_size]> input, 
        float[output_size] dResult
    )
    {
        float pre_act = 0.0f;
        
        // Gradient through LeakyReLU
        [MaxIters(256)]  // Max output_size is 128
        for (int i = 0; i < output_size; i++)
        {
            float sum = 0.0f;
            [MaxIters(1024)]  // Max input_size is 784
            for (int j = 0; j < input_size; j++) {
                sum += m_weights[i * input_size + j] * input.p[j];
            }
            pre_act = sum + m_biases[i];
            
            if (pre_act < 0.0f) {
                dResult[i] *= LEAKY_RELU_ALPHA;
            }
        }
        
        // Gradient of weights: += input^T * dResult  
        [MaxIters(1024)]  // Max input_size is 784
        for (int i = 0; i < input_size; i++)
        {
            [MaxIters(256)]  // Max output_size is 128
            for (int j = 0; j < output_size; j++)
            {
                // Atomic add for thread-safe gradient accumulation
                __atomic_add(
                    m_weights_grad[j * input_size + i], 
                    dResult[j] * input.p[i]
                );
            }
        }
        
        // Gradient of biases: += dResult
        [MaxIters(256)]  // Max output_size is 128
        for (int i = 0; i < output_size; i++)
        {
            __atomic_add(m_biases_grad[i], dResult[i]);
        }
        
        // Gradient of input: weight^T * dResult
        float[input_size] dInput = {};
        
        [MaxIters(1024)]  // Max input_size is 784
        for (int i = 0; i < input_size; i++)
        {
            float sum = 0.0f;
            [MaxIters(256)]  // Max output_size is 128
            for (int j = 0; j < output_size; j++)
            {
                sum += m_weights[j * input_size + i] * dResult[j];
            }
            dInput[i] += sum;
        }
        
        input = {input.p, dInput};
    }
}

// ============================================================================
// MLP Network Structure
// ============================================================================

public struct MNIST_MLP
{
    internal Layer<input_size, hidden1_size> m_layer1;
    internal Layer<hidden1_size, hidden2_size> m_layer2;
    internal Layer<hidden2_size, output_size> m_layer3;
    
    // Construct MLP from parameter buffer
    public __init(MNIST_MLP_Params* params)
    {
        const int layer1_total = (input_size * hidden1_size) + hidden1_size;
        const int layer2_total = (hidden1_size * hidden2_size) + hidden2_size;
        
        // Layer 1: 784 → 128
        {
            const int weights_size = input_size * hidden1_size;
            const int biases_offset = weights_size;
            
            m_layer1 = Layer<input_size, hidden1_size>(
                params->m_params, 
                params->m_params + biases_offset,
                params->m_grads, 
                params->m_grads + biases_offset
            );
        }
        
        // Layer 2: 128 → 64
        {
            const int weights_size = hidden1_size * hidden2_size;
            const int biases_offset = layer1_total + weights_size;
            
            m_layer2 = Layer<hidden1_size, hidden2_size>(
                params->m_params + layer1_total,
                params->m_params + biases_offset,
                params->m_grads + layer1_total,
                params->m_grads + biases_offset
            );
        }
        
        // Layer 3: 64 → 10
        {
            const int prev_layers = layer1_total + layer2_total;
            const int weights_size = hidden2_size * output_size;
            const int biases_offset = prev_layers + weights_size;
            
            m_layer3 = Layer<hidden2_size, output_size>(
                params->m_params + prev_layers,
                params->m_params + biases_offset,
                params->m_grads + prev_layers,
                params->m_grads + biases_offset
            );
        }
    }
    
    // Forward pass through entire network
    [Differentiable]
    [NoDiffThis]
    public float[output_size] eval(float[input_size] input)
    {
        float[hidden1_size] hidden1 = m_layer1.forward(input);
        float[hidden2_size] hidden2 = m_layer2.forward(hidden1);
        float[output_size] output = m_layer3.forward(hidden2);
        return output;
    }
}

// ============================================================================
// Parameter Management
// ============================================================================

public struct MNIST_MLP_Params
{
    public float* m_params;  // All weights and biases
    public float* m_grads;   // All gradients
}

// ============================================================================
// Adam Optimizer
// ============================================================================

public struct AdamState
{
    public float mean;
    public float variance;
    public float epsilon;
    public int iteration;
    public float learning_rate;
}

internal struct AdamOptimizer
{
    internal static void update(
        inout float param, 
        inout float grad, 
        inout AdamState state, 
        int batch_size, 
        float beta1, 
        float beta2
    )
    {
        // Average gradient over batch
        grad = grad / batch_size;
        
        // Gradient clipping to prevent exploding gradients
        float grad_clip = 5.0f;
        grad = clamp(grad, -grad_clip, grad_clip);
        
        // Update Adam moments
        state.mean = beta1 * state.mean + (1.0f - beta1) * grad;
        state.variance = beta2 * state.variance + (1.0f - beta2) * grad * grad;
        state.iteration++;
        
        // Bias correction
        float m_hat = state.mean / (1.0f - pow(beta1, state.iteration));
        float v_hat = state.variance / (1.0f - pow(beta2, state.iteration));
        float denom = sqrt(v_hat) + state.epsilon;
        
        // Update parameter
        param -= state.learning_rate * m_hat / denom;
        
        // Clear gradient
        grad = 0.0f;
    }
}

// Clear Adam state (call at initialization)
public void clearAdamState(inout AdamState state)
{
    state.mean = 0.0f;
    state.variance = 0.0f;
    state.iteration = 0;
    state.learning_rate = 0.001f;  // Default learning rate
    state.epsilon = 1e-8f;
}

// Update parameters using Adam optimizer
public void updateParams(
    MNIST_MLP_Params* params, 
    inout AdamState state, 
    int batch_size, 
    float beta1, 
    float beta2
)
{
    // Get current thread ID
    int call_id = CallShapeInfo::get_call_id().shape[0];
    
    // Update single parameter using Adam
    AdamOptimizer::update(
        params->m_params[call_id], 
        params->m_grads[call_id], 
        state, 
        batch_size, 
        beta1, 
        beta2
    );
}

// ============================================================================
// Loss Function (Cross-Entropy)
// ============================================================================

// Softmax function (for probabilities)
[Differentiable]
internal float[output_size] softmax(float[output_size] logits)
{
    float[output_size] probs = {};
    
    // Find max for numerical stability (no_diff since it's just for numerical stability)
    float max_logit = no_diff logits[0];

    for (int i = 1; i < output_size; i++) {
        if (no_diff logits[i] > max_logit) {
            max_logit = no_diff logits[i];
        }
    }
    
    // Compute exp and sum
    float sum = 0.0f;

    for (int i = 0; i < output_size; i++) {
        probs[i] = exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize
    for (int i = 0; i < output_size; i++) {
        probs[i] /= sum;
    }
    
    return probs;
}

// Cross-entropy loss (differentiable)
[Differentiable]
internal float cross_entropy_loss(
    inout MNIST_MLP mlp, 
    float[input_size] input, 
    int target_label  // target_label is constant, no gradients needed
)
{
    // Forward pass
    float[output_size] logits = mlp.eval(input);
    
    // Compute softmax probabilities (differentiable!)
    float[output_size] probs = softmax(logits);
    
    // Cross-entropy: -log(p[target])
    // Add small epsilon to prevent log(0)
    float target_prob = probs[target_label] + 1e-8f;
    float loss = -log(target_prob);
    
    return loss;
}

// ============================================================================
// Training Function
// ============================================================================

// Train on a single sample (batched - SlangPy dispatches once per sample)
public float trainSample(
    MNIST_MLP_Params* params,
    float[input_size] input,
    no_diff float label_as_float  // Label passed as float for batching, converted to int
)
{
    // Convert label back to int
    int target_label = int(label_as_float);
    
    // Create MLP from parameters
    MNIST_MLP mlp = MNIST_MLP(params);
    
    // Create differential pair ONLY for input (we want gradients w.r.t. network params)
    var d_input = diffPair(input);
    
    // Backward pass - compute all gradients!
    bwd_diff(cross_entropy_loss)(mlp, d_input, target_label, 1.0f);
    
    // Return current loss (for monitoring)
    float current_loss = cross_entropy_loss(mlp, input, target_label);
    return current_loss;
}

// ============================================================================
// Inference Function
// ============================================================================

// Predict class for a single sample
public int predictSample(
    MNIST_MLP_Params* params,
    float[input_size] input
)
{
    MNIST_MLP mlp = MNIST_MLP(params);
    float[output_size] logits = mlp.eval(input);
    
    // Find argmax
    int predicted_class = 0;
    float max_logit = logits[0];
    
    for (int i = 1; i < output_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            predicted_class = i;
        }
    }
    
    return predicted_class;
}

