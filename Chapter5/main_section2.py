"""
Section 2: Sphere Tracing with Differentiable Distance Functions

Demonstrates:
  - Differentiable SDF primitives (sphere, box, torus)
  - Smooth CSG operations (union, intersection, subtraction)
  - Differentiable sphere tracing (ray marching) for rendering
  - Inverse optimization: recover SDF shape parameters using Slang's bwd_diff

Pipeline:
  1. Render a reference SDF scene with known parameters.
  2. Perturb the parameters to wrong values (shapes move/resize).
  3. Compute gradients of an SDF-space loss (comparing distance field values
     at thousands of 3D sample points) using Slang's automatic differentiation.
  4. Run Adam to recover the original parameters, watching the shapes
     smoothly converge to their target positions.

  The SDF-space loss avoids sphere tracing discontinuities entirely -- evalScene
  is smoothly differentiable everywhere, giving reliable gradients for all
  16 parameters simultaneously.

Controls:
  1 - Side-by-side view (reference | current)
  2 - Current scene only
  Space - Pause / resume optimization
  R - Reset to initial (perturbed) parameters
  Esc - Quit
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from common import App
import slangpy as spy
import numpy as np
from slangpy.types import call_id

# ============================================================================
# Setup
# ============================================================================

app = App(
    title="Section 2: Differentiable SDF Optimization",
    width=1024,
    height=512,
    device_type=spy.DeviceType.automatic,
    include_paths=[Path(__file__).parent],
)
module = spy.Module.load_from_file(app.device, "sdf.slang")

# ============================================================================
# Scene definition
# ============================================================================

cam_pos = np.array([0.0, 1.5, -5.0], dtype=np.float32)
cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
cam_forward = cam_target - cam_pos
cam_forward /= np.linalg.norm(cam_forward)
world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
cam_right = np.cross(cam_forward, world_up)
cam_right /= np.linalg.norm(cam_right)
cam_up = np.cross(cam_right, cam_forward)

camera = {
    "_type": "Camera",
    "position": cam_pos.tolist(),
    "forward": cam_forward.tolist(),
    "up": cam_up.tolist(),
    "right": cam_right.tolist(),
    "fov": 0.8,
    "aspect": 1.0,
}

material = {
    "_type": "Material",
    "albedo": [0.7, 0.4, 0.2],
    "roughness": 0.25,
    "metallic": 0.1,
}

lights = [
    {"_type": "PointLight", "position": [4.0, 5.0, -3.0],
     "color": [1.0, 0.95, 0.9], "intensity": 20.0},
    {"_type": "PointLight", "position": [-3.0, 3.0, -4.0],
     "color": [0.7, 0.8, 1.0], "intensity": 12.0},
]

# ============================================================================
# Target SDF parameters (ground truth)
# ============================================================================

target_params = {
    "_type": "SDFSceneParams",
    "sphereCenter": [0.0, 0.5, 0.0],
    "sphereRadius": 0.8,
    "boxCenter": [1.2, 0.0, 0.5],
    "boxHalfExtents": [0.5, 0.5, 0.5],
    "torusCenter": [-1.0, 0.3, -0.3],
    "torusRadii": [0.6, 0.2],
    "smoothK": 0.3,
}

RENDER_RES = 256
render_resolution = spy.float2(RENDER_RES, RENDER_RES)

target_display = spy.Tensor.empty(app.device, dtype=spy.float4, shape=(RENDER_RES, RENDER_RES))
module.renderSDFScene(
    camera, call_id(), render_resolution,
    target_params, material, lights, 2,
    _result=target_display,
)
current_display = spy.Tensor.empty(app.device, dtype=spy.float4, shape=(RENDER_RES, RENDER_RES))

print("Target SDF image rendered.")

# ============================================================================
# 3D sample grid for SDF-space loss
# ============================================================================

GRID_N = 32
coords = np.linspace(-3.0, 3.0, GRID_N, dtype=np.float32)
gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
samples_np = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
NUM_SAMPLES = samples_np.shape[0]

samples_tensor = spy.Tensor.zeros(app.device, dtype=spy.float3, shape=(NUM_SAMPLES,))
samples_tensor.copy_from_numpy(samples_np.ravel())
print(f"SDF sample grid: {GRID_N}^3 = {NUM_SAMPLES} points in [-3, 3]^3")

# ============================================================================
# Perturbed initial parameters
# ============================================================================

np.random.seed(42)
PERTURBATION = 0.4


def perturb_list(vals, scale):
    return (np.array(vals, dtype=np.float32) + np.random.randn(len(vals)).astype(np.float32) * scale).tolist()


def perturb_scalar(val, scale):
    return float(val + np.random.randn() * scale)


INIT_PARAMS = {
    "_type": "SDFSceneParams",
    "sphereCenter": perturb_list([0.5, 0.5, 0.0], PERTURBATION),
    "sphereRadius": perturb_scalar(0.2, 0.2),
    "boxCenter": perturb_list([1.2, 1.0, 0.5], PERTURBATION),
    "boxHalfExtents": np.abs(np.array([0.5, 0.5, 0.5]) + np.random.randn(3) * 0.15).tolist(),
    "torusCenter": perturb_list([-2.0, 0.3, -0.3], PERTURBATION),
    "torusRadii": np.abs(np.array([0.6, 0.2]) + np.random.randn(2) * 0.1).tolist(),
    "smoothK": abs(perturb_scalar(0.3, 0.1)),
}

PARAM_KEYS = [
    ("sphereCenter", 3),
    ("sphereRadius", 1),
    ("boxCenter", 3),
    ("boxHalfExtents", 3),
    ("torusCenter", 3),
    ("torusRadii", 2),
    ("smoothK", 1),
]
NUM_PARAMS = 16


def params_to_vec(p):
    vec = np.zeros(NUM_PARAMS, dtype=np.float32)
    idx = 0
    for key, dim in PARAM_KEYS:
        if dim == 1:
            vec[idx] = float(p[key])
        else:
            for d in range(dim):
                vec[idx + d] = float(p[key][d])
        idx += dim
    return vec


def vec_to_params(vec):
    p = {"_type": "SDFSceneParams"}
    idx = 0
    for key, dim in PARAM_KEYS:
        if dim == 1:
            p[key] = float(vec[idx])
        else:
            p[key] = [float(vec[idx + d]) for d in range(dim)]
        idx += dim
    return p


target_vec = params_to_vec(target_params)
current_vec = params_to_vec(INIT_PARAMS)
init_vec = current_vec.copy()

print(f"Target sphere center:  {target_params['sphereCenter']}")
print(f"Initial sphere center: {[f'{x:.3f}' for x in INIT_PARAMS['sphereCenter']]}")

# ============================================================================
# Optimizer state
# ============================================================================

LR = 0.01
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
m_params = np.zeros(NUM_PARAMS, dtype=np.float32)
v_params = np.zeros(NUM_PARAMS, dtype=np.float32)

grad_tensor = spy.Tensor.zeros(app.device, dtype=spy.float4, shape=(4,))

NUM_ITERATIONS = 1000
DELAY_PER_ITER = 0.05
total_iter = 0
paused = False
view_mode = 0
losses = []
optimization_done = False

print(f"\nOptimizer: Adam (lr={LR}), SDF-space loss on {NUM_SAMPLES} sample points")
print(f"Running up to {NUM_ITERATIONS} iterations.\n")
print("Controls:")
print("  1 - Side-by-side (reference | current)")
print("  2 - Current scene only")
print("  Space - Pause / resume")
print("  R - Reset to initial parameters\n")


def reset():
    global current_vec, m_params, v_params, total_iter, losses, optimization_done
    current_vec = init_vec.copy()
    m_params = np.zeros(NUM_PARAMS, dtype=np.float32)
    v_params = np.zeros(NUM_PARAMS, dtype=np.float32)
    total_iter = 0
    losses = []
    optimization_done = False


def on_key(event: spy.KeyboardEvent):
    global paused, view_mode
    if event.type == spy.KeyboardEventType.key_press:
        if event.key == spy.KeyCode.key1:
            view_mode = 0
        elif event.key == spy.KeyCode.key2:
            view_mode = 1
        elif event.key == spy.KeyCode.space:
            paused = not paused
            print(f"{'PAUSED' if paused else 'RESUMED'}")
        elif event.key == spy.KeyCode.r:
            reset()
            print("Reset to initial perturbed parameters.")


app.on_keyboard_event = on_key

# ============================================================================
# Main loop
# ============================================================================

while app.process_events():
    resolution_display = spy.float2(app._window.width, app._window.height)

    # --- Optimization step ---
    if not paused and total_iter < NUM_ITERATIONS:
        cur_params = vec_to_params(current_vec)

        # Zero the gradient accumulator and run backward pass through evalScene
        grad_tensor.copy_from_numpy(np.zeros((4, 4), dtype=np.float32))
        module.computeSDFGradients(
            samples_tensor,
            cur_params, target_params,
            grad_tensor,
        )

        grad_raw = grad_tensor.to_numpy().reshape(NUM_PARAMS)
        grad = grad_raw / NUM_SAMPLES

        # Adam update
        total_iter += 1
        m_params = BETA1 * m_params + (1 - BETA1) * grad
        v_params = BETA2 * v_params + (1 - BETA2) * grad ** 2
        m_hat = m_params / (1 - BETA1 ** total_iter)
        v_hat = v_params / (1 - BETA2 ** total_iter)
        current_vec -= LR * m_hat / (np.sqrt(v_hat) + EPS)

        # Clamp positive-only parameters
        idx = 0
        for key, dim in PARAM_KEYS:
            if key in ("sphereRadius", "smoothK"):
                current_vec[idx:idx + dim] = np.maximum(current_vec[idx:idx + dim], 0.01)
            elif key in ("boxHalfExtents", "torusRadii"):
                current_vec[idx:idx + dim] = np.maximum(current_vec[idx:idx + dim], 0.05)
            idx += dim

        param_error = float(np.sum((current_vec - target_vec) ** 2))
        losses.append(param_error)

        if total_iter % 10 == 0 or total_iter == 1:
            cur_p = vec_to_params(current_vec)
            print(f"Iter {total_iter:3d}: param_error = {param_error:.6f}, "
                  f"sphere = {[f'{x:.3f}' for x in cur_p['sphereCenter']]}")

        if total_iter == NUM_ITERATIONS:
            optimization_done = True
            final_p = vec_to_params(current_vec)
            print(f"\nOptimization complete after {NUM_ITERATIONS} iterations.")
            print(f"  Sphere:  {[f'{x:.3f}' for x in final_p['sphereCenter']]}  "
                  f"(target: {target_params['sphereCenter']})")
            print(f"  Box:     {[f'{x:.3f}' for x in final_p['boxCenter']]}  "
                  f"(target: {target_params['boxCenter']})")
            print(f"  Torus:   {[f'{x:.3f}' for x in final_p['torusCenter']]}  "
                  f"(target: {target_params['torusCenter']})")
            print(f"  Radius:  {final_p['sphereRadius']:.3f}  (target: {target_params['sphereRadius']})")
            print(f"  smoothK: {final_p['smoothK']:.3f}  (target: {target_params['smoothK']})")

        time.sleep(DELAY_PER_ITER)

    # --- Display ---
    cur_params = vec_to_params(current_vec)

    module.renderSDFScene(
        camera, call_id(), render_resolution,
        cur_params, material, lights, 2,
        _result=current_display,
    )

    if view_mode == 0:
        module.renderSDFSideBySide(
            call_id(), resolution_display,
            target_display, current_display, render_resolution,
            _result=app.output,
        )
    elif view_mode == 1:
        module.renderSDFScene(
            camera, call_id(), resolution_display,
            cur_params, material, lights, 2,
            _result=app.output,
        )

    app.present()

# ============================================================================
# Convergence plot
# ============================================================================

if losses:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(losses, linewidth=2, color="#1f77b4")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter Error (MSE)")
    ax.set_title("SDF Shape Parameter Convergence")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plot_path = Path(__file__).parent / "convergence_plot_s2.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"Convergence plot saved to {plot_path.name}")
    plt.show()
