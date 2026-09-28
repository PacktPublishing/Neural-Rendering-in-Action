"""
Section 1: Gradient-Based Optimization with Differentiable Rendering

This demo shows how Slang's automatic differentiation lets us recover an unknown material parameter from a rendered
image without ever writing derivative code by hand.

Pipeline :
  1. Render a reference image of an icosphere with a known material (red).
  2. Perturb the material to a wrong value (blue).
  3. Use Slang's bwd_diff to compute dLoss/dAlbedo through the full render.
  4. Run Adam to recover the original material, watching the sphere turn from blue to red.

Controls:
  Space - Pause / resume optimization
  R - Reset to initial (blue) material
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
from mesh_utils import make_icosphere, faces_to_flat_vertices

# ============================================================================
# Step 0: Setup
# ============================================================================

app = App(
    title="Section 1: Differentiable Material Recovery",
    width=1024,
    height=512,
    device_type=spy.DeviceType.automatic,
    include_paths=[Path(__file__).parent],
)
# Side-by-side: each half is 512x512 (square)
module = spy.Module.load_from_file(app.device, "section1.slang")

# ============================================================================
# Step 1: Build the scene
# ============================================================================

vertices_indexed, faces = make_icosphere(subdivisions=2, radius=1.0)
flat_verts = faces_to_flat_vertices(vertices_indexed, faces)
num_tris = len(faces)
num_flat_verts = num_tris * 3

verts_tensor = spy.Tensor.zeros(app.device, dtype=spy.float3, shape=(num_flat_verts,))
verts_tensor.copy_from_numpy(flat_verts.reshape(-1))

print(f"Scene: icosphere with {num_tris} triangles")

# Camera
cam_pos = np.array([0.0, 0.5, -3.0], dtype=np.float32)
cam_fwd = np.array([0.0, 0.0, 0.0], dtype=np.float32) - cam_pos
cam_fwd = cam_fwd / np.linalg.norm(cam_fwd)
world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
cam_right = np.cross(cam_fwd, world_up)
cam_right = cam_right / np.linalg.norm(cam_right)
cam_up = np.cross(cam_right, cam_fwd)

camera = {
    "_type": "Camera",
    "position": cam_pos.tolist(),
    "forward": cam_fwd.tolist(),
    "up": cam_up.tolist(),
    "right": cam_right.tolist(),
    "fov": 0.8,
    "aspect": 1.0,
}

lights = [
    {"_type": "PointLight", "position": [3.0, 4.0, -2.0],
     "color": [1.0, 0.95, 0.9], "intensity": 15.0},
    {"_type": "PointLight", "position": [-2.0, 3.0, -3.0],
     "color": [0.7, 0.8, 1.0], "intensity": 10.0},
]

# ============================================================================
# Step 2: Render the reference image (the "ground truth" we want to recover)
# ============================================================================

TARGET_ALBEDO = np.array([0.8, 0.3, 0.2], dtype=np.float32)
TARGET_ROUGHNESS = 0.3

target_material = {
    "_type": "Material",
    "albedo": TARGET_ALBEDO.tolist(),
    "roughness": TARGET_ROUGHNESS,
    "metallic": 0.0,
}

RES = 256
resolution = spy.float2(RES, RES)

# Render a tone-mapped reference for the side-by-side display.
target_display = spy.Tensor.empty(app.device, dtype=spy.float4, shape=(RES, RES))
module.renderScene(
    camera, call_id(), resolution,
    verts_tensor, num_tris,
    target_material, lights, 2,
    _result=target_display,
)

# Reusable texture for rendering the current material each frame.
current_display = spy.Tensor.empty(app.device, dtype=spy.float4, shape=(RES, RES))

print(f"Reference rendered with albedo = {TARGET_ALBEDO.tolist()}")

# ============================================================================
# Step 3: Perturb — set the material to a wrong value
# ============================================================================

INIT_ALBEDO = np.array([0.0, 0.0, 1.0], dtype=np.float32)

current_albedo = INIT_ALBEDO.copy()

print(f"Initial (wrong) albedo  = {INIT_ALBEDO.tolist()}")
print(f"Target albedo           = {TARGET_ALBEDO.tolist()}")

# ============================================================================
# Step 4: Optimization loop — Adam + Slang autodiff
# ============================================================================

# Adam state
LR = 0.05
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
m_albedo = np.zeros(3, dtype=np.float32)
v_albedo = np.zeros(3, dtype=np.float32)

# GPU tensor for gradient accumulation (AtomicTensor<float3, 1> of size 1)
albedo_grad_tensor = spy.Tensor.zeros(app.device, dtype=spy.float3, shape=(1,))

NUM_ITERATIONS = 100
DELAY_PER_ITER = 0.08  # seconds between steps (so the color change is visible)
total_iter = 0
paused = False
losses = []
albedo_history = [INIT_ALBEDO.copy()]
optimization_done = False


def make_material():
    return {
        "_type": "Material",
        "albedo": current_albedo.tolist(),
        "roughness": TARGET_ROUGHNESS,
        "metallic": 0.0,
    }


def reset():
    global current_albedo, m_albedo, v_albedo, total_iter, losses, albedo_history
    current_albedo = INIT_ALBEDO.copy()
    m_albedo = np.zeros(3, dtype=np.float32)
    v_albedo = np.zeros(3, dtype=np.float32)
    total_iter = 0
    losses = []
    albedo_history = [INIT_ALBEDO.copy()]


print(f"\nOptimizer: Adam (lr={LR})")
print(f"Running up to {NUM_ITERATIONS} iterations of gradient-based optimization.\n")

print("Controls:")
print("  Space - Pause / resume")
print("  R - Reset to blue\n")


def on_key(event: spy.KeyboardEvent):
    global paused
    if event.type == spy.KeyboardEventType.key_press:
        if event.key == spy.KeyCode.space:
            paused = not paused
            print(f"{'PAUSED' if paused else 'RESUMED'}")
        elif event.key == spy.KeyCode.r:
            reset()
            print("Reset to initial blue material.")


app.on_keyboard_event = on_key

# ---- Main loop ----

while app.process_events():
    resolution_display = spy.float2(app._window.width, app._window.height)

    # --- Optimization step ---
    if not paused and total_iter < NUM_ITERATIONS:
        mat = make_material()

        # Zero the gradient accumulator
        albedo_grad_tensor.copy_from_numpy(np.zeros(3, dtype=np.float32))

        # Backward pass: Slang's bwd_diff computes dLoss/dAlbedo through the
        # entire render pipeline (ray casting -> intersection -> shading -> loss).
        # Both rendered and target colors are computed inside the same AD code path
        # (pixelLossMat), ensuring zero gradient when materials match exactly.
        module.computeAlbedoGradients(
            camera, spy.grid(shape=(RES, RES)), resolution,
            verts_tensor, num_tris,
            mat, target_material,
            lights, 2,
            albedo_grad_tensor,
        )

        # Read the accumulated gradient and normalize to per-pixel mean
        grad_raw = albedo_grad_tensor.to_numpy().reshape(3)
        grad = grad_raw / (RES * RES)

        # Adam update
        total_iter += 1
        m_albedo = BETA1 * m_albedo + (1 - BETA1) * grad
        v_albedo = BETA2 * v_albedo + (1 - BETA2) * grad ** 2
        m_hat = m_albedo / (1 - BETA1 ** total_iter)
        v_hat = v_albedo / (1 - BETA2 ** total_iter)
        current_albedo -= LR * m_hat / (np.sqrt(v_hat) + EPS)

        # Clamp to valid color range
        current_albedo = np.clip(current_albedo, 0.0, 1.0)

        # Track convergence
        param_error = float(np.sum((current_albedo - TARGET_ALBEDO) ** 2))
        losses.append(param_error)
        albedo_history.append(current_albedo.copy())

        print(f"Iteration {total_iter:3d}: "
              f"albedo = [{current_albedo[0]:.3f}, {current_albedo[1]:.3f}, {current_albedo[2]:.3f}]  "
              f"parameter error = {param_error:.6f}", end="\r")

        if total_iter == NUM_ITERATIONS:
            optimization_done = True
            print(f"\nOptimization complete after {NUM_ITERATIONS} iterations.")
            print(f"  Recovered: [{current_albedo[0]:.3f}, {current_albedo[1]:.3f}, {current_albedo[2]:.3f}]")
            print(f"  Target:    {TARGET_ALBEDO.tolist()}")

        time.sleep(DELAY_PER_ITER)

    # --- Display ---
    cur_mat = make_material()

    # Pre-render current material at the target resolution (standalone dispatch).
    # This ensures identical code paths for both left and right in side-by-side.
    module.renderScene(
        camera, call_id(), resolution,
        verts_tensor, num_tris,
        cur_mat, lights, 2,
        _result=current_display,
    )

    module.renderSideBySide(
        call_id(), resolution_display,
        target_display, current_display, resolution,
        _result=app.output,
    )

    app.present()

# ============================================================================
# Step 5: Results — show convergence plot
# ============================================================================

if losses:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: parameter error over iterations
    ax1.plot(losses, linewidth=2, color="#1f77b4")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("MSE(param)")
    ax1.set_title("Parameter error plot")
    ax1.grid(True, alpha=0.3)

    # Right: per-channel albedo trajectory toward the target
    history = np.array(albedo_history)
    ax2.plot(history[:, 0], label="Red", color="red", linewidth=2)
    ax2.plot(history[:, 1], label="Green", color="green", linewidth=2)
    ax2.plot(history[:, 2], label="Blue", color="blue", linewidth=2)
    ax2.axhline(TARGET_ALBEDO[0], color="red", linestyle="--", alpha=0.5)
    ax2.axhline(TARGET_ALBEDO[1], color="green", linestyle="--", alpha=0.5)
    ax2.axhline(TARGET_ALBEDO[2], color="blue", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Albedo value")
    ax2.set_title("Albedo channel convergence")
    ax2.legend(loc="center right", fontsize=8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(__file__).parent / "convergence_plot.png"
    plt.savefig(str(plot_path), dpi=150)
    print(f"Convergence plot saved to {plot_path.name}")
    plt.show()
