# SPDX-License-Identifier: Apache-2.0
#
# Step 6: Neural Block Compression
#
# Compresses a full PBR material set (albedo + normal + roughness) into a single
# shared latent texture decoded at runtime by a tiny MLP — the same principle as
# GPU block-compression (BC1/BC6H) but with a learned neural codec.
#
# Compression ratio (256x256 source, 6 channels):
#   Uncompressed: 256x256 x 6ch x 4B = 1.5 MB
#   NBC latent:   64x64  x 8ch x 4B = 128 KB  (+~6 KB weights)  =>  ~11x
#
# Metal PBR textures from: https://github.com/ChefSteveP/neural-texture-compression

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path
import urllib.request
import io

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install Pillow")

# ---------------------------------------------------------------------------
# Download / cache PBR textures as PNG files
# Textures are saved in sRGB so load_from_image(linearize=True) works for albedo.
# ---------------------------------------------------------------------------

TEXTURE_BASE = (
    "https://raw.githubusercontent.com/ChefSteveP/"
    "neural-texture-compression/main/sample_textures"
)
TEXTURE_SIZE = (256, 256)
data_path = Path(__file__).parent


def _download_png(url: str, local: Path) -> None:
    """Download a texture from url, resize, and save as PNG (sRGB)."""
    if local.exists():
        return
    print(f"Downloading {local.name} ...")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(TEXTURE_SIZE, Image.LANCZOS)
    img.save(local)


albedo_path    = data_path / "pbr_metal_albedo.png"
normal_path    = data_path / "pbr_metal_normal.png"
roughness_path = data_path / "pbr_metal_roughness.png"
material_path  = data_path / "pbr_metal_material.png"   # packed: (nx, ny, roughness)

_download_png(f"{TEXTURE_BASE}/metal-albedo.jpg",    albedo_path)
_download_png(f"{TEXTURE_BASE}/metal-normal.jpg",    normal_path)
_download_png(f"{TEXTURE_BASE}/metal-roughness.jpg", roughness_path)

# Build the packed material image (nx, ny, roughness) as a synthetic RGB PNG.
# This lets load_from_image return a Tensor<float3, 2> matching the Slang signature.
if not material_path.exists():
    normal_img    = np.array(Image.open(normal_path).resize(TEXTURE_SIZE),    dtype=np.float32) / 255.0
    roughness_img = np.array(Image.open(roughness_path).resize(TEXTURE_SIZE), dtype=np.float32) / 255.0
    packed = np.stack([normal_img[..., 0],    # nx stored in [0,1]
                       normal_img[..., 1],    # ny stored in [0,1]
                       roughness_img[..., 0]  # roughness (R channel, greyscale)
                      ], axis=-1)
    Image.fromarray((packed * 255).clip(0, 255).astype(np.uint8)).save(material_path)

# ---------------------------------------------------------------------------
# Load reference textures as Tensor<float3, 2> (compatible with blit + Slang)
# ---------------------------------------------------------------------------

# 5 panels side by side at 512px each (plus gaps) would need a 2600px-wide
# window, wider than a single 1920px-wide display can show — the OS then
# clamps the window and the rightmost panel(s) render off-screen. 320px
# keeps the whole row comfortably inside a 1920x1080 screen.
PANEL = 320
GAP   = 10

app    = App(width=PANEL * 5 + GAP * 4, height=PANEL,
             title="Neural Block Compression — metal PBR set")
module = spy.Module.load_from_file(app.device, "05_neural_block_compression.slang")

# linearize=True converts sRGB -> linear for albedo.
# linearize=False keeps raw stored values for normals / roughness.
albedo   = spy.Tensor.load_from_image(app.device, albedo_path,   linearize=True)
material = spy.Tensor.load_from_image(app.device, material_path, linearize=False)
normal   = spy.Tensor.load_from_image(app.device, normal_path,   linearize=False)

# ---------------------------------------------------------------------------
# Network wrappers  (mirrors 05 pattern exactly)
# ---------------------------------------------------------------------------

class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.biases       = spy.Tensor.from_numpy(app.device, np.zeros(outputs, dtype=np.float32))
        self.weights      = spy.Tensor.from_numpy(
            app.device, np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype(np.float32))
        self.biases_grad  = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)
        self.m_biases     = spy.Tensor.zeros_like(self.biases)
        self.m_weights    = spy.Tensor.zeros_like(self.weights)
        self.v_biases     = spy.Tensor.zeros_like(self.biases)
        self.v_weights    = spy.Tensor.zeros_like(self.weights)

    def optimize(self, lr: float, step: int):
        module.optimizer_step(self.biases,  self.biases_grad,
                              self.m_biases,  self.v_biases,  lr, step)
        module.optimizer_step(self.weights, self.weights_grad,
                              self.m_weights, self.v_weights, lr, step)


class LatentTexture(spy.InstanceList):
    def __init__(self, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])
        init = np.random.uniform(0.0, 1.0, (height, width, num_latents)).astype(np.float32)
        self.texture       = spy.Tensor.from_numpy(app.device, init)
        self.texture_grads = spy.Tensor.zeros_like(self.texture)
        self.m_texture     = spy.Tensor.zeros_like(self.texture)
        self.v_texture     = spy.Tensor.zeros_like(self.texture)

    def optimize(self, lr: float, step: int):
        module.optimizer_step(self.texture, self.texture_grads,
                              self.m_texture, self.v_texture, lr, step)


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        # 64x64 latent with 8 channels = 4x spatial downscale of 256x256 source
        self.latent_texture = LatentTexture(64, 64, 8)
        self.layer0 = NetworkParameters(8,  32)
        self.layer1 = NetworkParameters(32, 32)
        self.layer2 = NetworkParameters(32,  6)  # 3 albedo + 2 normal + 1 roughness

    def optimize(self, lr: float, step: int):
        self.latent_texture.optimize(lr, step)
        self.layer0.optimize(lr, step)
        self.layer1.optimize(lr, step)
        self.layer2.optimize(lr, step)


network = Network()

# ---------------------------------------------------------------------------
# Shader warm-up
#
# Every module.xxx() kernel is JIT-compiled by Slang the first time it's called.
# Without this warm-up, that one-time compile happens *inside* the very first
# call to app.present(), which is the point where the GPU is first forced to
# flush and synchronize everything queued so far (all render/train kernels for
# that frame) against the swapchain. On some GPU/driver combinations, bundling
# "compile every kernel for the first time" + "run a full frame" + "present"
# into a single synchronized unit takes long enough to trip Windows' TDR
# (Timeout Detection and Recovery) watchdog, which then kills the GPU device —
# even though every later frame would have been fast.
#
# Fix: compile every kernel once, at trivial size, on a disposable network,
# and explicitly wait for the GPU before the real loop (and its first present)
# ever starts. This keeps compilation off the present-timed critical path.
# ---------------------------------------------------------------------------
print("Warming up shaders (one-time compile)... this may take a while")

_warmup_network = Network()  # throwaway — never touches the real `network`'s weights
_warmup_res = spy.int2(1, 1)
_warmup_out = spy.Tensor.empty(app.device, (1, 1), "float3")

module.render_albedo(pixel=spy.call_id(), resolution=_warmup_res,
                     network=_warmup_network, _result=_warmup_out)
module.render_normal(pixel=spy.call_id(), resolution=_warmup_res,
                     network=_warmup_network, _result=_warmup_out)
module.loss(pixel=spy.call_id(), resolution=_warmup_res,
            albedo_ref=albedo, network=_warmup_network, _result=_warmup_out)
module.calculate_grads(
    seed=spy.wang_hash(seed=0, warmup=2),
    batch_index=spy.grid((1, 1)),
    batch_size=spy.int2(1, 1),
    albedo_ref=albedo,
    material_ref=material,
    network=_warmup_network,
)
_warmup_network.optimize(0.0, 1)  # lr=0.0 -> compiles optimizer_step without changing anything
app.device.wait()                 # block here, not at the first real present()

del _warmup_network, _warmup_res, _warmup_out
print("Warm-up complete.")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

optimize_counter = 0
LEARNING_RATE    = 0.001
ITERS_PER_FRAME  = 20
BATCH_SIZE       = (64, 64)
# Reconstruction error is only a few percent, so we amplify it for display.
# Raise this if panel 5 still looks mostly black, lower it if it's all red.
DIFF_GAIN        = 15.0

res = spy.int2(*TEXTURE_SIZE)

# Reference albedo in gamma space, cached once — used every frame to build the
# amplified diff panel on the CPU (no extra GPU kernel needed for that).
albedo_ref_srgb = np.clip(albedo.to_numpy(), 0.0, 1.0) ** (1.0 / 2.2)


def _amplified_diff_panel(neural_albedo: "spy.Tensor") -> "spy.Tensor":
    """Amplified, heatmapped albedo error, computed in NumPy from tensors we
    already have on hand. Deliberately avoids adding a new GPU shader kernel —
    this panel is purely a display aid, not part of the render/train pipeline."""
    decoded_srgb = np.clip(neural_albedo.to_numpy(), 0.0, 1.0) ** (1.0 / 2.2)
    mag = np.abs(decoded_srgb - albedo_ref_srgb).max(axis=-1)   # worst channel
    t = np.clip(mag * DIFF_GAIN, 0.0, 1.0)

    # Black -> blue -> green -> yellow -> red heatmap
    stops  = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=np.float32)
    colors = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)
    idx  = np.clip(np.searchsorted(stops, t, side="right") - 1, 0, len(stops) - 2)
    frac = ((t - stops[idx]) / (stops[idx + 1] - stops[idx]))[..., None]
    heat = colors[idx] + (colors[idx + 1] - colors[idx]) * frac

    # Build as Tensor<float3, 2> (like `albedo`), not a plain 3D float tensor.
    result = spy.Tensor.empty(app.device, heat.shape[:2], "float3")
    result.copy_from_numpy(np.ascontiguousarray(heat.astype(np.float32)))
    return result

# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------
H, W          = TEXTURE_SIZE
NUM_CHANNELS  = 6          # albedo RGB + normal XY + roughness
LATENT_W      = 64
LATENT_H      = 64
NUM_LATENTS   = 8
LAYER_SIZES   = [(NUM_LATENTS, 32), (32, 32), (32, NUM_CHANNELS)]

uncompressed_bytes = H * W * NUM_CHANNELS * 4
latent_bytes       = LATENT_W * LATENT_H * NUM_LATENTS * 4
weight_bytes       = sum((inp * out + out) * 4 for inp, out in LAYER_SIZES)
compressed_bytes   = latent_bytes + weight_bytes
compression_ratio  = uncompressed_bytes / compressed_bytes

print("Compiling shaders... this may take a while")
print("Panels: [ref albedo] [neural albedo] [ref normal] [neural normal] [amplified albedo diff]")
print()
print(f"  Uncompressed : {uncompressed_bytes:>8,} bytes  ({uncompressed_bytes/1024:.1f} KB)")
print(f"  Latent tex   : {latent_bytes:>8,} bytes  ({latent_bytes/1024:.1f} KB)  [{LATENT_W}x{LATENT_H} x {NUM_LATENTS}ch]")
print(f"  MLP weights  : {weight_bytes:>8,} bytes  ({weight_bytes/1024:.1f} KB)")
print(f"  Compressed   : {compressed_bytes:>8,} bytes  ({compressed_bytes/1024:.1f} KB)")
print(f"  Ratio        : {compression_ratio:.1f}x")
print()

while app.process_events():

    # ---- Forward passes for display ----
    neural_albedo = spy.Tensor.empty_like(albedo)
    module.render_albedo(pixel=spy.call_id(), resolution=res,
                         network=network, _result=neural_albedo)

    neural_normal = spy.Tensor.empty_like(albedo)
    module.render_normal(pixel=spy.call_id(), resolution=res,
                         network=network, _result=neural_normal)

    loss_out = spy.Tensor.empty_like(albedo)
    module.loss(pixel=spy.call_id(), resolution=res,
                albedo_ref=albedo, network=network, _result=loss_out)

    diff_vis = _amplified_diff_panel(neural_albedo)

    # ---- Blit five panels ----
    offset = 0
    # Panel 1: reference albedo
    app.blit(albedo,        size=spy.int2(PANEL), offset=spy.int2(offset, 0),
             tonemap=False, bilinear=True)
    offset += PANEL + GAP
    # Panel 2: neural-decoded albedo
    app.blit(neural_albedo, size=spy.int2(PANEL), offset=spy.int2(offset, 0),
             tonemap=False, bilinear=True)
    offset += PANEL + GAP
    # Panel 3: reference normal map
    app.blit(normal,        size=spy.int2(PANEL), offset=spy.int2(offset, 0),
             tonemap=False, bilinear=True)
    offset += PANEL + GAP
    # Panel 4: neural-decoded normal map
    app.blit(neural_normal, size=spy.int2(PANEL), offset=spy.int2(offset, 0),
             tonemap=False, bilinear=True)
    offset += PANEL + GAP
    # Panel 5: amplified, heatmapped albedo error (black/blue = match, red = high error)
    app.blit(diff_vis,      size=spy.int2(PANEL), offset=spy.int2(offset, 0),
             tonemap=False, bilinear=True)

    # ---- Training ----
    for _ in range(ITERS_PER_FRAME):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(BATCH_SIZE),
            batch_size=spy.int2(BATCH_SIZE),
            albedo_ref=albedo,
            material_ref=material,
            network=network,
        )
        optimize_counter += 1
        network.optimize(LEARNING_RATE, optimize_counter)

    mean_loss = float(np.mean(loss_out.to_numpy()))
    print(f"Step {optimize_counter:6d} | albedo loss: {mean_loss:.5f} | compression: {compression_ratio:.1f}x")

    app.present()
