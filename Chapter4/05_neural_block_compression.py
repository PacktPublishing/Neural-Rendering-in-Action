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

app    = App(width=512 * 4 + 10 * 3, height=512,
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
# Main loop
# ---------------------------------------------------------------------------

optimize_counter = 0
LEARNING_RATE    = 0.001
ITERS_PER_FRAME  = 20
BATCH_SIZE       = (64, 64)
PANEL            = 512
GAP              = 10

res = spy.int2(*TEXTURE_SIZE)

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
print("Panels: [ref albedo] [neural albedo] [ref normal] [neural normal]")
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

    # ---- Blit four panels ----
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
