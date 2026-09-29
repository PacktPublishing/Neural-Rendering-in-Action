# SPDX-License-Identifier: Apache-2.0
"""Compare neural texture representations from this chapter on one figure.

Trains three of the chapter's techniques from scratch, headless (no window),
against mandrill.tiff, then renders a 1x4 comparison figure:

    1. Original texture       (mandrill.tiff, ground truth)
    2. SIREN                  (02_siren_activation.slang)
    3. Frequency encoding     (03_frequency_encoding.slang)
    4. Latent texture         (04_latent_texture.slang)

The fourth panel is the last representation in the chapter that reconstructs
this same single texture. 05_neural_block_compression.slang extends the same
latent-texture idea to a *set* of shared PBR materials (albedo/normal/
roughness) with one decoder -- a different comparison shape, so it isn't a
drop-in fourth panel here.

Usage:
    python make_technique_comparison.py [--steps 4000] [--seed 0]

Takes a few minutes: three independent from-scratch training runs plus
shader compilation.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import slangpy as spy

HERE = Path(__file__).parent
FIGURES = HERE / "figures"

BATCH_SIZE = (64, 64)
LEARNING_RATE = 0.001


def load_reference(device: spy.Device) -> spy.Tensor:
    raw = spy.Tensor.load_from_image(device, HERE / "mandrill.png", linearize=True)
    arr = raw.to_numpy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = np.ascontiguousarray(arr[:, :, :3])
        image = spy.Tensor.empty(device, arr.shape[:2], "float3")
        image.copy_from_numpy(arr)
        return image
    return raw


class NetworkParameters(spy.InstanceList):
    """Shared across techniques -- only the weight-init limit differs (SIREN)."""

    def __init__(self, module: spy.Module, device: spy.Device, inputs: int, outputs: int, w_limit: float = 0.5):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self._module = module
        self.biases = spy.Tensor.from_numpy(device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            device, np.random.uniform(-w_limit, w_limit, (outputs, inputs)).astype("float32")
        )
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    def optimize(self, learning_rate: float, optimize_counter: int):
        self._module.optimizer_step(
            self.biases, self.biases_grad, self.m_biases, self.v_biases, learning_rate, optimize_counter
        )
        self._module.optimizer_step(
            self.weights, self.weights_grad, self.m_weights, self.v_weights, learning_rate, optimize_counter
        )


class LatentTexture(spy.InstanceList):
    def __init__(self, module: spy.Module, device: spy.Device, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])
        self._module = module
        initial = np.random.uniform(0.0, 1.0, (height, width, num_latents)).astype("float32")
        self.texture = spy.Tensor.from_numpy(device, initial)
        self.texture_grads = spy.Tensor.zeros_like(self.texture)
        self.m_texture = spy.Tensor.zeros_like(self.texture)
        self.v_texture = spy.Tensor.zeros_like(self.texture)

    def optimize(self, learning_rate: float, optimize_counter: int):
        self._module.optimizer_step(
            self.texture, self.texture_grads, self.m_texture, self.v_texture, learning_rate, optimize_counter
        )


class SirenNetwork(spy.InstanceList):
    OMEGA_0 = 30.0

    def __init__(self, module: spy.Module, device: spy.Device):
        super().__init__(module["Network"])
        # SIREN init (Sitzmann et al. 2020): first layer U(-1/fan_in, 1/fan_in),
        # hidden layers U(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0).
        self.layer0 = NetworkParameters(module, device, 2, 32, w_limit=1.0 / 2)
        self.layer1 = NetworkParameters(module, device, 32, 32, w_limit=np.sqrt(6.0 / 32) / self.OMEGA_0)
        self.layer2 = NetworkParameters(module, device, 32, 3, w_limit=np.sqrt(6.0 / 32) / self.OMEGA_0)

    def optimize(self, learning_rate: float, optimize_counter: int):
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


class FrequencyNetwork(spy.InstanceList):
    def __init__(self, module: spy.Module, device: spy.Device):
        super().__init__(module["Network"])
        self.layer0 = NetworkParameters(module, device, 16, 32)
        self.layer1 = NetworkParameters(module, device, 32, 32)
        self.layer2 = NetworkParameters(module, device, 32, 3)

    def optimize(self, learning_rate: float, optimize_counter: int):
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


class LatentNetwork(spy.InstanceList):
    def __init__(self, module: spy.Module, device: spy.Device):
        super().__init__(module["Network"])
        self.latent_texture = LatentTexture(module, device, 64, 64, 4)
        self.layer0 = NetworkParameters(module, device, 4, 32)
        self.layer1 = NetworkParameters(module, device, 32, 32)
        self.layer2 = NetworkParameters(module, device, 32, 3)

    def optimize(self, learning_rate: float, optimize_counter: int):
        self.latent_texture.optimize(learning_rate, optimize_counter)
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


def train(name: str, module: spy.Module, network, image: spy.Tensor, steps: int, log_every: int = 500):
    resolution = spy.int2(image.shape[1], image.shape[0])
    optimize_counter = 0
    print(f"[{name}] training for {steps} steps...")
    for step in range(steps):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(BATCH_SIZE),
            batch_size=spy.int2(BATCH_SIZE),
            reference=image,
            network=network,
        )
        optimize_counter += 1
        network.optimize(LEARNING_RATE, optimize_counter)

        if (step + 1) % log_every == 0 or step == steps - 1:
            loss_output = spy.Tensor.empty_like(image)
            module.loss(pixel=spy.call_id(), resolution=resolution, network=network, reference=image, _result=loss_output)
            print(f"[{name}] step {step + 1:6d}/{steps} | loss {np.mean(loss_output.to_numpy()):.5f}")

    final = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=resolution, network=network, _result=final)
    return final.to_numpy()


def to_display(linear_rgb: np.ndarray) -> np.ndarray:
    """Reference/network output are linear (matching linearize=True on load);
    gamma-correct the same way the loss does before comparing so the figure
    shows what the loss actually optimized for, not a too-dark linear image."""
    return np.clip(linear_rgb, 0.0, None) ** (1.0 / 2.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000, help="Adam steps per technique")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=FIGURES / "fig_texture_comparison.png")
    args = parser.parse_args()

    np.random.seed(args.seed)

    device = spy.create_device(spy.DeviceType.d3d12, include_paths=[HERE])
    image = load_reference(device)

    panels = [("Original texture", to_display(image.to_numpy()))]

    siren_module = spy.Module.load_from_file(device, str(HERE / "02_siren_activation.slang"))
    siren_net = SirenNetwork(siren_module, device)
    panels.append(("SIREN", to_display(train("SIREN", siren_module, siren_net, image, args.steps))))

    frequency_module = spy.Module.load_from_file(device, str(HERE / "03_frequency_encoding.slang"))
    frequency_net = FrequencyNetwork(frequency_module, device)
    panels.append(("Frequency encoding", to_display(train("Frequency encoding", frequency_module, frequency_net, image, args.steps))))

    latent_module = spy.Module.load_from_file(device, str(HERE / "04_latent_texture.slang"))
    latent_net = LatentNetwork(latent_module, device)
    panels.append(("Latent texture", to_display(train("Latent texture", latent_module, latent_net, image, args.steps))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03, wspace=0.03)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved {args.output}]")


if __name__ == "__main__":
    main()
