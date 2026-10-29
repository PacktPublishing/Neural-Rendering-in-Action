# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np
from pathlib import Path

# Create the app and load the slang module.
app = App(width=512 * 3 + 10 * 2, height=512, title="Simple MLP with SIREN Activations")
module = spy.Module.load_from_file(app.device, "02_siren_activation.slang")

# Load some materials.
data_path = Path(__file__).parent
_image_raw = spy.Tensor.load_from_image(app.device, data_path.joinpath("test_texture.png"), linearize=True)
_arr = _image_raw.to_numpy()
if _arr.ndim == 3 and _arr.shape[2] == 4:
    _arr = np.ascontiguousarray(_arr[:, :, :3])
    image = spy.Tensor.empty(app.device, _arr.shape[:2], "float3")
    image.copy_from_numpy(_arr)
else:
    image = _image_raw

OMEGA_0 = 30.0


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int, is_first_layer: bool = False):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs

        # SIREN initialization (Sitzmann et al. 2020)
        # First layer: U(-1/fan_in, 1/fan_in)
        # Hidden layers: U(-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0)
        if is_first_layer:
            w_limit = 1.0 / inputs
        else:
            w_limit = np.sqrt(6.0 / inputs) / OMEGA_0

        self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            app.device, np.random.uniform(-w_limit, w_limit, (outputs, inputs)).astype("float32")
        )

        # Gradients for the biases and weights.
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)

        # Temp data for Adam optimizer.
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.biases,
            self.biases_grad,
            self.m_biases,
            self.v_biases,
            learning_rate,
            optimize_counter,
        )
        module.optimizer_step(
            self.weights,
            self.weights_grad,
            self.m_weights,
            self.v_weights,
            learning_rate,
            optimize_counter,
        )


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.layer0 = NetworkParameters(2, 32, is_first_layer=True)
        self.layer1 = NetworkParameters(32, 32)
        self.layer2 = NetworkParameters(32, 3)

    def optimize(self, learning_rate: float, optimize_counter: int):
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


network = Network()

optimize_counter = 0

print("Compiling shaders... this may take a while")

while app.process_events():

    offset = 0
    # Blit reference texture
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10
    res = spy.int2(256, 256)
    batch_size = (64, 64)

    # Render current neural texture
    lr_output = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=res, network=network, _result=lr_output)
    app.blit(lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10

    # Show loss between neural texture and reference texture.
    loss_output = spy.Tensor.empty_like(image)
    module.loss(
        pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_output
    )
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)

    learning_rate = 0.001

    for i in range(20):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            batch_index=spy.grid(batch_size),
            batch_size=spy.int2(batch_size),
            reference=image,
            network=network,
        )
        optimize_counter += 1

        network.optimize(learning_rate, optimize_counter)

    print(f"Step {optimize_counter:6d} | Loss: {np.mean(loss_output.to_numpy()):.5f}")

    # Present the window.
    app.present()
