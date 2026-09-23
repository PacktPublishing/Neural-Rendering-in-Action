import slangpy as spy
import pathlib
import numpy as np
import torch

if __name__ == "__main__":

    # Create a SlangPy device and use the local folder for Slang includes
    device = spy.create_device(include_paths=[
            pathlib.Path(__file__).parent.absolute(),
    ])

    # Load the module
    module = spy.Module.load_from_file(device, "autodiff_backward_mode.slang")

    slang_module = device.load_module(module.name)

    shader_program = device.link_program([slang_module], [slang_module.entry_point("computeBackwardGradients")])    
    kernel = device.create_compute_kernel(shader_program)

    output_buffer = device.create_buffer(element_count=4, resource_type_layout=kernel.reflection.computeBackwardGradients.output, usage=spy.BufferUsage.unordered_access)
    kernel.dispatch(thread_count=[1, 1, 1], x = 3.0, y = 4.0, w1 = 0.5, w2 = 0.7, w3 = 1.2, output = output_buffer)

    print(output_buffer.to_numpy().view(np.float32))

