import slangpy as spy
import pathlib
import numpy as np

if __name__ == "__main__":

    # Create a SlangPy device and use the local folder for Slang includes
    device = spy.create_device(include_paths=[
            pathlib.Path(__file__).parent.absolute(),
    ])

    # Load the module
    module = spy.Module.load_from_file(device, "autodiff_fwd_mode.slang")

    slang_module = device.load_module(module.name)

    shader_program = device.link_program([slang_module], [slang_module.entry_point("computeForwardDerivative")])    
    kernel = device.create_compute_kernel(shader_program)

    output_buffer = device.create_buffer(element_count=2, resource_type_layout=kernel.reflection.computeForwardDerivative.output, usage=spy.BufferUsage.unordered_access)
    kernel.dispatch(thread_count=[1, 1, 1], x = 3.0, y = 4.0, output = output_buffer)

    print(output_buffer.to_numpy().view(np.float32))
