#!/usr/bin/env python3
"""
Circle fitting with Slang autodiff, gradients computed one thread per pixel.

The gradient of the loss w.r.t. the circle parameters is produced by three
dispatches rather than a single [numthreads(1,1,1)] kernel looping over the
image. They have to be three separate dispatches: the statistics must be
complete before any gradient can be computed, and a compute shader has no
device-wide barrier inside a single dispatch.

    accumulateStatistics   one thread per pixel, atomics into stats[]
    finalizeLoss           one thread, loss + dLoss/d(statistics)
    accumulatePixelGrads   one thread per pixel, atomics into grads[]

All three go into one command buffer, so a training iteration still costs a
single submit.

    python 02_match_target_image.py
"""
import slangpy as spy
import pathlib
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class MatchTargetImage:
    def __init__(self, image_size=128):
        self.image_size = image_size
        
        print(f"Initializing MatchTargetImage...")
        print(f"Image size: {image_size}x{image_size}")
        
        self.device = spy.create_device(include_paths=[
            pathlib.Path(__file__).parent.absolute(),
        ])
        
        self.load_shaders()
        
        self.output_dir = "results"
        os.makedirs(self.output_dir, exist_ok=True)
        abs_output_dir = os.path.abspath(self.output_dir)
        print(f"Output directory: {abs_output_dir}")
        
        self.create_target_image()
        
        self.initialize_parameters()
        
        print("Setup complete!")
    
    def load_shaders(self):
        print("Loading Slang shaders...")

        # Load image-based loss shader with autodiff
        self.image_loss_slang_module = self.device.load_module("loss_fcn.slang")

        def kernel(entry_point):
            program = self.device.link_program(
                [self.image_loss_slang_module],
                [self.image_loss_slang_module.entry_point(entry_point)]
            )
            return self.device.create_compute_kernel(program)

        self.circle_kernel = kernel("renderCirclesToBuffer")
        self.mask_extraction_kernel = kernel("extractMaskShader")
        self.centroid_sums_kernel = kernel("accumulateCentroidSums")
        self.finalize_centroid_kernel = kernel("finalizeCentroid")

        # The three gradient kernels. They have to be three separate dispatches:
        # the statistics must be complete before any gradient can be computed,
        # and a compute shader has no device-wide barrier inside a dispatch.
        self.stats_kernel = kernel("accumulateStatistics")
        self.finalize_kernel = kernel("finalizeLoss")
        self.pixel_grads_kernel = kernel("accumulatePixelGrads")

        # Accumulators, allocated once and cleared on the GPU each iteration.
        def buffer(n):
            return self.device.create_buffer(
                element_count=n, struct_size=4,
                usage=spy.BufferUsage.unordered_access)

        self.stats_buffer = buffer(4)   # rendered_sum, cx_sum, cy_sum, intersection_sum
        self.ctx_buffer = buffer(9)     # loss, dLoss/d(statistics), centroid, 1/R
        self.grads_buffer = buffer(3)   # grad_x, grad_y, grad_r

        print("Shaders loaded successfully!")
        print("="*60)

    def create_target_image(self):
        print("Creating target image...")
        
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.float32)
        
        center_x, center_y = 3 * self.image_size // 4, 3 * self.image_size // 4
        radius = 15
        color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        self.target_center_x = float(center_x)
        self.target_center_y = float(center_y)
        self.target_radius = float(radius)
        
        y_coords, x_coords = np.meshgrid(
            np.arange(self.image_size, dtype=np.float32),
            np.arange(self.image_size, dtype=np.float32),
            indexing='ij'
        )
        
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        mask = distances <= radius
        image[mask] = color
        
        target_path = os.path.join(self.output_dir, "target.png")
        print(f"Saving target image...")
        self.save_image(image, target_path)
        
        self.target_image = image
        
        self.precompute_target_statistics()
        
        print(f"Target parameters: x={self.target_center_x:.1f}, y={self.target_center_y:.1f}, r={self.target_radius:.1f}")
    
    def precompute_target_statistics(self):
        print("Precomputing target statistics...")
        
        # Convert target image to RGBA for Slang
        target_rgba = np.concatenate([self.target_image, np.ones((self.image_size, self.image_size, 1), dtype=np.float32)], axis=2)
        target_rgba_flat = target_rgba.flatten()
        
        image_buffer = self.device.create_buffer(
            element_count=len(target_rgba_flat),
            struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
            data=target_rgba_flat
        )
        
        mask_buffer = self.device.create_buffer(
            element_count=self.image_size * self.image_size,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access
        )

        self.mask_extraction_kernel.dispatch(
            thread_count=[self.image_size, self.image_size, 1],
            image_size=(self.image_size, self.image_size),
            input_image=image_buffer,
            output_mask=mask_buffer
        )
        
        target_mask_data = mask_buffer.to_numpy().view(np.float32)
        
        centroid_buffer = self.device.create_buffer(
            element_count=2,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access
        )

        # Sums for the centroid reduction: [total_weight, weighted_x, weighted_y]
        centroid_sums = self.device.create_buffer(
            element_count=3,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access
        )

        # Accumulate one thread per pixel, then divide in a second dispatch.
        # Both go into one command buffer, so this is still a single submit.
        encoder = self.device.create_command_encoder()
        encoder.clear_buffer(centroid_sums)

        self.centroid_sums_kernel.dispatch(
            thread_count=[self.image_size, self.image_size, 1],
            command_encoder=encoder,
            image_size=(self.image_size, self.image_size),
            mask=mask_buffer,
            centroid_sums=centroid_sums
        )

        self.finalize_centroid_kernel.dispatch(
            thread_count=[1, 1, 1],
            command_encoder=encoder,
            image_size=(self.image_size, self.image_size),
            centroid_sums=centroid_sums,
            output_centroid=centroid_buffer
        )

        self.device.submit_command_buffer(encoder.finish())
        
        centroid_data = centroid_buffer.to_numpy().view(np.float32)
        
        self.target_mask_buffer = mask_buffer 
        
        self.target_cx = float(centroid_data[0])
        self.target_cy = float(centroid_data[1])
        self.target_pixels = float(np.sum(target_mask_data))
        
        print(f"Target statistics: cx={self.target_cx:.4f}, cy={self.target_cy:.4f}, pixels={self.target_pixels:.1f}")
    
    def initialize_parameters(self):
        # Deliberately a poor guess: small, and diagonally opposite the target,
        # so the run has to travel to converge.
        self.current_params = {
            'x': 10.0,
            'y': 10.0,
            'radius': 5.0
        }
        
        print(f"Initial: x={self.current_params['x']:.1f}, y={self.current_params['y']:.1f}, r={self.current_params['radius']:.1f}")
    
    def render_circle(self, params):
        output_buffer = self.device.create_buffer(
            element_count=self.image_size * self.image_size * 4,
            struct_size=4,
            usage=spy.BufferUsage.unordered_access
        )
        
        self.circle_kernel.dispatch(
            thread_count=[self.image_size, self.image_size, 1],
            output=output_buffer,
            image_size=(self.image_size, self.image_size),
            circle={'x': params['x'], 'y': params['y'], 'radius': params['radius']}
        )
        
        return output_buffer
    
    def compute_autodiff_gradients(self, params):
        size = (self.image_size, self.image_size)
        threads = [self.image_size, self.image_size, 1]

        # All three dispatches go into one command buffer, so this still costs a
        # single submit per training iteration.
        encoder = self.device.create_command_encoder()
        encoder.clear_buffer(self.stats_buffer)
        encoder.clear_buffer(self.grads_buffer)

        self.stats_kernel.dispatch(
            thread_count=threads,
            command_encoder=encoder,
            circle_x=params['x'],
            circle_y=params['y'],
            circle_radius=params['radius'],
            image_size=size,
            target_mask=self.target_mask_buffer,
            stats=self.stats_buffer,
        )

        self.finalize_kernel.dispatch(
            thread_count=[1, 1, 1],
            command_encoder=encoder,
            image_size=size,
            target_cx=self.target_cx,
            target_cy=self.target_cy,
            target_pixels=self.target_pixels,
            stats=self.stats_buffer,
            ctx=self.ctx_buffer,
        )

        self.pixel_grads_kernel.dispatch(
            thread_count=threads,
            command_encoder=encoder,
            circle_x=params['x'],
            circle_y=params['y'],
            circle_radius=params['radius'],
            image_size=size,
            target_mask=self.target_mask_buffer,
            ctx=self.ctx_buffer,
            grads=self.grads_buffer,
        )

        self.device.submit_command_buffer(encoder.finish())

        grad_data = self.grads_buffer.to_numpy().view(np.float32)
        ctx_data = self.ctx_buffer.to_numpy().view(np.float32)

        return {
            'loss': float(ctx_data[0]),   # the loss is produced by finalizeLoss
            'dx': float(grad_data[0]),
            'dy': float(grad_data[1]),
            'dr': float(grad_data[2]),
        }

    def update_parameters(self, gradients, learning_rate=0.5, momentum=0.9):
        if not hasattr(self, 'velocity'):
            self.velocity = {'x': 0.0, 'y': 0.0, 'radius': 0.0}
        
        self.velocity['x'] = momentum * self.velocity['x'] - learning_rate * gradients['dx']
        self.velocity['y'] = momentum * self.velocity['y'] - learning_rate * gradients['dy']
        self.velocity['radius'] = momentum * self.velocity['radius'] - learning_rate * gradients['dr']
        
        self.current_params['x'] += self.velocity['x']
        self.current_params['y'] += self.velocity['y']
        self.current_params['radius'] += self.velocity['radius']
        
        self.current_params['radius'] = max(1.0, self.current_params['radius'])
    
    def save_image(self, image_array, filename):
        """Save image array to file."""
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 1.0) * 255
            image_array = image_array.astype(np.uint8)
        
        if len(image_array.shape) == 3:
            img = Image.fromarray(image_array)
        else:
            img = Image.fromarray(image_array, mode='L')
        
        img.save(filename)
        abs_path = os.path.abspath(filename)
        print(f"  Saved image: {abs_path}")
    
    def print_comparison(self):
        """Print parameter comparison."""
        x, y, r = self.current_params['x'], self.current_params['y'], self.current_params['radius']
        target_x, target_y, target_r = self.target_center_x, self.target_center_y, self.target_radius
        
        print(f"\n=== PARAMETER COMPARISON ===")
        print(f"Target:  x={target_x:6.1f}, y={target_y:6.1f}, r={target_r:6.1f}")
        print(f"Learned: x={x:6.1f}, y={y:6.1f}, r={r:6.1f}")
        
        x_diff = abs(x - target_x)
        y_diff = abs(y - target_y)
        r_diff = abs(r - target_r)
        
        print(f"Diff:    x={x_diff:6.1f}, y={y_diff:6.1f}, r={r_diff:6.1f}")
        
        x_accuracy = 100 * (1 - x_diff/target_x) if target_x > 0 else 0
        y_accuracy = 100 * (1 - y_diff/target_y) if target_y > 0 else 0
        r_accuracy = 100 * (1 - r_diff/target_r) if target_r > 0 else 0
        
        print(f"X accuracy: {x_accuracy:.1f}%")
        print(f"Y accuracy: {y_accuracy:.1f}%")
        print(f"Radius accuracy: {r_accuracy:.1f}%")
        
        return x_accuracy, y_accuracy, r_accuracy
    
    def create_debug_visualization(self, iteration, rendered_buffer):
        """Create the six-panel debug view for one iteration."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Circle Learning Debug - Iteration {iteration}', fontsize=16)
        
        # Download rendered image
        rendered_data = rendered_buffer.to_numpy().view(np.float32).reshape(self.image_size, self.image_size, 4)
        rendered_rgb = rendered_data[:, :, :3]
        
        # 1. Target image
        axes[0, 0].imshow(self.target_image)
        axes[0, 0].set_title('Target Circle')
        axes[0, 0].axis('off')
        
        # 2. Rendered image
        axes[0, 1].imshow(rendered_rgb)
        axes[0, 1].set_title('Rendered Circle')
        axes[0, 1].axis('off')
        
        # 3. Difference image - proper color-coded visualization using masks
        diff = np.zeros_like(rendered_rgb)

        # These are the optimizer's own numbers, not a re-derivation of them.
        # target_mask is the very buffer the loss integrates against, and the
        # rendered coverage is the alpha renderCirclesToBuffer wrote. Recovering
        # coverage from the composited RGB instead would silently threshold it
        # at alpha = 0.5 and put these panels in a different branch of the loss
        # than the optimizer is actually in.
        target_mask = self.target_mask_buffer.to_numpy().view(np.float32).reshape(
            self.image_size, self.image_size)
        rendered_mask = rendered_data[:, :, 3]
        
        # Red channel: target areas not covered by rendered
        diff[:, :, 0] = target_mask * (1 - rendered_mask)
        # Blue channel: rendered areas not matching target
        diff[:, :, 2] = rendered_mask * (1 - target_mask)
        
        axes[0, 2].imshow(diff)
        axes[0, 2].set_title('Difference (Red=Target, Blue=Rendered)')
        axes[0, 2].axis('off')
        
        # 4. Target mask
        axes[1, 0].imshow(target_mask, cmap='gray', vmin=0.0, vmax=1.0)
        axes[1, 0].set_title('Target Mask') 
        axes[1, 0].axis('off')
        
        # 5. Rendered mask
        axes[1, 1].imshow(rendered_mask, cmap='gray', vmin=0.0, vmax=1.0)
        axes[1, 1].set_title('Rendered Mask')
        axes[1, 1].axis('off')
        
        # 6. Intersection mask
        intersection = target_mask * rendered_mask
        intersection_pixels = np.sum(intersection)
        max_intersection_value = np.max(intersection)
        
        # Use consistent vmin/vmax to avoid auto-scaling misleading visualization
        axes[1, 2].imshow(intersection, cmap='gray', vmin=0.0, vmax=1.0)
        if intersection_pixels < 1.0:
            axes[1, 2].set_title(f'Intersection Mask (NO OVERLAP! sum={intersection_pixels:.2f})')
        else:
            axes[1, 2].set_title(f'Intersection Mask ({intersection_pixels:.0f} px, max={max_intersection_value:.3f})')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        debug_path = os.path.join(self.output_dir, f"debug_iteration_{iteration:04d}.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return debug_path
    
    def plot_training_analysis(self, losses):
        """Create training analysis plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title("Learning Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.grid(True)
        loss_curve_path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        abs_loss_path = os.path.abspath(loss_curve_path)
        print(f"Saved loss curve: {abs_loss_path}")
    
    def train(self, num_iterations=4000, learning_rate=0.5, momentum=0.9):
        print(f"Training for {num_iterations} iterations")
        losses = []

        for iteration in range(num_iterations):
            # Compute gradients using Slang autodiff
            result = self.compute_autodiff_gradients(self.current_params)
            losses.append(result['loss'])
            
            # Report before updating, so the loss on the progress line, the
            # debug image and the parameter comparison all describe the same
            # circle - the one the gradient was just computed at.
            if iteration % 20 == 0 or iteration == num_iterations - 1:
                print(f"\nIteration {iteration:3d}: Loss = {result['loss']:.6f}")
                print(f"  Gradients: dx={result['dx']:.6f}, dy={result['dy']:.6f}, dr={result['dr']:.6f}")            

                # Create debug visualization
                rendered_buffer = self.render_circle(self.current_params)
                print(f"  Creating debug visualization...")
                debug_path = self.create_debug_visualization(iteration, rendered_buffer)
                abs_debug_path = os.path.abspath(debug_path)
                print(f"  Saved debug visualization: {abs_debug_path}")

                x_acc, y_acc, r_acc = self.print_comparison()

                # Check for early stopping.
                if x_acc >= 98 and y_acc >= 98 and r_acc >= 98:
                    print(f"\n EARLY STOPPING: All parameters achieved 98%+ accuracy! Iteration {iteration}")
                    print(f"   Stopped at iteration {iteration} (saved {num_iterations - iteration - 1} iterations)")
                    break

            # Update parameters
            self.update_parameters(result, learning_rate, momentum)

        
        # Save final results
        final_rendered = self.render_circle(self.current_params)
        final_data = final_rendered.to_numpy().view(np.float32).reshape(self.image_size, self.image_size, 4)
        final_path = os.path.join(self.output_dir, "final_result.png")
        print(f"\nSaving final result...")
        self.save_image(final_data[:, :, :3], final_path)
        
        # Plot training analysis
        self.plot_training_analysis(losses)
        
        print(f"\n" + "="*60)
        print(f"Training completed!")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Check '{self.output_dir}' directory for results.")
        print("="*60)
        
        return losses

def main():
    print("Matching target image with Slang (parallel gradients)")
    print("="*60)
    learner = MatchTargetImage(image_size=128)  
    losses = learner.train(num_iterations=4000, learning_rate=0.5, momentum=0.9)
    
    print(f"\nTraining completed with {len(losses)} iterations")

if __name__ == "__main__":
    main()
