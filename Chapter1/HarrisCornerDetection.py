import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter


def compute_gradients(image):
    """
    Calculate how quickly brightness changes in x and y directions.
    
    Think of this like finding the slope of a hill - we want to know
    how steep the image is at each point, both horizontally and vertically.
    
    Parameters:
    -----------
    image : 2D array
        Grayscale image (values from 0-255 or 0-1)
    
    Returns:
    --------
    Ix : 2D array
        Gradient in x-direction (horizontal change)
    Iy : 2D array
        Gradient in y-direction (vertical change)
    """
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    Ix = signal.convolve2d(image, sobel_x, mode='same', boundary='symm')
    Iy = signal.convolve2d(image, sobel_y, mode='same', boundary='symm')
    
    return Ix, Iy


def create_gaussian_kernel(size, sigma):
    """
    Create a Gaussian smoothing kernel.
    
    This is like a "blur filter" that gives more weight to nearby pixels
    and less weight to far away pixels.
    
    Parameters:
    -----------
    size : int
        Size of the kernel (e.g., 5 means 5x5 kernel)
    sigma : float
        Standard deviation - controls how much smoothing
    
    Returns:
    --------
    kernel : 2D array
        Normalized Gaussian kernel
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel = kernel / np.sum(kernel)
    
    return kernel


def compute_structure_tensor(Ix, Iy, window_size=5, sigma=1.0):
    """
    Build the structure tensor (second moment matrix) M for each pixel.
    
    This summarizes the gradient information in a neighborhood around
    each pixel. The structure tensor tells us about the local image structure.
    
    Parameters:
    -----------
    Ix, Iy : 2D arrays
        Image gradients in x and y directions
    window_size : int
        Size of Gaussian window for smoothing
    sigma : float
        Standard deviation for Gaussian smoothing
    
    Returns:
    --------
    Ixx, Iyy, Ixy : 2D arrays
        Components of structure tensor M = [[Ixx, Ixy], [Ixy, Iyy]]
    """
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    gaussian = create_gaussian_kernel(window_size, sigma)
    
    Ixx = signal.convolve2d(Ixx, gaussian, mode='same', boundary='symm')
    Iyy = signal.convolve2d(Iyy, gaussian, mode='same', boundary='symm')
    Ixy = signal.convolve2d(Ixy, gaussian, mode='same', boundary='symm')
    
    return Ixx, Iyy, Ixy


def compute_harris_response(Ixx, Iyy, Ixy, k=0.04):
    """
    Calculate the Harris corner response for each pixel.
    
    The response R tells us how "corner-like" each point is:
    - Large positive R = corner
    - Negative R = edge
    - Small R = flat region
    
    Parameters:
    -----------
    Ixx, Iyy, Ixy : 2D arrays
        Components of structure tensor
    k : float
        Harris constant (typically 0.04-0.06)
    
    Returns:
    --------
    R : 2D array
        Harris corner response for each pixel
    """
    determinant = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    R = determinant - k * (trace ** 2)
    
    return R


def find_corners(R, threshold_fraction=0.01, neighborhood_size=3):
    """
    Identify corner locations from Harris response map.
    
    We do this in two steps:
    1. Threshold: Keep only strong responses
    2. Non-maximum suppression: Keep only local peaks
    
    Parameters:
    -----------
    R : 2D array
        Harris corner response map
    threshold_fraction : float
        Keep points where R > threshold_fraction * max(R)
    neighborhood_size : int
        Size of neighborhood for non-maximum suppression
    
    Returns:
    --------
    corners : 2D boolean array
        True at corner locations, False elsewhere
    """
    max_response = np.max(R)
    threshold_value = threshold_fraction * max_response
    local_max = maximum_filter(R, size=neighborhood_size)
    corners = (R == local_max) & (R > threshold_value)
    
    return corners


def harris_corner_detector(image, k=0.04, window_size=5, sigma=1.0, 
                           threshold=0.01, neighborhood_size=3):
    """
    Detect corners in an image using the Harris Corner Detector.
    
    This is the main function that runs all steps of the algorithm.
    
    Parameters:
    -----------
    image : 2D array
        Input grayscale image
    k : float
        Harris constant (0.04-0.06 typically)
    window_size : int
        Size of Gaussian window
    sigma : float
        Gaussian standard deviation
    threshold : float
        Corner threshold (fraction of max response)
    neighborhood_size : int
        Size for non-maximum suppression
    
    Returns:
    --------
    corners : 2D boolean array
        Detected corner locations
    R : 2D array
        Harris response map (for visualization)
    """
    Ix, Iy = compute_gradients(image)
    Ixx, Iyy, Ixy = compute_structure_tensor(Ix, Iy, window_size, sigma)
    R = compute_harris_response(Ixx, Iyy, Ixy, k)
    corners = find_corners(R, threshold, neighborhood_size)
    
    return corners, R


def visualize_detection_process(image, Ix, Iy, R, corners):
    """
    Create a comprehensive visualization of the Harris detection process.
    
    Shows each major step so you can see what's happening.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Harris Corner Detection Process', fontsize=16, fontweight='bold')
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('1. Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(Ix, cmap='RdBu')
    axes[0, 1].set_title('2a. Horizontal Gradient (Iₓ)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(Iy, cmap='RdBu')
    axes[0, 2].set_title('2b. Vertical Gradient (Iᵧ)')
    axes[0, 2].axis('off')
    
    gradient_magnitude = np.sqrt(Ix**2 + Iy**2)
    axes[1, 0].imshow(gradient_magnitude, cmap='hot')
    axes[1, 0].set_title('Gradient Strength')
    axes[1, 0].axis('off')
    
    im = axes[1, 1].imshow(R, cmap='jet')
    axes[1, 1].set_title('3. Harris Response')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    corner_overlay = image.copy()
    y_coords, x_coords = np.where(corners)
    axes[1, 2].imshow(corner_overlay, cmap='gray')
    axes[1, 2].scatter(x_coords, y_coords, c='red', s=20, marker='x')
    axes[1, 2].set_title(f'4. Detected Corners ({len(x_coords)} found)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from PIL import Image
    
    print("Harris Corner Detection Demo")
    print("=" * 50)
    
    image_path = input("Enter path to image file (or press Enter for test pattern): ").strip()
    
    if image_path == "":
        print("\nCreating a test checkerboard pattern...")
        test_image = np.zeros((200, 200))
        for i in range(0, 200, 40):
            for j in range(0, 200, 40):
                if (i // 40 + j // 40) % 2 == 0:
                    test_image[i:i+40, j:j+40] = 255
        image = test_image
    else:
        print(f"\nLoading image from: {image_path}")
        img = Image.open(image_path)
        image = np.array(img.convert('L'))
        print(f"Image loaded: {image.shape[0]}x{image.shape[1]} pixels")
    
    print("\nRunning Harris Corner Detection...")
    corners, R = harris_corner_detector(
        image,
        k=0.04,
        window_size=5,
        sigma=1.0,
        threshold=0.01,
        neighborhood_size=3
    )
    
    num_corners = np.sum(corners)
    print(f"\nDetection complete! Found {num_corners} corners")
    
    print("\nComputing gradients for visualization...")
    Ix, Iy = compute_gradients(image)
    
    print("Displaying results...")
    visualize_detection_process(image, Ix, Iy, R, corners)
