# Harris Corner Detection

A beginner-friendly implementation of the Harris Corner Detection algorithm with step-by-step visualization.

## Features

- Clean, well-documented implementation of Harris Corner Detection
- Step-by-step visualization of the detection process
- Support for any grayscale image format (JPG, PNG, BMP, etc.)
- Built-in test pattern for quick testing

## Requirements

- Python 3.7 or higher
- See `requirements.txt` for package dependencies

Last verified with Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, Matplotlib 3.11.1, and
Pillow 12.3.0.

## Installation

### Windows

#### 1. Create a Virtual Environment

Open PowerShell or Command Prompt and navigate to the project directory:

```bash
cd path\to\Neural-Rendering-in-Action\Chapter1
```

Create a virtual environment:

```bash
python -m venv chap1env
```

#### 2. Activate the Virtual Environment

**PowerShell:**
```powershell
.\chap1env\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Command Prompt:**
```cmd
chap1env\Scripts\activate.bat
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### macOS / Linux

#### 1. Create a Virtual Environment

```bash
cd path/to/Neural-Rendering-in-Action/Chapter1
python3 -m venv venv
```

#### 2. Activate the Virtual Environment

```bash
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the Script

With the virtual environment activated:

```bash
python HarrisCornerDetection.py
```

### Using Your Own Image

When prompted, enter the full path to your image:

```
Enter path to image file (or press Enter for test pattern): C:\path\to\your\image.jpg
```

**Examples:**
- Windows: `C:\Users\YourName\Pictures\photo.jpg`
- macOS/Linux: `/Users/YourName/Pictures/photo.jpg`

### Using the Test Pattern

Simply press Enter when prompted to generate a checkerboard test pattern:

```
Enter path to image file (or press Enter for test pattern): [Press Enter]
```

## Output

The script will display a 6-panel visualization showing:

1. **Original Image** - Your input image
2. **Horizontal Gradient (Iₓ)** - Changes in brightness horizontally
3. **Vertical Gradient (Iᵧ)** - Changes in brightness vertically
4. **Gradient Strength** - Combined magnitude of gradients
5. **Harris Response** - Corner response map (higher values = stronger corners)
6. **Detected Corners** - Final detected corners marked with red X's

## Using as a Library

You can also import and use the functions in your own Python scripts:

```python
import numpy as np
from PIL import Image
from HarrisCornerDetection import harris_corner_detector, visualize_detection_process, compute_gradients

# Load your image
img = Image.open('path/to/image.jpg')
image = np.array(img.convert('L'))  # Convert to grayscale

# Detect corners
corners, R = harris_corner_detector(
    image,
    k=0.04,              # Harris constant (0.04-0.06 typically)
    window_size=5,       # Gaussian window size
    sigma=1.0,           # Gaussian standard deviation
    threshold=0.01,      # Corner threshold (1% of max response)
    neighborhood_size=3  # Non-maximum suppression window
)

# Get corner coordinates
y_coords, x_coords = np.where(corners)
print(f"Found {len(x_coords)} corners")

# Visualize results
Ix, Iy = compute_gradients(image)
visualize_detection_process(image, Ix, Iy, R, corners)
```

## Parameter Tuning

If you're not getting good results, try adjusting these parameters:

- **`k`** (0.04-0.06): Controls sensitivity to corners vs edges
  - Lower values → more corners detected
  - Higher values → fewer, stronger corners

- **`threshold`** (0.001-0.1): Minimum corner strength
  - Lower values → more corners detected
  - Higher values → only strongest corners

- **`neighborhood_size`** (3-7): Non-maximum suppression window
  - Smaller values → corners can be closer together
  - Larger values → corners must be more separated

- **`sigma`** (0.5-2.0): Gaussian smoothing strength
  - Lower values → more sensitive to noise
  - Higher values → smoother, less sensitive

## Algorithm Overview

The Harris Corner Detector works in 4 main steps:

1. **Compute Gradients** - Calculate how brightness changes in x and y directions using Sobel operators
2. **Build Structure Tensor** - Smooth and combine gradient information in local neighborhoods
3. **Compute Harris Response** - Calculate corner response: R = det(M) - k·trace(M)²
4. **Find Corners** - Apply threshold and non-maximum suppression to locate corners

## Deactivating the Virtual Environment

When you're done, deactivate the virtual environment:

```bash
deactivate
```

## Troubleshooting

### Import Errors

If you get import errors, make sure:
1. The virtual environment is activated (you should see `(venv)` in your prompt)
2. All packages are installed: `pip install -r requirements.txt`
3. You're using Python 3.7+: `python --version`

### No Display Window

If matplotlib doesn't show the window:
- On Windows: Make sure you have a GUI environment
- On Linux: You may need to install `python3-tk`
- On macOS: Make sure you're not using a system Python

### Image Not Found

Make sure to use the full absolute path to your image file, and use proper path separators:
- Windows: Use backslashes `\` or forward slashes `/`
- macOS/Linux: Use forward slashes `/`

## License

For educational purposes.

## Author

Created for Neural Rendering in Action - Chapter 1

