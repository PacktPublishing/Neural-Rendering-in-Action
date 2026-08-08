import cv2
import numpy as np

# Load images (ensure they are the same size and rectified)
imgL = cv2.imread('im2.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('im6.png', cv2.IMREAD_GRAYSCALE)
if imgL is None or imgR is None:
    raise SystemExit('Could not load the stereo pair; check the file paths.')

# Baseline (meters) and focal length (pixels) for the chosen stereo pair.
# The values correspond to the Middlebury 2005 dataset (im2.png / im6.png).
# Replace these with the calibration of your own rig.
BASELINE_M = 0.16
FOCAL_PX = 3740.0

def nothing(x):
    pass

# The StereoBM object is created once and reconfigured in the loop.
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# Create a window for the UI
cv2.namedWindow('Stereo Tuning', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stereo Tuning', 600, 200)

# Create trackbars for parameters
# numDisparities: must be a multiple of 16
cv2.createTrackbar('numDisparities', 'Stereo Tuning', 1, 15, nothing)
# blockSize: must be an odd number >= 5
cv2.createTrackbar('blockSize', 'Stereo Tuning', 5, 50, nothing)
cv2.createTrackbar('uniquenessRatio', 'Stereo Tuning', 15, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'Stereo Tuning', 0, 200, nothing)

while True:
    # 1. Get current positions of trackbars
    # Convert numDisparities to a multiple of 16 (e.g., slider 1 = 16)
    n_disp = cv2.getTrackbarPos('numDisparities', 'Stereo Tuning') * 16
    if n_disp == 0: n_disp = 16

    # Ensure blockSize is odd and at least 5
    b_size = cv2.getTrackbarPos('blockSize', 'Stereo Tuning')
    if b_size < 5: b_size = 5
    if b_size % 2 == 0: b_size += 1

    uniqueness = cv2.getTrackbarPos('uniquenessRatio', 'Stereo Tuning')
    speckle = cv2.getTrackbarPos('speckleWindowSize', 'Stereo Tuning')

    # 2. Update StereoBM object (reconfigure, do not recreate)
    stereo.setNumDisparities(n_disp)
    stereo.setBlockSize(b_size)
    stereo.setUniquenessRatio(uniqueness)
    stereo.setSpeckleWindowSize(speckle)

    # 3. Compute disparity.
    # StereoBM returns CV_16S holding disparity * 16 (4 fractional bits),
    # so divide by 16 to get disparity in pixels.
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # Pixels with no valid match are flagged below the minimum disparity
    # (minDisparity - 1); mask them out so they do not skew the
    # normalisation or the depth conversion.
    valid = disparity > 0

    # 4. Convert disparity to metric depth:  Z = f * B / d
    depth = np.zeros_like(disparity)
    depth[valid] = (FOCAL_PX * BASELINE_M) / disparity[valid]

    # 5. Normalise only the valid pixels for visualisation (0 to 255)
    disp_vis = np.zeros(disparity.shape, dtype=np.uint8)
    if valid.any():
        disp_vis[valid] = cv2.normalize(disparity[valid], None, alpha=0,
                                        beta=255, norm_type=cv2.NORM_MINMAX,
                                        dtype=cv2.CV_8U).ravel()

    # Apply a colormap to make it easier to read
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_AUTUMN)
    disp_color[~valid] = 0

    # 6. Display the result
    cv2.imshow('Disparity Map', disp_color)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
