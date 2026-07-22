import cv2
import numpy as np

# Load images (ensure they are the same size and rectified)
imgL = cv2.imread('im2.png', 0)
imgR = cv2.imread('im6.png', 0)

def nothing(x):
    pass

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

    # 2. Update StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=n_disp, blockSize=b_size)
    stereo.setUniquenessRatio(uniqueness)
    stereo.setSpeckleWindowSize(speckle)

    # 3. Compute Disparity
    disparity = stereo.compute(imgL, imgR)

    # 4. Normalize for visualization (0 to 255)
    disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, 
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply a colormap to make it easier to read
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_AUTUMN)

    # 5. Display the result
    cv2.imshow('Disparity Map', disp_color)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()