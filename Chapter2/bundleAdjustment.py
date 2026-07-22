import numpy as np
import cv2
from scipy.optimize import least_squares

def triangulate_manual(P1, P2, pts1, pts2):
    """
    Manually triangulates 2D points into 3D using SVD.
    """
    points_3d = []
    for i in range(len(pts1)):
        # Construct the design matrix A for the linear system AX = 0
        A = np.array([
            pts1[i,0] * P1[2,:] - P1[0,:],
            pts1[i,1] * P1[2,:] - P1[1,:],
            pts2[i,0] * P2[2,:] - P2[0,:],
            pts2[i,1] * P2[2,:] - P2[1,:]
        ])
        # Solve using Singular Value Decomposition
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1]
        X /= X[3] # Homogeneous to Cartesian
        points_3d.append(X[:3])
    return np.array(points_3d)

def run_reconstruction():
    # 1. LOAD IMAGES
    image_paths = ['im2.png', 'im6.png']  # Add more images here for multiple views, e.g., ['left.png', 'middle.png', 'right.png']
    imgs = [cv2.imread(path, 0) for path in image_paths]
    
    if any(img is None for img in imgs):
        print("Error: Could not load one or more images.")
        return

    print(f"Loaded {len(imgs)} images successfully.")

    # 2. CAMERA SETUP (Assume standard focal length)
    h, w = imgs[0].shape
    K = np.array([[800, 0, w/2], [0, 800, h/2], [0, 0, 1]], dtype=np.float64)
    print(f"Camera intrinsics: focal length 800, principal point ({w/2}, {h/2})")

    # 3. FEATURE MATCHING AND POSE ESTIMATION
    poses = [np.eye(4)]  # first camera at origin
    rvecs = []
    tvecs = []
    pts3d_list = []
    pts2d_list = []
    for i in range(1, len(imgs)):
        print(f"\nProcessing pair: image {i-1} and image {i}")
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(imgs[i-1], None)
        kp2, des2 = sift.detectAndCompute(imgs[i], None)
        
        print(f"Image {i-1}: {len(kp1)} keypoints, descriptors shape: {des1.shape if des1 is not None else 'None'}")
        print(f"Image {i}: {len(kp2)} keypoints, descriptors shape: {des2.shape if des2 is not None else 'None'}")
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        print(f"Total matches found: {len(matches)}")
        
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        print(f"Good matches after ratio test: {len(good)}")
        
        pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float64([kp2[m.trainIdx].pt for m in good])

        # 4. INITIAL POSE ESTIMATION
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
        print(f"Essential matrix:\n{E}")
        
        # Compute fundamental matrix from essential matrix
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
        print(f"Fundamental matrix:\n{F}")
        print(f"Rank of fundamental matrix: {np.linalg.matrix_rank(F)}")
        
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        inliers = np.sum(mask == 255)
        print(f"Inliers after essential matrix estimation: {inliers}")

        # Filter inliers
        pts1_in = pts1[mask.ravel() == 255]
        pts2_in = pts2[mask.ravel() == 255]

        # 5. TRIANGULATION
        P1 = K @ poses[-1][:3, :4]
        P2 = K @ np.hstack((R, t))
        
        points_3d = triangulate_manual(P1, P2, pts1_in, pts2_in)
        
        print(f"Successfully triangulated {len(points_3d)} points between image {i-1} and {i}.")
        rvec, _ = cv2.Rodrigues(R)
        rvecs.append(rvec.flatten())
        tvecs.append(t.flatten())
        pts3d_list.append(points_3d)
        pts2d_list.append((pts1_in, pts2_in))
        pose = np.eye(4)
        pose[:3,:3] = R
        pose[:3,3] = t.flatten()
        poses.append(pose)

    # Check if any points were triangulated
    total_points = sum(len(p) for p in pts3d_list)
    print(f"\nTotal 3D points triangulated across all pairs: {total_points}")
    if total_points == 0:
        print("No 3D points triangulated. Bundle adjustment cannot be performed.")
        return

    # 6. BUNDLE ADJUSTMENT (Optimization)
    print("\nStarting bundle adjustment...")
    def reproj_error(params, pts3d_shapes, pts2d_list, K):
        offset = 0
        rvecs_opt = []
        for _ in range(len(pts3d_shapes)):
            rvecs_opt.append(params[offset:offset+3])
            offset += 3
        tvecs_opt = []
        for _ in range(len(pts3d_shapes)):
            tvecs_opt.append(params[offset:offset+3])
            offset += 3
        pts3d_all = []
        for shape in pts3d_shapes:
            n = shape[0] * 3
            pts3d_all.append(params[offset:offset+n].reshape(shape))
            offset += n
        errors = []
        for i in range(len(pts3d_all)):
            pts3d = pts3d_all[i]
            pts1, pts2 = pts2d_list[i]
            # project to first camera
            if i == 0:
                rvec1 = np.zeros(3)
                tvec1 = np.zeros(3)
            else:
                rvec1 = rvecs_opt[i-1]
                tvec1 = tvecs_opt[i-1]
            p1, _ = cv2.projectPoints(pts3d, rvec1, tvec1, K, None)
            error1 = (p1.reshape(-1, 2) - pts1).ravel()
            # project to second camera
            rvec2 = rvecs_opt[i]
            tvec2 = tvecs_opt[i]
            p2, _ = cv2.projectPoints(pts3d, rvec2, tvec2, K, None)
            error2 = (p2.reshape(-1, 2) - pts2).ravel()
            errors.append(np.concatenate([error1, error2]))
        return np.concatenate(errors)

    pts3d_shapes = [p.shape for p in pts3d_list]
    init_params = np.concatenate(rvecs + tvecs + [p.flatten() for p in pts3d_list])
    print(f"Initial parameters shape: {init_params.shape}")
    res = least_squares(reproj_error, init_params, loss='huber', f_scale=1.0,
                        args=(pts3d_shapes, pts2d_list, K), verbose=1)

    print("Bundle Adjustment Optimized!")
    print(f"Final cost: {res.cost}, number of function evaluations: {res.nfev}")
    
run_reconstruction()