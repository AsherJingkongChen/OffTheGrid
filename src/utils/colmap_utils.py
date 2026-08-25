import numpy as np
from scipy.spatial.transform import Rotation as R
import os

def export_to_colmap_txt(output_dir, w2c_extrinsics, intrinsics, points3D, points_colors, image_names):
    """
    Exports reconstruction to COLMAP .txt format.
    w2c_extrinsics: (N, 4, 4) tensor
    intrinsics: (N, 3, 3) tensor
    points3D: (G, 3) tensor (Gaussian means)
    """
    colmap_dir = os.path.join(output_dir, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)

    # 1. Write cameras.txt
    # Assuming PINHOLE model: CAMERA_ID, MODEL, WIDTH, HEIGHT, F_X, F_Y, C_X, C_Y
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        for i in range(len(intrinsics)):
            K = intrinsics[i].cpu().numpy()
            # Simple assumption: W/H from the load_size in your args
            # Using 518 as default from your script
            f.write(f"{i+1} PINHOLE 518 518 {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")

    # 2. Write images.txt
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        for i in range(len(w2c_extrinsics)):
            pose = w2c_extrinsics[i].cpu().numpy()
            rot = pose[:3, :3]
            t = pose[:3, 3]
            
            quat = R.from_matrix(rot).as_quat() # x, y, z, w
            # COLMAP expects w, x, y, z
            f.write(f"{i+1} {quat[3]} {quat[0]} {quat[1]} {quat[2]} {t[0]} {t[1]} {t[2]} {i+1} {os.path.basename(image_names[i])}\n")
            f.write("\n") # Empty line for points2D (not used in feedforward export)

    # 3. Write points3D.txt
    # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK_LIST[IMAGE_ID, POINT2D_IDX]
    # We export Gaussian means as points; colors are set to neutral gray (128)
    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        pts = points3D.reshape(-1, 3).cpu().numpy()
        colors = points_colors.reshape(-1, 3).cpu().numpy()
        for i, pt in enumerate(pts):
            f.write(f"{i+1} {pt[0]} {pt[1]} {pt[2]} {colors[i,0]} {colors[i,1]} {colors[i,2]} 0 1 1\n")