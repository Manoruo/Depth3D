import open3d as o3d
import numpy as np 
import cv2
import matplotlib.pyplot as plt 

def back_project(fx, fy, depth, cx, cy):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Camera frame
    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth

    # Convert to Vicon GCS (Z up) --> Nick was right
    x_v = x_c
    y_v = z_c
    z_v = -y_c

    return np.stack((x_v, y_v, z_v), axis=-1)   # HxWx3



def make_o3d_pointcloud(rgb_path, depth_path, T, fx, fy, cx, cy):
    
    if type(rgb_path) != str:
        rgb = rgb_path
    else:
        rgb = cv2.imread(rgb_path)[:, :, ::-1]
    
    if type(depth_path) != str:
        depth = depth_path
    else:
        depth = np.load(depth_path).astype(np.float32) / 1000 ## convert mm depth map to meter

    # ----- CREATE MASK -----
    mask = (depth > 0) & np.isfinite(depth)   # boolean mask HxW

    # Unproject entire depth → xyz
    p_cam = back_project(fx, fy, depth, cx, cy)   # HxW x 3
    
    # Flatten arrays
    p_cam_flat = p_cam.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    mask_flat = mask.reshape(-1)
    
    # Apply mask
    p_cam_valid = p_cam_flat[mask_flat]
    rgb_valid = rgb_flat[mask_flat] / 255.0

    # Apply transform
    R = T[:3, :3]
    t = T[:3, 3]

    pts_world = []

    for p_cam in p_cam_valid:  # pts_cam can be any shape (N,3)
        p_world = R @ p_cam + t
        pts_world.append(p_world)
    pts_world = np.array(pts_world)
    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(rgb_valid)

    return pcd



def plot_sbs_img_depth(img, depth):
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    for ax, im, title in zip(axs, [img, depth], ['Input', 'Predicted Depth']):
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(title)
    plt.show()


def create_frame(T, size=0.1):
    """
    Create an Open3D coordinate frame at a given pose.

    Args:
        T: 4x4 transformation matrix (camera pose)
        size: length of axis lines
    Returns:
        Open3D geometry (coordinate frame)
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)  # place it in the world
    return frame
