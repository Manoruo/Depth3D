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



def make_pointcloud(rgb, depth, T, fx, fy, cx, cy, max_dist=None):
    
    if type(rgb) == str:
        rgb = cv2.imread(rgb)[:, :, ::-1]
    
    if type(depth) == str:
        depth = np.load(depth).astype(np.float32) / 1000 ## convert mm depth map to meter

    # ----- CREATE MASK -----
    mask = (depth > 0) & np.isfinite(depth)   # boolean mask HxW

    if max_dist:
        mask = mask & (depth <= max_dist)

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


def merge_point_clouds_from_data(pcd1_w, pcd2_w, T_world_cam1=None, T_world_cam2=None,
                                 max_depth=3, voxel_size_icp=0.3, voxel_size_fuse=0.02):
    """
    Merge two point clouds with ICP alignment using provided poses.

    Args:
        pcd1_w (o3d.geometry.PointCloud): First point cloud.
        pcd2_w (o3d.geometry.PointCloud): Second point cloud.
        T_world_cam1 (np.ndarray): 4x4 pose of first camera.
        T_world_cam2 (np.ndarray): 4x4 pose of second camera.
        max_depth (float): Depth threshold for filtering.
        voxel_size_icp (float): Voxel size for ICP downsampling.
        voxel_size_fuse (float): Voxel size for final fused cloud.

    Returns:
        o3d.geometry.PointCloud: Merged and fused point cloud.
    """
    # Filter by depth
    pcd1_w = pcd1_w.select_by_index(
        np.where(np.asarray(pcd1_w.points)[:, 2] < max_depth)[0]
    )
    pcd2_w = pcd2_w.select_by_index(
        np.where(np.asarray(pcd2_w.points)[:, 2] < max_depth)[0]
    )

    # Initial relative transform
    if T_world_cam1 and T_world_cam2:

        T_init = np.linalg.inv(T_world_cam2) @ T_world_cam1
    else:
        T_init = np.eye(4)
    # Downsample for ICP
    pcd1_ds = pcd1_w.voxel_down_sample(voxel_size_icp)
    pcd2_ds = pcd2_w.voxel_down_sample(voxel_size_icp)

    # Noise filtering
    pcd1_ds, _ = pcd1_ds.remove_radius_outlier(nb_points=5, radius=voxel_size_icp*3)
    pcd2_ds, _ = pcd2_ds.remove_radius_outlier(nb_points=5, radius=voxel_size_icp*3)

    # Estimate normals
    pcd1_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size_icp*2, max_nn=30))
    pcd2_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size_icp*2, max_nn=30))

    # Colored ICP
    # result_icp = o3d.pipelines.registration.registration_colored_icp(
    #     source=pcd2_ds,
    #     target=pcd1_ds,
    #     max_correspondence_distance=3,
    #     init=T_init
    # )
    result_icp = o3d.pipelines.registration.registration_icp(
    pcd2_ds, pcd1_ds, 1, T_init,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())

    print(f"Colored ICP fitness: {result_icp.fitness:.3f}")
    print(f"Colored ICP RMSE: {result_icp.inlier_rmse:.3f}")

    # Apply transformation to full-res cloud
    pcd2_w.transform(result_icp.transformation)

    # Merge clouds
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined += pcd1_w
    pcd_combined += pcd2_w

    # Final voxel downsample for fusion
    pcd_fused = pcd_combined.voxel_down_sample(voxel_size_fuse)

    return pcd_fused


