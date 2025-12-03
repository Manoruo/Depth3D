import rosbag
from sensor_msgs.msg import Image
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import time
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

def find_closest_tf(image_time, tf_list):
    return min(tf_list, key=lambda x: abs(x[0] - image_time))

def quat_to_rot(q):
        R = np.eye(3)

        a = q.w
        b = q.x
        c = q.y
        d = q.z

        a2 = a*a
        b2 = b*b
        c2 = c*c
        d2 = d*d

        ab = a*b
        ac = a*c
        ad = a*d

        bc = b*c
        bd = b*d

        cd = c*d

        R = np.zeros((3, 3))
        R[0, 0] = a2 + b2 - c2 - d2
        R[0, 1] = 2*(bc - ad)
        R[0, 2] = 2*(bd + ac)
        R[1, 0] = 2*(bc + ad)
        R[1, 1] = a2 - b2 + c2 - d2
        R[1, 2] = 2*(cd - ab)
        R[2, 0] = 2*(bd - ac)
        R[2, 1] = 2*(cd + ab)
        R[2, 2] = a2 - b2 - c2 + d2
        return R

def to_homogeneous(xyz, R):
    # Homogeneous matrix
    Hmtrx = np.hstack((R, xyz.reshape(3,1)))
    return np.vstack((Hmtrx, np.array([0,0,0,1])))

def get_pose_and_R(msg):
    if str(type(msg)).find("PoseStamped") != -1:
        pos = msg.pose.position
        r = msg.pose.orientation   
    else:
        pos = msg.transforms[0].transform.translation
        r = msg.transforms[0].transform.rotation
    
    pos = np.array([pos.x, pos.y, pos.z])
    Rot = quat_to_rot(r)
    return pos, Rot

def plot_transforms(transforms, show_path=True):
    """
    Plot a list of transforms in 3D using Open3D.
    
    Each element in transforms should be like:
    t[1].transforms[0].transform.translation / rotation (quaternion)
    """
    geometries = []
    positions = []

    for msg in transforms:
        # Extract translation
        pos, R = get_pose_and_R(msg)
        positions.append(pos)
        # Create coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        frame.rotate(R, center=np.zeros(3))
        frame.translate(pos)
        geometries.append(frame)

    # Optionally add a line to show path
    if show_path and len(positions) > 1:
        positions = np.array(positions)
        lines = [[i, i+1] for i in range(len(positions)-1)]
        colors = [[0, 0, 0] for _ in lines]  # black line
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(positions),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    # Visualize all
    o3d.visualization.draw_geometries(geometries)

def pose_to_tsdf(H):
    """
    Convert a standard 4x4 pose matrix into the TSDF frame used by MonoNav.
    """

    T = np.array([
    [1, 0, 0],
    [0, 0,1],
    [0, 1, 0]
    ]) # correct 

    R = H[:3, :3]
    t = H[:3, 3]

    # Apply frame change
    R_o3d = T @ R @ T.T
    t_o3d = T @ t

    # Build new homogeneous matrix
    H_tsdf = np.eye(4)
    H_tsdf[:3, :3] = R_o3d
    H_tsdf[:3, 3] = t_o3d

    return H_tsdf
# -------------------------------
# PROCESS BAG AND SAVE DATA
# -------------------------------

if __name__ == "__main__":
    
   
    bag_paths = ['lcb_1.bag', 'lcb_2.bag', 'lcb_3.bag', 'night1.bag', 'night2.bag', 'night3.bag'] # put names of your bag files here


    # Process each bag here. You will get output in the data folder

    for bag_path in bag_paths:
        path_name = './data/archive/bags/' + bag_path
        save_dir = 'data/' + bag_path.split('.bag')[0]
        crazyflie_img_dir = os.path.join(save_dir, "crazyflie-rgb-images")
        crazyflie_pose_dir = os.path.join(save_dir, "crazyflie-poses") # this is for extrinsics coming from /cf3/pose
        cf_tfs_dir = os.path.join(save_dir, "crazyflie-tf-poses") # this is for extrinsics coming from /tf topic

        print("Saving files to:", save_dir)

        # -------------------------------
        # LOAD ROS BAG
        # -------------------------------
        bridge = CvBridge()
        images = []
        tfs = []
        cf3_poses = []

        with rosbag.Bag(path_name) as bag:
            for topic, msg, t in bag.read_messages(topics=['/cf3/camera/image_raw', '/tf', '/cf3/pose']):
                if topic == '/cf3/camera/image_raw':
                    images.append((t.to_sec(), msg))
                elif topic == '/tf':
                    tfs.append((t.to_sec(), msg))
                else:
                    cf3_poses.append((t.to_sec(), msg))
        
        #plot_transforms([t[1] for t in tfs])

        
        os.makedirs(crazyflie_img_dir, exist_ok=True)
        os.makedirs(crazyflie_pose_dir, exist_ok=True)
        os.makedirs(cf_tfs_dir, exist_ok=True)

        # Align the data by using the time stamps saved for each pose / image
        for frame_number, (image_time, image_msg) in enumerate(images):
            # Convert ROS Image to OpenCV
            cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Find closest message in cf3/pose 
            closest_tf_time, closest_tf_msg = find_closest_tf(image_time, cf3_poses)
            tim_diff1 = abs(closest_tf_time - image_time)
            pos, R = get_pose_and_R(closest_tf_msg)
            H1 = to_homogeneous(pos, R)

            ## Now do the same thing over again for the /tf message

            # Find closest message in /tf
            closest_tf_time, closest_tf_msg = find_closest_tf(image_time, tfs)     
            tim_diff2 = abs(closest_tf_time - image_time)
            pos, R = get_pose_and_R(closest_tf_msg)
            H2 = to_homogeneous(pos, R)
         


            # Save everything if there is alignment (ensure the estimated pose for image is reliable by looking at time stamp)
            img_path = os.path.join(crazyflie_img_dir, f"crazyflie_frame-{frame_number:06d}.rgb.jpg")
            pose_path = os.path.join(crazyflie_pose_dir, f"crazyflie_frame-{frame_number:06d}.pose.txt")
            tf_path = os.path.join(cf_tfs_dir, f"crazyflie_frame-{frame_number:06d}.pose.txt")
            
            if tim_diff1 < 1 or tim_diff2 < 1:
                np.savetxt(pose_path, H1)
                cv2.imwrite(img_path, cv_image)
                np.savetxt(tf_path, H2)
            else:
                print("Skipping image:", img_path)
                continue
            
            # Display image
            cv2.imshow('crazyflie', cv_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Done saving images and aligned poses from bag.")
