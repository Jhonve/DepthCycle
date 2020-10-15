import math
import copy

import numpy as np
import cv2

import open3d as o3d

# intrinsics of camera
k_vfov = 45
k_hfov = 60
k_pixel_width = 640
k_pixel_height = 480
k_depth_scale = 4000

def readLabel(label_path):
    label = cv2.imread(label_path, 2)
    return label

def readDepth(depth_path):
    depth = cv2.imread(depth_path, 2)
    return depth

def readRGB(rgb_path):
    rgb = cv2.imread(rgb_path)
    return rgb

def cameraIntrinsicTransform(vfov=60, hfov=60, pixel_width=320,pixel_height=240):
    camera_intrinsics = np.zeros((3,3))
    camera_intrinsics[2,2] = 1
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(hfov/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(vfov/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

def perspective2Ortho3DPoints(depth_path, vfov=60, hfov=60, pixel_width=320, pixel_height=240, depth_scale=4000):
    intrinsic_mat = cameraIntrinsicTransform(vfov, hfov, pixel_width, pixel_height)

    dense_depth = readDepth(depth_path)
    dense_point = np.zeros((k_pixel_height, k_pixel_width, 3))
    assert(dense_depth.shape[0] == k_pixel_height and dense_depth.shape[1] == k_pixel_width)

    dense_point[:, :, 2] = dense_depth / depth_scale
    dense_point[:, :, 0] = np.arange(1, k_pixel_width + 1)
    temp_arange = np.zeros((k_pixel_width, k_pixel_height, 1))
    temp_arange[:, :, 0] = np.arange(1, k_pixel_height + 1)
    dense_point[:, :, 1] = temp_arange.T

    dense_point[:, :, 0] = (dense_point[:, :, 0] - intrinsic_mat[0][2]) * dense_point[:, :, 2] / intrinsic_mat[0][0]
    dense_point[:, :, 1] = (dense_point[:, :, 1] - intrinsic_mat[1][2]) * dense_point[:, :, 2] / intrinsic_mat[1][1]

    dense_point_vis = np.reshape(dense_point, (k_pixel_height * k_pixel_width, 3))
    point_cloud_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_point_vis))
    o3d.visualization.draw_geometries([point_cloud_vis])

    return dense_point

def getPointCloudAndVis(depth_path, vfov=60, hfov=60, pixel_width=320, pixel_height=240, depth_scale=4000):
    dense_depth = o3d.io.read_image(depth_path)
    
    intrinsic_mat = cameraIntrinsicTransform(vfov, hfov, pixel_width, pixel_height)
    # print(intrinsic_mat)

    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(pixel_width, pixel_height, 
                                float(intrinsic_mat[0][0]), float(intrinsic_mat[1][1]), 
                                float(intrinsic_mat[0][2]), float(intrinsic_mat[1][2]))
    
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(dense_depth, camera_intrinsic, depth_scale=depth_scale)

    inversed_point_cloud = copy.deepcopy(point_cloud)
    inversed_point_cloud = inversed_point_cloud.rotate([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, 0, 0])

    # o3d.visualization.draw_geometries([point_cloud, inversed_point_cloud])
    o3d.visualization.draw_geometries([inversed_point_cloud])

if __name__ == "__main__":
    getPointCloudAndVis("preparation/data/scanNet/depth/000100.png",
                k_vfov, k_hfov, k_pixel_width, k_pixel_height, k_depth_scale)

    perspective2Ortho3DPoints("preparation/data/scanNet/depth/000100.png",
                            k_vfov, k_hfov, k_pixel_width, k_pixel_height, k_depth_scale)


    print("Go Lakers!")