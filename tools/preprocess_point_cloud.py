import open3d as o3d
import copy
import trimesh
import numpy as np
import random

#largely stolen from these tutorials
#https://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html
#https://www.open3d.org/docs/0.7.0/tutorial/Advanced/global_registration.html#global-registration
#see my notes in registrationLearning2.py for more details on the functions
#to apply a transformation use pointCloud.transform(transMat)
#open3d will not export face colors, must use trimesh

#NOTE
#there seems to be something non-deterministic about this process, set seed when using!
#transformations are done in place, make sure you operate with copies!

#setting a seed must be done within each library
#additionally, the threading is incorperated with randomness
#put the following at the top of the script
# import os
# os.environ["OMP_NUM_THREADS"] = "1"

# seed = 826
# random.seed(seed)
# np.random.seed(seed)
# o3d.utility.random.seed(seed)


###############################################################################
#PREPROCESSING
###############################################################################
#voxelization (down sampline), estimation of normal vectors, and geometric 
#feature extration
#pcd is an o3d point cloud
#voxel_size is the granularity, have been using size of 2
#returns the down sampled point cloud and the extracted features
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh