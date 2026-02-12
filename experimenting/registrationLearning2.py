

#following this basic tutorial but with my data, unless that fails then I will 
#try thier data first
#https://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html

import os
import open3d as o3d
import numpy as np
import copy


#bringing in my data
os.chdir("K:/IOSSegData/clean/testCleanU")
u001 = o3d.io.read_point_cloud("001_U.ply")
u002 = o3d.io.read_point_cloud("002_U.ply")

#basic visualization of the point clouds and how they are currently in relation 
#to each other
#color the two clouds differently from each other
u001temp = copy.deepcopy(u001)
u002temp = copy.deepcopy(u002)
u001temp.paint_uniform_color([1, 0.706, 0])
u002temp.paint_uniform_color([0, 0.651, 0.929])
#visualize them each separately, then together
o3d.visualization.draw(u001temp)
o3d.visualization.draw(u002temp)
o3d.visualization.draw_geometries([u001temp, u002temp])
#zoom out so that the points start to look like a surface, our data not as dense
#as the examples. the seem generally well aligned to begin with, we will run registration
#on them an see if it improves, though it will be hard to tell as these are actually
#two different scans
#why not just use an individual from RME?



#a helper function mapped out in the tutorial
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    

#before using ICP alignment, we must have a rough idea of the transformation needed
#for the alignment to use as an initialization. if two things are generally aligned
#already then you can use an identity transformation (np.eye(4))
#if not, we use a global registration tool to get an initial transfomration
#following this tutorial
#https://www.open3d.org/docs/0.7.0/tutorial/Advanced/global_registration.html#global-registration



#preprocessing function for the point cloud
#arguements, pcd = point cloud data, voxel size = size of rasterization (3d cubification)
#
#this downsamples the point cloud (not sure if this is something I want to keep)
#looks like it could just be taken out of there
#
#estimates normals, this finds a normal vector for each point by creating a surface
#using the points around it, we only read the data in as a point cloud so it cant 
#create normal vectors using the faces of the surface, this is something that could 
#investigated later, see o3d.io.read_triangle_mesh("file.ply")
#
#computes FPFH (fast point feature histogram) feature for each point, used later in the prcess
#The FPFH feature is a 33-dimensional vector that describes the local geometric
# property of a point. A nearest neighbor query in the 33-dimensinal space can 
#return points with similar local geometric structures.
#this pretty much just describes local curvature
#
#one thing that will need experimenting here is the voxel size as it is application
#dependent, things may be on different scales. is any voxelization needed? are we
#losing information? can whole process be done better by utilizing the face information
#and not just the point clouds?
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    o3d.geometry.estimate_normals(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


#a function to read in the two clouds, run the preprocessing, define an initial
#transformation (identity transfomation), and visualize the point clouds as they
#currently are. not in love with this function, but produces the necessary outputs,
#can play with it later to make it more to my liking, using for now
def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh











