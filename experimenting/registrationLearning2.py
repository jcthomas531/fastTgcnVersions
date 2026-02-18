

#following this basic tutorial but with my data, unless that fails then I will 
#try thier data first
#https://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html

import os
import open3d as o3d
import numpy as np
import copy


#bringing in my data
#IOSSeg data
# os.chdir("K:/IOSSegData/clean/testCleanU")
# u001 = o3d.io.read_point_cloud("001_U.ply")
# u002 = o3d.io.read_point_cloud("002_U.ply")
#iowaRmeData
os.chdir("K:/iowaRme/fullScans/pat058")
u001 = o3d.io.read_point_cloud("pat058u_01.ply")
u002 = o3d.io.read_point_cloud("pat058u_12.ply")



#looking at scale of scans
#voxelization size must be in relation to this scale
print(u001.get_axis_aligned_bounding_box())
print(u001.get_axis_aligned_bounding_box().get_extent())


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



#a plotting function from the tutorial
#this just takes two point clouds and a transformation, applys the transfromation
#to one of the point clouds and then plots the 2 point clouds in different colors
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
###############################################################################
#GLOBAL REGISTRATION
###############################################################################
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
#voxelization is essentially a downsampling, done to save time since this is just
#an initial registration
#
#the version of this function supplied in the tutorial is based on an old syntax
#of open3d. my rewritten verion is how it should be
# def preprocess_point_cloud(pcd, voxel_size):
#     print(":: Downsample with a voxel size %.3f." % voxel_size)
#     pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)

#     radius_normal = voxel_size * 2
#     print(":: Estimate normal with search radius %.3f." % radius_normal)
#     o3d.geometry.estimate_normals(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

#     radius_feature = voxel_size * 5
#     print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
#     pcd_fpfh = o3d.registration.compute_fpfh_feature(
#         pcd_down,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh

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


#a function to read in the two clouds, run the preprocessing, perform an initial 
#transformation to distrub the orientation, and visualize the point clouds 
#when i go to actually apply this to my data, I will remove the disturb transfromation
#not in love with this function, but produces the necessary outputs,
#editing a little bit so it doesnt do a read in each time
#exporting sourceC and targetC just to stay consistent although targetC does not
#change, just sourceC is being distrubed
#
# def prepare_dataset(voxel_size):
#     print(":: Load two point clouds and disturb initial pose.")
#     source = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
#     target = o3d.io.read_point_cloud("../../TestData/ICP/cloud_bin_1.pcd")
#     trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
#                              [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
#     source.transform(trans_init)
#     draw_registration_result(source, target, np.identity(4))
#     source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
#     return source, target, source_down, target_down, source_fpfh, target_fpfh
#
def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    sourceC = copy.deepcopy(source)
    targetC = copy.deepcopy(target)
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    sourceC.transform(trans_init)
    draw_registration_result(sourceC, targetC, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(sourceC, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(targetC, voxel_size)
    return sourceC, targetC, source_down, target_down, source_fpfh, target_fpfh

#usage
sourceP, targetP, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(u001, u002,  2)



#first rough registration, RANSAC registratrion
#I have updated the version from the tutorial for fit the new syntax and changed
#the positional arguments to all be named arguements
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source  = source_down,
        target = target_down,
        source_feature = source_fpfh,
        target_feature = target_fpfh, 
        mutual_filter = False, #new version of open3d requires this arguement
        max_correspondence_distance = distance_threshold,
        estimation_method  = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n = 4,
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], 
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

#usage
result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, 2)
draw_registration_result(source_down, target_down,
                             result_ransac.transformation)



#refined registration after the rough registration
#i think this is unnecessary bc we are only doing this rough alignment to faciliated
#the actual alignment tuturoial
#this function is suboptimal as it uses result_ransac object without having it 
#passed through. It is relying on it being in the global environement
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


result_icp = refine_registration(sourceP, targetP, source_fpfh, target_fpfh,
                                     2)
draw_registration_result(sourceP, targetP, result_icp.transformation)


###############################################################################
#ICP REGISTRATION
###############################################################################
#extending from above

threshold = .2 #i have no idea what this is, 
#documentation for evaluate_registration says: Maximum correspondence points-pair distance.
#see the refine_registration function above, used as about half of the voxelization
transInit = result_ransac.transformation


#evaluate the rough alignment from RANSAC
evalRansac = o3d.pipelines.registration.evaluate_registration(sourceP, targetP,
                                                        threshold, transInit)
evalRansac

#point to point ICP registration
reg_p2p = o3d.pipelines.registration.registration_icp(
        source = sourceP,
        target = targetP, 
        max_correspondence_distance = threshold,
        init = transInit,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30)) #default is 30

reg_p2p
reg_p2p.transformation
draw_registration_result(sourceP, targetP, reg_p2p.transformation)


#point to plane ICP registration
reg_p2l = o3d.pipelines.registration.registration_icp(
        source = sourceP,
        target = targetP, 
        max_correspondence_distance = threshold,
        init = transInit,
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 30)) #default is 30

reg_p2l
reg_p2l.transformation
draw_registration_result(sourceP, targetP, reg_p2l.transformation)
