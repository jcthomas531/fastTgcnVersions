import open3d as o3d
import copy
import trimesh
import numpy as np
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import  preprocess_point_cloud as ppc

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
#REGISTER POINT CLOUDS AND OBTAIN TRANSFORMATION
###############################################################################
#registration of source onto target
#both the soure and target should be open3d point cloud objects
#source is what will be moving
#target is what we are wanting to match to
def getRegistration(source, target, method = "point2point", voxel_size = 2, iters = 30):
    #validating arguements
    if not method == "point2point" and not method == "point2plane":
        raise ValueError("method arguement must be either 'point2point' or 'point2plane'")
    
    #make copies of the pointclouds
    sourceC = copy.deepcopy(source)
    targetC = copy.deepcopy(target)
    
    #preprocessing
    source_down, source_fpfh = ppc.preprocess_point_cloud(sourceC, voxel_size)
    target_down, target_fpfh = ppc.preprocess_point_cloud(targetC, voxel_size)
    
    #global registration, used as rough estimate for ICP
    distance_threshold = voxel_size * 1.5 #value from tutorial, could be changed
    print("performing RANSAC registration")
    ransacRes = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source  = source_down,
        target = target_down,
        source_feature = source_fpfh,
        target_feature = target_fpfh, 
        mutual_filter = False, 
        max_correspondence_distance = distance_threshold,
        estimation_method  = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n = 4, #value from tutorial, could be changed
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), #value from tutorial, could be changed
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], 
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)) #value from tutorial, could be changed
    
    #extract transformation matrix
    ransacTrans = ransacRes.transformation
    
    #ICP registration
    threshold = voxel_size * 0.4 #value from tutorial, could be changed
    print("performing ICP registration")
    if method == "point2point":
        icpRes = o3d.pipelines.registration.registration_icp(
                source = sourceC,
                target = targetC, 
                max_correspondence_distance = threshold,
                init = ransacTrans,
                estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iters)) #default is 30
    elif method == "point2plane":
        icpRes = o3d.pipelines.registration.registration_icp(
                source = sourceC,
                target = targetC, 
                max_correspondence_distance = threshold,
                init = ransacTrans,
                estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iters)) #default is 30
    
    
    
    return icpRes



#EXAMPLE WORKFLOW
# #iowaRmeData
# os.chdir("K:/iowaRme/fullScans/pat058")
# #second scan will serve as source, what is being transformed
# u058_12 = o3d.io.read_point_cloud("pat058u_12.ply")
# #first scan will serve as target
# u058_01 = o3d.io.read_point_cloud("pat058u_01.ply")
# #plot how it looks prior to transformation (with actual colors)
# o3d.visualization.draw_geometries([u058_12, u058_01])
# monochromePlot(u058_12, u058_01)
# #calculate registration
# reg01_12 = getRegistration(source = u058_12, target=u058_01)
# #look at stats on the registration performance
# reg01_12
# #look at transformation matri
# reg01_12.transformation
# #transform source to align with target
# u058_12Trans = u058_12.transform(reg01_12.transformation)
# #plot registered point clouds
# monochromePlot(u058_12Trans, u058_01)

#ILLUSTRATION
# #looking at how this is effected by various parameters
# #calculate registration
# reg30 = getRegistration(source = u058_12, target=u058_01, iters = 30)
# print(reg30)
# #calculate registration
# reg50 = getRegistration(source = u058_12, target=u058_01, iters = 50)
# print(reg50)
# #calculate registration
# reg100 = getRegistration(source = u058_12, target=u058_01, iters = 100)
# print(reg100)
# #calculate registration
# reg200 = getRegistration(source = u058_12, target=u058_01, iters = 200)
# print(reg200)    
# #iterations seem to not have a large impac






