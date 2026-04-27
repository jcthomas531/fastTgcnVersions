import os
import open3d as o3d
import copy
import trimesh
import numpy as np

#largely stolen from these tutorials
#https://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html
#https://www.open3d.org/docs/0.7.0/tutorial/Advanced/global_registration.html#global-registration
#see my notes in registrationLearning2.py for more details on the functions
#to apply a transformation use pointCloud.transform(transMat)
#open3d will not export face colors, must use trimesh

#NOTE
#there seems to be something non-deterministic about this process, set seed when using



###############################################################################
#MONOCHROME PLOTTING
###############################################################################
#plot both point clouds, slight moditication of draw_registration_result from tutorial
#takes two point clouds, source and target
def monochromePlot(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])


###############################################################################
#PREPROCESSING
###############################################################################
#voxelization (down sampline), estimation of normal vectors, and geometric 
#feature extration
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

###############################################################################
#REGISTER POINT CLOUDS AND OBTAIN TRANSFORMATION
###############################################################################
#registration of source onto target
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
    source_down, source_fpfh = preprocess_point_cloud(sourceC, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(targetC, voxel_size)
    
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


###############################################################################
#APPLY TRANSFORMATION AND EXPORT
###############################################################################
#the above transformation is done on the point cloud, we want to apply it to the 
#mesh. open3d wont let you export face colors, trimesh will with some massaging
#trimesh is a bit more editable and user friendly than open3d
#there are also registration methods in trimesh that would be good to investigate
#but for the moment I am just going to stick with using the registration I have
#previously constructed and using trimesh to export
#https://trimesh.org/trimesh.registration.html

#inFile, the path to the 
def registerAndExport(inFile, outFile, trans):
    
    #read in mesh
    mesh = trimesh.load(inFile)
    
    #apply transformation
    meshTrans = mesh.apply_transform(trans)
    
    #attach face colors in a way that trimesh will export
    colors = meshTrans.visual.face_colors.astype(np.uint8)
    meshTrans.face_attributes["red"] = colors[:, 0]
    meshTrans.face_attributes["green"] = colors[:, 1]
    meshTrans.face_attributes["blue"] = colors[:, 2]
    meshTrans.face_attributes["alpha"] = colors[:, 3]
    
    #export mesh
    meshTrans.export(file_obj = outFile, file_type = "ply", encoding = "ascii")
    
    return True


#EXAMPLE WORKFLOW
#the rme scans still need a bit of massaging into the correct format before they
#can be used in this process. for now we will register a teeth3ds scan to an iosseg
#scan (the will not line up exactly). Iosseg is the target, teeth3ds is the source
# #obtaining transformation
# iossegPath = "K:/IOSSegData/clean/trainCleanU/007_U.ply"
# teeth3dsPath = "K:/teeth3DS/scanData/upperPlyDecim016/00OMSZGW_UDecim016.ply"
# targetCloud = o3d.io.read_point_cloud(iossegPath)
# sourceCloud = o3d.io.read_point_cloud(teeth3dsPath)
# reg = getRegistration(source = sourceCloud, target=targetCloud)
# #register the meshes and export
# os.chdir("K:/testDir/")
# registerAndExport(inFile = teeth3dsPath, outFile = "registerTest.ply", trans = reg.transformation)
# #check registration via o3d meshes and monochrome plot
# mesh1 = o3d.io.read_triangle_mesh("registerTest.ply")
# mesh2 = o3d.io.read_triangle_mesh(iossegPath)
# monochromePlot(mesh1, mesh2)
# #check coloring
# import sys
# sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
# import plyFunctions as pf
# pf.readAndPlot("registerTest.ply", "U")
# pf.readAndPlot(iossegPath, "U")


###############################################################################
#A FUNCTION TO RUN THE WHOLE REGISTRATION PROCESS
###############################################################################

#source: what is being transformed
#target: the base for the registration
#REMEMBER TO SET A SEED, THIS PROCESS DOES NOT APPEAR DETERMINISTIC
#
#REALLY IMPORTANT THING TO NOTE: REGISTERED SCAN HAS 6 LESS VERTICES THAN NON_REGISTERED
#WHERE ARE THEY GOING
#

#targetFile: path the target mesh
#sourceFile: path to the source mesh
#registerFile: where the registered source file should be saved to
def fullRegistFlow(targetFile, sourceFile, registerFile):
    #bring in the meshs as point clouds
    targetCloud = o3d.io.read_point_cloud(targetFile)
    sourceCloud = o3d.io.read_point_cloud(sourceFile)
    #obtrain the registration transformation
    reg = getRegistration(source = sourceCloud, target=targetCloud)
    #register as meshes and export
    registerAndExport(inFile = sourceFile, outFile = registerFile, trans = reg.transformation)




















