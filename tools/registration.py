import os
import open3d as o3d
import copy


#largely stolen from these tutorials
#https://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html
#https://www.open3d.org/docs/0.7.0/tutorial/Advanced/global_registration.html#global-registration
#see my notes in registrationLearning2.py for more details on the functions
#to apply a transformation use pointCloud.transform(transMat)

#NOTE
#there seems to be something non-deterministic about this process, set seed when using




#plot both point clouds, slight moditication of draw_registration_result from tutorial
#takes two point clouds, source and target
def monochromePlot(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])

#preprocessing
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


#registration of source onto target
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

# #applying the transformation to the file as a mesh and then exporting
# #i will also load in the target as a mesh just to show that the alignment
# #process works
# u058_12Mesh = o3d.io.read_triangle_mesh("pat058u_12.ply")
# u058_01Mesh = o3d.io.read_triangle_mesh("pat058u_01.ply")
# o3d.visualization.draw_geometries([u058_12Mesh, u058_01Mesh])
# monochromePlot(u058_12Mesh, u058_01Mesh)
# #now we apply the transformation that was calculated for the point cloud
# u058_12MeshTrans = u058_12Mesh.transform(reg01_12.transformation)
# o3d.visualization.draw_geometries([u058_12MeshTrans, u058_01Mesh])
# monochromePlot(u058_12MeshTrans, u058_01Mesh)
# #then it can be exported
# os.chdir("K:/testDir")
# o3d.io.write_triangle_mesh("regiOutTest.ply", u058_12MeshTrans, write_ascii = True)

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
    