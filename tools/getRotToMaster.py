import sys
sys.path.append("tools")
import open3d as o3d
import  preprocess_point_cloud as ppc
#function to obtain the rotation matrix that aligns a particular scan to the master arch
#filePath is the file path to the scan you want the registration for
#things must be centered and scaled prior to using this
def getRotToMaster(filePath, masterArchPath):
    
    #load in master arch
    mPath = masterArchPath
    mPc = o3d.io.read_point_cloud(mPath)
    
    #load in arch to rotate (souce)
    sPc = o3d.io.read_point_cloud(filePath)
    
    #set up
    voxel_size = .05
    iters = 100
    
    #source is what will be moving
    #target is what we are wanting to match to
    
    
    #initial alignment transformation
    source_down, source_fpfh = ppc.preprocess_point_cloud(sPc, voxel_size)
    target_down, target_fpfh = ppc.preprocess_point_cloud(mPc, voxel_size)
    distance_threshold = voxel_size * .75
    ransacRes = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source  = source_down,
        target = target_down,
        source_feature = source_fpfh,
        target_feature = target_fpfh, 
        mutual_filter = False, 
        max_correspondence_distance = distance_threshold,
        estimation_method  = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n = 4, #4 is value from tutorial, could be changed
        checkers = [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), #value from tutorial, could be changed
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], 
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)) #value from tutorial, could be changed
    #extract transformation matrix
    ransacTrans = ransacRes.transformation
    
    #icp
    threshold = voxel_size * 0.4
    icpRes = o3d.pipelines.registration.registration_icp(
            source = sPc,
            target = mPc, 
            max_correspondence_distance = threshold,
            init = ransacTrans,
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iters))
    
    #return roation matrix
    return icpRes.transformation

# #example
# import monochromePlot as mcp
# #load in master
# mPc = o3d.io.read_point_cloud("K:/masterArches/masterArch1/mA1Full.ply")
# #load in arch to rotate
# sPc = o3d.io.read_point_cloud('K:/testDir/masterAlignTest/t3dsRemesh_goodNoBase.ply')
# #visualize original positions
# mcp.monochromePlot(source = sPc, target = mPc)
# #get rotation
# aaa = getRotToMaster(filePath = 'K:/testDir/masterAlignTest/t3dsRemesh_goodNoBase.ply')
# #apply rotation
# sPcTrans = sPc.transform(aaa) #i think this transformation is actually done in place
# #visualize new positions
# mcp.monochromePlot(source = sPcTrans, target = mPc)