import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimesh
import trimeshExtractFaceLabels as tefl
import numpy as np
import trimeshToDf_labels as ttdl
import trimeshToDfNoLabels as ttdnl
import dfToPlyExport as dtpe

#aligning to master arch
#some of these test scans already have random rotations applied to them
#will probably want to try a number of random rotations on each scan



# #center and scale the test arches
# #teeth3ds scans
# #patients
# pats3ds = {
#     "badNoBase": "00OMSZGW_U",
#     "badBase": "0IU0UV8E_U",
#     "goodNoBase": "0LF355FQ_U",
#     "goodBase": "0EAKT1CU_U",
#     }
# #directories
# fullT3dsDir = "K:/teeth3DS/scanData/upperPly/"
# remeshT3dsDir = "K:/teeth3DS/scanData/upperPlyRemesh/"
# fullT3dsPaths = {name: fullT3dsDir + i + ".ply" for name, i in pats3ds.items()}
# remeshT3dsPaths = {name: remeshT3dsDir + i + "_remesh.ply" for name, i in pats3ds.items()}

# #iosseg scan, bad no base
# iosPath = "K:/IOSSegData/clean/allCleanU/059_U.ply"

# #iowa expansion scan
# iowaExpFullPath = "K:/iowaExpansion/fullRugaeAnnotScans/pre/pat001Pre_annot.ply"
# iowaExpRemeshPath = "K:/iowaExpansion/segReadyScans/pre/pat001Pre_segReady.ply"

# #iowaRme
# iowaRmeFullPath = "K:/iowaRme/preDelivAndFinalScans/finalScanU/fullScans/pat001u_fin.ply"
# iowaRmeRemeshPath = "K:/iowaRme/preDelivAndFinalScans/finalScanU/segReadyScans/pat001u_fin_segReady.ply"

# #all paths
# allPaths = [
#     *fullT3dsPaths.values(),
#     *remeshT3dsPaths.values(),
#     iosPath,
#     iowaExpFullPath,
#     iowaExpRemeshPath,
#     iowaRmeFullPath,
#     iowaRmeRemeshPath
#     ]


# #center and scale each of the test scans and export them
# #helper functions
# def centerScaleLabs(inPath, name, outDir):
#     mesh = trimesh.load(inPath, process = False)
#     #extract face color information from trimesh
#     colorDf = tefl.trimeshExtractFaceLabels(mesh)
#     #center mesh
#     mesh.apply_translation(-mesh.centroid)
#     #obtain scaling factor
#     scaleFac = 1/np.max(mesh.extents)
#     #scale mesh
#     mesh.apply_scale(scaleFac)
#     #export
#     vertDf, faceDf = ttdl.trimeshToDf_labels(mesh, colorDf = colorDf)
#     outPath = outDir + name + ".ply"
#     dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = outPath)
# def centerScaleNoLabs(inPath, name, outDir):
#     mesh = trimesh.load(inPath, process = False)
#     #center mesh
#     mesh.apply_translation(-mesh.centroid)
#     #obtain scaling factor
#     scaleFac = 1/np.max(mesh.extents)
#     #scale mesh
#     mesh.apply_scale(scaleFac)
#     #export
#     vertDf, faceDf = ttdnl.trimeshToDfNoLabels(mesh)
#     outPath = outDir + name + ".ply"
#     dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = outPath)

# #names for output
# names = [
#     *["t3dsFull_" + i for i in pats3ds.keys()],
#     *["t3dsRemesh_" + i for i in pats3ds.keys()],
#     "ios_badNoBase",
#     "iowaExpFull",
#     "iowaExpRemesh",
#     "iowaRmeFull",
#     "iowaRmeRemesh"
#     ]

# #loop thru all files
# for i in range(len(allPaths)):
#     if names[i] in [
#             "iowaExpFull",
#             "iowaExpRemesh",
#             "iowaRmeFull",
#             "iowaRmeRemesh"
#             ]:
#         centerScaleNoLabs(
#             inPath = allPaths[i],
#             name = names[i],
#             outDir = "K:/testDir/masterAlignTest/"
#             )
#     else:
#         centerScaleLabs(
#             inPath = allPaths[i],
#             name = names[i],
#             outDir = "K:/testDir/masterAlignTest/"
#             )
#     print("center, scale, and export complete: " + names[i])

###############################################################################
#alignement scheme
import open3d as o3d
import  preprocess_point_cloud as ppc
import monochromePlot as mcp


#master arch
m1Path = "K:/masterArches/masterArch1/mA1Full.ply"
m1Pc = o3d.io.read_point_cloud(m1Path)

#test arches
testDir = "K:/testDir/masterAlignTest/"
import os
testFiles = os.listdir(testDir)
testFiles = [i for i in testFiles if i.endswith(".ply")]
#random rotation
#testPc.transform(randRots[i])

# #orient to master and export
# for i in range(len(testFiles)):
#     filei = testFiles[i]
#     print(filei)
#     testPc = o3d.io.read_point_cloud(testDir + testFiles[i])
#     #mcp.monochromePlot(source = testPc, target = m1Pc)
#     #set up
#     voxel_size = .1
#     iters = 30
#     #source is what will be moving
#     #target is what we are wanting to match to
#     source_down, source_fpfh = ppc.preprocess_point_cloud(testPc, voxel_size)
#     target_down, target_fpfh = ppc.preprocess_point_cloud(m1Pc, voxel_size)
#     distance_threshold = voxel_size * 1.5
#     ransacRes = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source  = source_down,
#         target = target_down,
#         source_feature = source_fpfh,
#         target_feature = target_fpfh, 
#         mutual_filter = False, 
#         max_correspondence_distance = distance_threshold,
#         estimation_method  = o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         ransac_n = 4, #value from tutorial, could be changed
#         checkers = [
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), #value from tutorial, could be changed
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold)
#         ], 
#         criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)) #value from tutorial, could be changed
#     #extract transformation matrix
#     ransacTrans = ransacRes.transformation
#     #transform source to align with target
#     testPcTrans = testPc.transform(ransacTrans)
#     #plot registered point clouds
#     mcp.monochromePlot(testPcTrans, m1Pc)
    
#     #load in as trimesh, rotate, and export
#     #ignoring labels for now as this is not important to process
#     meshi = trimesh.load(testDir + testFiles[i], process = False)
#     meshi.apply_transform(ransacTrans)
#     vertDf, faceDf = ttdnl.trimeshToDfNoLabels(meshi)
#     dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = "K:/testDir/masterAlignTest/aligned/" + testFiles[i])


    



#now that they are all aligned, lets bring them in and apply specific rotations 
#that may be challenging so that we can see if this process will work for rotation
import originRotMatrix as orm

alignDir = "K:/testDir/masterAlignTest/aligned/"
alignFiles = os.listdir(alignDir)
alignFiles = [i for i in alignFiles if i.endswith(".ply")]
#obtain rotation matrix
xRot = orm.originRotMatrix(degrees = 90, axis = "x")
yRot = orm.originRotMatrix(degrees = 75, axis = "y")
zRot = orm.originRotMatrix(degrees = 180, axis = "z")

for i in range(len(testFiles)):
    filei = alignFiles[i]
    print(filei)
    testPc = o3d.io.read_point_cloud(alignDir + filei)
    #apply rotations
    testPc = testPc.transform(xRot)
    testPc = testPc.transform(yRot)
    testPc = testPc.transform(zRot)
    #visualize
    #mcp.monochromePlot(source = testPc, target = m1Pc)
    #set up
    voxel_size = .05
    iters = 100
    #source is what will be moving
    #target is what we are wanting to match to
    source_down, source_fpfh = ppc.preprocess_point_cloud(testPc, voxel_size)
    target_down, target_fpfh = ppc.preprocess_point_cloud(m1Pc, voxel_size)
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
    # #transform source to align with target
    # testPcTrans = testPc.transform(ransacTrans)
    # #plot registered point clouds
    # mcp.monochromePlot(testPcTrans, m1Pc)
    
    #icp
    threshold = voxel_size * 0.4
    icpRes = o3d.pipelines.registration.registration_icp(
            source = testPc,
            target = m1Pc, 
            max_correspondence_distance = threshold,
            init = ransacTrans,
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = iters)) #default is 30
    
    
    testPcTrans = testPc.transform(icpRes.transformation)
    mcp.monochromePlot(testPcTrans, m1Pc)
    


