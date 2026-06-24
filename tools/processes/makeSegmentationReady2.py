import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import pyvista as pv
import pyacvd  
import trimesh
import open3d as o3d
import numpy as np
import getRegistration as gr
import random
import copy
import trimeshToDfNoLabels as tdnl
import dfToPlyExport as dpe

#seed setting
import os
os.environ["OMP_NUM_THREADS"] = "1"
seed = 826
random.seed(seed)
np.random.seed(seed)
o3d.utility.random.seed(seed)

#pull variables from snakemake
inFile = sys.argv[1]
outFile = sys.argv[2]


#TEMPORARY VERSION OF makeSegmentationReady TO FACILITATE PREDICTION FOR THE
#MODEL TRAINED ON CENTERED, SCALED, AND RANDOMLY ROTATED DATA
#IF THIS MODEL IS SUCCESSFUL, THIS WILL TAKE OVER FOR makeSegmentationReady
#AND REVIEVE A BETTER NAME



#read in file
meshOrig = pv.read(inFile)

#remesh
#this decimates and gives an isotropic remesh
#this will give approx 8500*2 faces, not excatly sure how it work but the subdivide value doesnt matter for resulting faces
meshIso = meshOrig.acvd.remesh(8500, subdivide=3)
print("decimated and remeshed")

#make into trimesh
meshIsoTri = pv.to_trimesh(meshIso)

#REMOVE ORIENTATION
# #make into point cloud for registration
# meshIsoPcd = o3d.geometry.PointCloud()
# meshIsoPcd.points = o3d.utility.Vector3dVector(np.asarray(meshIsoTri.vertices))

# #orientation
# #get registration to arbitrary target scan from teeth3ds
# #registration could also be done on the full iowaExpansion scan however i assume
# #that will take longer and this is just a rough orientation anyway
# teeth3dsTarget = "K:/trainTestSets/teeth3dsDecim016/train/00OMSZGW_UDecim016.ply"
# targetCloud = o3d.io.read_point_cloud(teeth3dsTarget)
# regi = gr.getRegistration(source = meshIsoPcd, target = targetCloud)

# #apply transformation
# meshTrans = copy.deepcopy(meshIsoTri)
# meshTrans.apply_transform(regi.transformation)
# print("oriented")

#center mesh
meshIsoTri.apply_translation(-meshIsoTri.centroid)
#obtain scaling factor
scaleFac = 1/np.max(meshIsoTri.extents)
#scale mesh
meshIsoTri.apply_scale(scaleFac)

#format and export
transVert, transFace = tdnl.trimeshToDfNoLabels(meshIsoTri)
print("formated")
dpe.dfToPlyExport(vertDf = transVert, faceDf = transFace, outFile = outFile)
print("exported")
