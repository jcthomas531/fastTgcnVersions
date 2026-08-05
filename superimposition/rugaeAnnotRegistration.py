import sys
sys.path.append("tools")
import open3d as o3d
import numpy as np
import pickle
import trimesh
import copy
import random

import getRegistration as gr
import dfToPlyExport as dpe
import trimeshToDf_labels as ttdl
import trimeshExtractFaceLabels as tefl


#this as well as the os.environ statemetn up top is necessary for reproducing randomness
seed = 826
random.seed(seed)
np.random.seed(seed)
o3d.utility.random.seed(seed)




#testing
# prePath = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMast/pre/pat001Pre_formCSOriMast.ply"
# postPath = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMast/post/pat001Post_formCSOriMast.ply"
# transPath = "Y:/dissModels/intraoralSegmentation/superimposition/testPickle.pkl"
# outPlyPath = "K:/iowaExpTest/testDir/testPly.ply"

#pull variables from snakemake
prePath = sys.argv[1]
postPath = sys.argv[2]
transPath = sys.argv[3]
outPlyPath = sys.argv[4]


####
#helper function
####
#reads in the data and returns a point cloud for the annotated region
def getLabeledPC(filePath):
    #load in mesh and extrach vertex and face information as data frames
    mesh = trimesh.load(filePath, process = False)
    colorDf = tefl.trimeshExtractFaceLabels(mesh)
    vDf, fDf = ttdl.trimeshToDf_labels(mesh, colorDf=colorDf)
    
    #create data frame of only labeled faces
    labFaces = fDf.loc[(fDf["red"] == 255) & (fDf["green"] == 0) & (fDf["blue"] == 127)]
    
    #create list of unique vertices associated with labeled faces
    vertInd = list(set(
        vert
        for sublist in labFaces["vertex_indices"]
        for vert in sublist
        ))
    
    #subset vertex data frame to only those associated with labeled faces, only x y z coords, convert to numpy
    labVerts = vDf.iloc[vertInd].loc[:,["x", "y", "z"]].to_numpy()
    
    #convert to o3d point cloud
    labPointCloud = o3d.geometry.PointCloud()
    labPointCloud.points = o3d.utility.Vector3dVector(labVerts)
    
    return labPointCloud



####
#read in scans and return labeled area as point cloud
####
preCloud = getLabeledPC(prePath)
postCloud = getLabeledPC(postPath)
#o3d.visualization.draw_geometries([postCloud])



####
#transformation for superimposition
####
#obtain registration
regTrans = gr.getRegistration(source = postCloud, target = preCloud)

#export transformation for future use
#cannot export the entire object easily, just exporting transformation now but
#can return here later to export more pieces of the object if they become necessary
filePath = open(transPath, "wb")
pickle.dump(obj = regTrans.transformation,
            file = filePath)
filePath.close()
#can be read in like: 
# with open("Y:/dissModels/intraoralSegmentation/superimposition/testPickle.pkl", "rb") as f:
#     obj = pickle.load(f)

####
#apply transformation for superimposition
####
#load in files as trimesh objects
preMesh = trimesh.load(prePath, process = False)
postMesh = trimesh.load(postPath, process = False)
#copy post mesh and apply transformation, must copy bc the trans happens in palce
postMeshTrans = copy.deepcopy(postMesh)
postMeshTrans.apply_transform(regTrans.transformation) #this occurs in place

#extract colors and convert to data frames for export
colorDfTrans = tefl.trimeshExtractFaceLabels(postMeshTrans)
transDfVert, transDfFace = ttdl.trimeshToDf_labels(postMeshTrans, colorDf=colorDfTrans)

#export
dpe.dfToPlyExport(vertDf = transDfVert, faceDf = transDfFace, outFile = outPlyPath)

