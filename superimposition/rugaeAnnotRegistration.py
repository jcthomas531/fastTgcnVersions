import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import getRegistration as gr
import trimeshToDfNoLabels as tdnl
import dfToPlyExport as dpe

from  plyfile import PlyData
import pandas as pd
import open3d as o3d
import numpy as np
import pickle
import trimesh
import copy

#testing
prePath = "K:/iowaExpansion/fullRugaeAnnotScans/pre/pat001Pre_annot.ply"
postPath = "K:/iowaExpansion/fullRugaeAnnotScans/post/pat001Post_annot.ply"
transPath = "Y:/dissModels/intraoralSegmentation/superimposition/testPickle.pkl"
outPlyPath = "K:/iowaExpansion/testDir/testPly.ply"


####
#read in data
####
#small helper function
def datToDf(x):
    xDf = {
        "vert": pd.DataFrame(x["vertex"].data), 
        "face": pd.DataFrame(x["face"].data)
        }
    return xDf

#for pre data
preDat = PlyData.read(prePath)
preDf = datToDf(preDat)

#for post data
postDat = PlyData.read(postPath)
postDf = datToDf(postDat)


####
#create o3d point cloud object with just vertices that are labeld as 1
####
#small helper function
def labeledCloud(vertData):
    #just the vertexes labeled with 1
    vertLab = (
        vertData.loc[vertData["scalar_Classification"] == 1, ["x", "y", "z"]]
        .to_numpy()
               )
    #convert to open3d point cloud
    labPointCloud = o3d.geometry.PointCloud()
    labPointCloud.points = o3d.utility.Vector3dVector(vertLab)
    return labPointCloud

#for pre data
preCloud = labeledCloud(preDf["vert"])

#for post data
postCloud = labeledCloud(postDf["vert"])

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
#note, this transformation retains the scalar classification variable for the vertex data
# postMeshTrans.metadata["_ply_raw"]["vertex"]["data"]["scalar_Classification"]


#format and export transformed mesh
transDfVert, transDfFace = tdnl.trimeshToDfNoLabels(postMeshTrans,
                                                    pointLab=postMeshTrans.metadata["_ply_raw"]["vertex"]["data"]["scalar_Classification"])
dpe.dfToPlyExport(vertDf = transDfVert, faceDf = transDfFace, outFile = outPlyPath, pointLabCol = "scalar_Classification")

