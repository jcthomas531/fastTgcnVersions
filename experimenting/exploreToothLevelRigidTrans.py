from  plyfile import PlyData
import pandas as pd
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import readAndFormat as raf
import verticesByToothLabel as vtl
import numpy as np
import open3d as o3d

prePath = "K:/iowaExpansion/segResults/segResults_remeshT3dsEpoch270/pre/pat001Pre_modelReady_seg.ply"
postPath = "K:/iowaExpansion/segResults/segResults_remeshT3dsEpoch270/post/pat001Post_modelReady_seg.ply"


#some notes
#this is just based off raw segementation results, in reality, there would need
#to be some post hoc clean up to make these segmentations ready, fine for now


####
#read in data
####
preDf = raf.readAndFormat(file = prePath)
preDfFace = preDf["face"]
preDfVert = preDf["vert"]

postDf = raf.readAndFormat(file = postPath)
postDfFace = postDf["face"]
postDfVert = postDf["vert"]



preTeethVert = vtl.verticesByToothLabel(vertDat = preDfVert, faceDat = preDfFace)
postTeethVert = vtl.verticesByToothLabel(vertDat = postDfVert, faceDat = postDfFace)


#now we want to calculate the rigid transformation for each tooths vertices

#lets begin with just a single tooth



#make into point clouds
pre7Xyz = preTeethVert["toothVertXyz"]["gum"].to_numpy()
pre7Pcd = o3d.geometry.PointCloud()
pre7Pcd.points = o3d.utility.Vector3dVector(pre7Xyz)


post7Xyz = postTeethVert["toothVertXyz"]["gum"].to_numpy()
post7Pcd = o3d.geometry.PointCloud()
post7Pcd.points = o3d.utility.Vector3dVector(post7Xyz)


pre7Pcd.paint_uniform_color((1, 0.75, 0))
post7Pcd.paint_uniform_color((1, 0, 1))
o3d.visualization.draw_geometries([pre7Pcd, post7Pcd])

import copy
# mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
# T = np.eye(4)
# T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
# T[0, 3] = 1
# T[1, 3] = 1.3
# print(T)
# mesh_t = copy.deepcopy(mesh).transform(T)
# o3d.visualization.draw_geometries([mesh, mesh_t])


import pickle
#load in the rugae annot transformations
with open("K:/iowaExpansion/superimposition/transformations/annotRugaeTrans/pat001AnnotRugaeTrans.pkl", "rb") as f:
    t7 = pickle.load(f)



post7PcdT = copy.deepcopy(post7Pcd).transform(t7)

post7PcdT.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pre7Pcd, post7Pcd, post7PcdT])

#something about this transformation isnt seeming right
#that is bc these segmented scans are based on the "segReady" scans which are
#rotated to match a teeth3ds scan which completely changes the transformation that is
#needed to match the two scans. We can us the transformation that was created before
#but the segemented scans must be the same orientation as when that transformation was calculated



