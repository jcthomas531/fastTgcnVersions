import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools/registration")
import registrationFuns as rg
import os
import open3d as o3d
import copy
import trimesh
import numpy as np

import random
random.seed(826)


#EXAMPLE WORKFLOW
#iowaRmeData
os.chdir("K:/iowaRme/fullScans/pat058")
#second scan will serve as source, what is being transformed
u058_12 = o3d.io.read_point_cloud("pat058u_12.ply")
#first scan will serve as target
u058_01 = o3d.io.read_point_cloud("pat058u_01.ply")
#plot how it looks prior to transformation (with actual colors)
o3d.visualization.draw_geometries([u058_01])
o3d.visualization.draw_geometries([u058_12])
o3d.visualization.draw_geometries([u058_12, u058_01])
rg.monochromePlot(u058_12, u058_01)
#calculate registration
reg01_12 = rg.getRegistration(source = u058_12, target=u058_01)
#look at stats on the registration performance
reg01_12
#look at transformation matri
reg01_12.transformation
#transform source to align with target
u058_12Trans = u058_12.transform(reg01_12.transformation)
#plot registered point clouds
rg.monochromePlot(u058_12Trans, u058_01)



#EXAMPLE WORKFLOW
#the rme scans still need a bit of massaging into the correct format before they
#can be used in this process. for now we will register a teeth3ds scan to an iosseg
#scan (the will not line up exactly). Iosseg is the target, teeth3ds is the source
#obtaining transformation
iossegPath = "K:/IOSSegData/clean/trainCleanU/007_U.ply"
teeth3dsPath = "K:/teeth3DS/scanData/upperPlyDecim016/00OMSZGW_UDecim016.ply"
targetCloud = o3d.io.read_point_cloud(iossegPath)
sourceCloud = o3d.io.read_point_cloud(teeth3dsPath)
reg = rg.getRegistration(source = sourceCloud, target=targetCloud)
#register the meshes and export
os.chdir("K:/testDir/")
rg.registerAndExport(inFile = teeth3dsPath, outFile = "registerTest2.ply", trans = reg.transformation)
#check registration via o3d meshes and monochrome plot
mesh1 = o3d.io.read_triangle_mesh("registerTest2.ply")
mesh2 = o3d.io.read_triangle_mesh(iossegPath)
rg.monochromePlot(mesh1, mesh2)
#check coloring
import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
import plyFunctions as pf
pf.readAndPlot("registerTest2.ply", "U")
pf.readAndPlot(iossegPath, "U")
