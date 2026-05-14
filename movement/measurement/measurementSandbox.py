import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
sys.path.append("Y:/dissModels/intraoralSegmentation/tools/registration")
import plyFunctions as pf
import os
import numpy as np
import pyvista as pv
import registrationFuns as rf
import open3d as o3d
import pandas as pd
import math

os.chdir("K:/iowaRme/testDir/segTestOutput")

#find and plot centroids
#centroids for pat055 at first time point
pat055u_01 = pf.readAndFormat("pat055u_01_dec016Form.ply", arch = "U")
pat055u_01Cent = pf.toothCentroids(face = pat055u_01["face"], vertex = pat055u_01["vert"])
#centroids for pat055 at second time point
pat055u_16 = pf.readAndFormat("pat055u_16_dec016Form.ply", arch = "U")
pat055u_16Cent = pf.toothCentroids(face = pat055u_16["face"], vertex = pat055u_16["vert"])

#something to consider here is that scans at different time points may display
#a different number of unique teeth. in the vast majority of cases, that is probably
#not actually true, but often in child arches, more classes are used than there
#are teeth, so the match up at time point one and time point 2 may not be 1 to 1

#surface for first time point
s1 = pf.giveSurf(face = pat055u_01["face"], vertex = pat055u_01["vert"])
s2 = pf.giveSurf(face = pat055u_16["face"], vertex = pat055u_16["vert"])
#plot first time point
plot1 = pv.Plotter()
plot1.add_mesh(s1, scalars = "rgba", rgb = True)
#plot1.add_mesh(s2, scalars = "rgba", rgb = True)
#add points for centroids at first time point
plot1.add_points(np.array(pat055u_01Cent.iloc[:,range(1,4)]),
                    color = "black", point_size=10,
                    render_points_as_spheres=True)
#add points for centroids at second time point
plot1.add_points(np.array(pat055u_16Cent.iloc[:,range(1,4)]),
                    color = "black", point_size=10,
                    render_points_as_spheres=True)
plot1.show()



#this shows that registration is necessary before this point
#there are many options for registration and since it is just a matrix multiplication
#it could be done at any time, even with the full scans
#for the time being, we will just register on the entire scan but this will
#probably need to be changed later


#obtaining transformation
#second scan will serve as source, what is being transformed
#first scan will serve as target, the base for the registration
targetFile = "pat055u_01_dec016Form.ply"
sourceFile = "pat055u_16_dec016Form.ply"
targetCloud = o3d.io.read_point_cloud(targetFile)
sourceCloud = o3d.io.read_point_cloud(sourceFile)
reg = rf.getRegistration(source = sourceCloud, target=targetCloud)
#register the meshes and export
rf.registerAndExport(inFile = sourceFile, outFile = "pat055u_16_dec016FormReg.ply", trans = reg.transformation)
#
#REALLY IMPORTANT THING TO NOTE: REGISTERED SCAN HAS 6 LESS VERTICES THAN NON_REGISTERED
#WHERE ARE THEY GOING
#
#check registration via o3d meshes and monochrome plot
mesh1 = o3d.io.read_triangle_mesh("pat055u_16_dec016FormReg.ply")
mesh2 = o3d.io.read_triangle_mesh(targetFile)
rf.monochromePlot(mesh1, mesh2)
#check coloring
# pf.readAndPlot("pat055u_16_dec016FormReg.ply", "U")
# pf.readAndPlot(targetFile, "U")


#doing the centroids again but this time with the registered second scan
#centroids for pat055 at first time point
pat055u_01 = pf.readAndFormat("pat055u_01_dec016Form.ply", arch = "U")
pat055u_01Cent = pf.toothCentroids(face = pat055u_01["face"], vertex = pat055u_01["vert"])
#centroids for pat055 at second time point
pat055u_16 = pf.readAndFormat("pat055u_16_dec016FormReg.ply", arch = "U")
pat055u_16Cent = pf.toothCentroids(face = pat055u_16["face"], vertex = pat055u_16["vert"])

#something to consider here is that scans at different time points may display
#a different number of unique teeth. in the vast majority of cases, that is probably
#not actually true, but often in child arches, more classes are used than there
#are teeth, so the match up at time point one and time point 2 may not be 1 to 1

#surface for first time point
s1 = pf.giveSurf(face = pat055u_01["face"], vertex = pat055u_01["vert"])
s2 = pf.giveSurf(face = pat055u_16["face"], vertex = pat055u_16["vert"])
#plot first time point
plot1 = pv.Plotter()
plot1.add_mesh(s1, scalars = "rgba", rgb = True)
plot1.add_mesh(s2, color = "red", opacity = .5)
#add points for centroids at first time point
plot1.add_points(np.array(pat055u_01Cent.iloc[:,range(1,4)]),
                    color = "black", point_size=10,
                    render_points_as_spheres=True)
#add points for centroids at second time point
plot1.add_points(np.array(pat055u_16Cent.iloc[:,range(1,4)]),
                    color = "red", point_size=10,
                    render_points_as_spheres=True)
plot1.show()





#lets calculate the distance between each centroid for the two time points
#give the two time points suffixes for their centroid location columns
toModify = pat055u_01Cent.columns[range(1,4)]
pat055u_01Cent = pat055u_01Cent.rename(columns={i: f"{i}_1" for i in toModify})
pat055u_16Cent = pat055u_16Cent.rename(columns={i: f"{i}_16" for i in toModify})
#join the two time points together
pat055uBothCent = pat055u_01Cent.merge(pat055u_16Cent, on = "toothNum", how = "left")
#calculate the distance vector between the two time points
pat055uBothCent["xDiff"] = pat055uBothCent["x_1"] - pat055uBothCent["x_16"]
pat055uBothCent["yDiff"] = pat055uBothCent["y_1"] - pat055uBothCent["y_16"]
pat055uBothCent["zDiff"] = pat055uBothCent["z_1"] - pat055uBothCent["z_16"]
#find the l2 norm of the difference vector
pat055uBothCent["l2Norm"] = (pat055uBothCent["xDiff"]**2 + pat055uBothCent["yDiff"]**2 + pat055uBothCent["zDiff"]**2) ** (1/2)
#unit vector values for the distance vector
pat055uBothCent["xDiffUnit"] = pat055uBothCent["xDiff"] / pat055uBothCent["l2Norm"]
pat055uBothCent["yDiffUnit"] = pat055uBothCent["yDiff"] / pat055uBothCent["l2Norm"]
pat055uBothCent["zDiffUnit"] = pat055uBothCent["zDiff"] / pat055uBothCent["l2Norm"]

#save as a csv so i can play with it in R
pat055uBothCent.to_csv("H:/schoolFiles/dissertation/movementModeling/measurement/firstMovement.csv",
                       index = False)
























