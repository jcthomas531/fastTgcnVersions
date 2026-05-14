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
# os.listdir()

#register second time points to first time points and save mesh
#pat055
# rf.fullRegistFlow(targetFile = "pat055u_01_dec016Form.ply",
#                   sourceFile = "pat055u_16_dec016Form.ply",
#                   registerFile = "pat055u_16_dec016FormReg.ply")
# #pat056
# rf.fullRegistFlow(targetFile = "pat056u_01_dec016Form.ply",
#                   sourceFile = "pat056u_15_dec016Form.ply",
#                   registerFile = "pat056u_15_dec016FormReg.ply")
# #pat057
# rf.fullRegistFlow(targetFile = "pat057u_01_dec016Form.ply",
#                   sourceFile = "pat057u_15_dec016Form.ply",
#                   registerFile = "pat057u_15_dec016FormReg.ply")
# #pat058
# rf.fullRegistFlow(targetFile = "pat058u_01_dec016Form.ply",
#                   sourceFile = "pat058u_12_dec016Form.ply",
#                   registerFile = "pat058u_12_dec016FormReg.ply")


#calculate distances and export
#pat055
pat055Diff = pf.twoScanCentroids(scan1File = "pat055u_01_dec016Form.ply",
                                 scan2File = "pat055u_16_dec016FormReg.ply")
pat055Diff.to_csv("H:/schoolFiles/dissertation/movementModeling/measurement/pat055uDist_dec016.csv",
                  index=False)
#pat056
pat056Diff = pf.twoScanCentroids(scan1File = "pat056u_01_dec016Form.ply",
                                 scan2File = "pat056u_15_dec016FormReg.ply")
pat056Diff.to_csv("H:/schoolFiles/dissertation/movementModeling/measurement/pat056uDist_dec016.csv",
                  index=False)
#pat057
pat057Diff = pf.twoScanCentroids(scan1File = "pat057u_01_dec016Form.ply",
                                 scan2File = "pat057u_15_dec016FormReg.ply")
pat057Diff.to_csv("H:/schoolFiles/dissertation/movementModeling/measurement/pat057uDist_dec016.csv",
                  index=False)
#pat058
pat058Diff = pf.twoScanCentroids(scan1File = "pat058u_01_dec016Form.ply",
                                 scan2File = "pat058u_12_dec016FormReg.ply")
pat058Diff.to_csv("H:/schoolFiles/dissertation/movementModeling/measurement/pat058uDist_dec016.csv",
                  index=False)


#visualizations
#scan1File, scan2File file paths to the time point 1 and registered time point 2
#cent is the centroid data frame
def registAndCentPlot(scan1File, scan2File, cent):
    #read in data
    scan1 = pf.readAndFormat(file = scan1File, arch = "U")
    scan2 = pf.readAndFormat(file = scan2File, arch = "U")
    
    #get surface
    surf1 = pf.giveSurf(face = scan1["face"], vertex = scan1["vert"])
    surf2 = pf.giveSurf(face = scan2["face"], vertex = scan2["vert"])
    
    #plot
    plot1 = pv.Plotter()
    plot1.add_mesh(surf1, scalars = "rgba", rgb = True)
    plot1.add_mesh(surf2, color = "red", opacity = .5)
    #plot1.add_mesh(surf2, scalars = "rgba", rgb = True, opacity = .5)
    #add points for centroids at first time point
    plot1.add_points(np.array(cent.loc[:,["x_pre", "y_pre", "z_pre"]]),
                        color = "black", point_size=10,
                        render_points_as_spheres=True)
    #add points for centroids at second time point
    plot1.add_points(np.array(cent.loc[:,["x_post", "y_post", "z_post"]]),
                        color = "red", point_size=10,
                        render_points_as_spheres=True)
    plot1.show()
    
    
    
    
    
    
registAndCentPlot(scan1File = "pat055u_01_dec016Form.ply",
                  scan2File = "pat055u_16_dec016FormReg.ply",
                  cent = pat055Diff)

registAndCentPlot(scan1File = "pat056u_01_dec016Form.ply",
                  scan2File = "pat056u_15_dec016FormReg.ply",
                  cent = pat055Diff)

registAndCentPlot(scan1File = "pat057u_01_dec016Form.ply",
                  scan2File = "pat057u_15_dec016FormReg.ply",
                  cent = pat055Diff)

registAndCentPlot(scan1File = "pat058u_01_dec016Form.ply",
                  scan2File = "pat058u_12_dec016FormReg.ply",
                  cent = pat055Diff)