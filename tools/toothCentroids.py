import numpy as np
import pandas as pd


#function that calculates the centroids for all teethtype in the face data
#its output is a dataframe and it includes "gum" centroid as well
#designed to work in the workflow established by previous functions
def toothCentroids(face, vertex):
    #make a copy of the data sets so we dont edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    #first we get all of the unique teeth in the face data
    #i am going to keep "gum" in here, we can discard it later
    uTeeth = faceC["toothNum"].unique()
    #make a data frame to hold all of the centroids
    centHolder = pd.DataFrame(np.nan, index=range(len(uTeeth)),
                              columns=["toothNum", "x", "y", "z"])
    centHolder["toothNum"] = uTeeth
    #loop through all uTeeth values
    for i in range(len(centHolder)):
        toothi = centHolder["toothNum"][i]
        #subset to only include observations with specified tooth num, then take just the vertex
        #indices column, then "explode" the lists into individual values, then get just 
        #the unique ones, then make it into a list
        vertInd = faceC[faceC["toothNum"] == toothi]["vertex_indices"].explode().unique().tolist()
        #now we want to take those indices and subset the vertex information to only 
        #include those, also take only the x,y,z coordinate
        vertVals = vertexC.iloc[vertInd,][["x", "y", "z"]]
        #calculate and store the centriods
        centHolder.iloc[i,range(1, 4)] = vertVals.mean()
    
    return centHolder

#example
# import os
# import sys
# sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
# import plyFunctions as pf
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = pf.plyRead("076_L.ply")
# l76["face"] = pf.toothVars(l76["face"], arch = "L")
# tc = toothCentroids(face = l76["face"], vertex = l76["vert"])
# #can then be visualized via
# s1 = pf.giveSurf(face = l76["face"], vertex = l76["vert"])
# plotTest = pv.Plotter()
# plotTest.add_mesh(s1, scalars = "rgba", rgb = True)
# plotTest.add_points(np.array(tc.iloc[:,range(1,4)]),
#                     color = "black", point_size=10,
#                     render_points_as_spheres=True)
# plotTest.show()