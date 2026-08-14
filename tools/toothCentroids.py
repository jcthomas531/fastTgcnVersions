import numpy as np
import pandas as pd
import sys
sys.path.append("tools")
#sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import toothGroupCent as tgc



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
    
    #overall centroid for the entire scan
    #get unique vertex indices for all faces
    overallVertInd = faceC["vertex_indices"].explode().unique().tolist()
    #get vertex values
    overallVertVals = vertexC.iloc[overallVertInd,][["x", "y", "z"]]
    #calculate centroid
    overallCentDf = overallVertVals.mean().to_frame().T
    #add label
    overallCentDf.insert(0, "toothNum", "allScan")
    
    #overall centroid for the teeth in the scan, ie gum excluded
    #get all gum indices from face data
    gumInd = faceC["toothNum"].isin(["gum"])
    #get vertex indices for all vertices that are not gums
    noGumVertInd = faceC[~gumInd]["vertex_indices"].explode().unique().tolist()
    #get vertex values
    noGumVertVals = vertexC.iloc[noGumVertInd,][["x", "y", "z"]] 
    #calculate centroid
    noGumCentDf = noGumVertVals.mean().to_frame().T
    #add label
    noGumCentDf.insert(0, "toothNum", "noGum")
    
    #centroid for tooth type groups
    #only perform the calculation when there is an even number of teeth in the group
    #ie there is not one on one side but not on the other
    molars = ["1","2","3","14","15","16"]
    teethInMolars = [i for i in molars if i in uTeeth]
    premolars = ["4", "5", "12", "13"]
    teethInPremolars = [i for i in premolars if i in uTeeth]
    posterior = molars + premolars
    teethInPosterior = teethInMolars + teethInPremolars
    canines = ["6", "11"]
    teethInCanines = [i for i in canines if i in uTeeth]
    incisors = ["7", "8", "9", "10"]
    teethInIncisors = [i for i in incisors if i in uTeeth]
    anterior = canines + incisors
    teethInAnterior = teethInCanines + teethInIncisors
    #for when the present teeth are odd
    emptyXyz = pd.DataFrame({"x": [np.nan],"y": [np.nan],"z": [np.nan]})
    
    
    #molar
    if len(teethInMolars) % 2 == 0:
        molarCent = tgc.toothGroupCent(face = faceC, vertex=vertexC, toothNumList=molars)
    else:
        molarCent = emptyXyz
    #premolar
    if len(teethInPremolars) % 2 == 0:
        premolarCent = tgc.toothGroupCent(face = faceC, vertex = vertexC, toothNumList = premolars)
    else:
        premolarCent = emptyXyz
    #posterior
    if len(teethInPosterior) % 2 == 0:
        posteriorCent = tgc.toothGroupCent(face = faceC, vertex = vertexC, toothNumList = posterior)
    else:
        posteriorCent = emptyXyz
    #canines
    if len(teethInCanines) % 2 == 0:
        canineCent = tgc.toothGroupCent(face = faceC, vertex = vertexC, toothNumList = canines)
    else:
        canineCent = emptyXyz
    #incisors
    if len(teethInIncisors) % 2 == 0:
        incisorCent = tgc.toothGroupCent(face = faceC, vertex = vertexC, toothNumList = incisors)
    else:
        incisorCent = emptyXyz
    #anterior
    if len(teethInAnterior) % 2 == 0:
        anteriorCent = tgc.toothGroupCent(face = faceC, vertex = vertexC, toothNumList = anterior)
    else:
        anteriorCent = emptyXyz
    
    groupCentDf = pd.concat([molarCent, premolarCent, posteriorCent, canineCent, incisorCent, anteriorCent])
    groupCentDf.insert(0, "toothNum", ["molar", "premolar", "posterior", "canine", "incisor", "anterior"])
    
    #add rows for the overall scan centroid and the centroid for all teeth
    centHolder2 = pd.concat([centHolder, overallCentDf, noGumCentDf, groupCentDf], ignore_index=True)
    
    return centHolder2

#example
# import os
# import sys
# sys.path.append("tools")
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