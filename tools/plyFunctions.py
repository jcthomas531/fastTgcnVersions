#helpful functions
import pandas as pd
import pyvista as pv
from  plyfile import PlyData
import numpy as np
import pdb

###############################################################################


#functions that just give the tooth number and color information for each arch
#there is probably a better way to do this but right now this will work
#arch is either "U" or "L"
def colorNumFrame(arch):
    if arch == "U":
        #the matchups
        numCol = pd.DataFrame(
            {
             "toothNum": ["16","15","14","13","12","11","10","3","2","1","4","9","5","6","8","7","gum"], #universal numbering system
             "fdiNum": [28, 27, 26, 25, 24, 23, 22, 16, 17, 18, 15, 21, 14, 13, 11, 12, 0], #FDI numbering system
             "color": ['155-048-255', '255-099-071', '255-211-155','131-111-255','255-106-106',
                       '060-179-113', '255-246-143', '255-000-255', '030-144-255', '000-255-127',
                       '000-255-255', '127-255-000', '255-255-000', '000-255-000', '255-000-000',
                       '000-000-255', '255-255-255']
            }
            )
        #making it nice to use
        numCol = numCol.assign(
            red = pd.to_numeric(numCol["color"].str.extract(r"(^[0-9]{3})")[0], errors='raise'),
            green = pd.to_numeric(numCol["color"].str.extract(r"^[0-9]{3}-([0-9]{3})-[0-9]{3}$")[0], errors='raise'),
            blue = pd.to_numeric(numCol["color"].str.extract(r"([0-9]{3}$)")[0], errors='raise')
            )
    elif arch == "L":
        #the matchups
        numCol = pd.DataFrame(
            {
            "toothNum": ["25","24","26","23","27","22","28","21","29","20","30","19","31","18","32","17","gum"], #universal numbering system
            "fdiNum": [41, 31, 42, 32, 43, 33, 44, 34, 45, 35, 46, 36, 47, 37, 48, 38, 0], #FDI numbering system
            "color": ['139-000-000', '255-048-048', '144-238-144', '000-191-255', '000-139-139',
                      '255-165-000', '000-000-139','202-255-112', '139-000-139', '200-255-255',
                      '255-105-180', '255-228-255',  '230-230-250', '255-155-255', '255-228-181',
                      '255-069-000', '255-255-255']
            }
            )
        #make individual columns for the colors for easy use and make them numeric
        numCol = numCol.assign(
            red = pd.to_numeric(numCol["color"].str.extract(r"(^[0-9]{3})")[0], errors='raise'),
            green = pd.to_numeric(numCol["color"].str.extract(r"^[0-9]{3}-([0-9]{3})-[0-9]{3}$")[0], errors='raise'),
            blue = pd.to_numeric(numCol["color"].str.extract(r"([0-9]{3}$)")[0], errors='raise')
            )
    else: 
        #for some reason this is not working
        raise ValueError("arch arguement must be either 'L' or 'U'")
        
        
        
    return numCol

# example
# colorNumFrame("L")
# colorNumFrame("U")

###############################################################################

#function that reads in a ply file and formats in how I like
#this will require PlyData and pandas package
def plyRead(file):
    #pdb.set_trace()
    #read in the object
    plyObject = PlyData.read(file)
    #get the vertex and face data
    plyVert = pd.DataFrame(plyObject["vertex"].data)
    plyFace = pd.DataFrame(plyObject["face"].data)
    #create new variable in face data for tooth color that concats RGB vals
    plyFace["color"] = (plyFace["red"].astype(str).str.zfill(3) + "-" +
                         plyFace["green"].astype(str).str.zfill(3) + "-" +
                         plyFace["blue"].astype(str).str.zfill(3))
    #return a dictionary of the vertex and face information
    #this dictionary object seems like a named list in R
    return {"vert": plyVert, "face": plyFace}
#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")

###############################################################################

#function that adds tooth number and other tooth characteristics to the face data frame
#face takes a face data frame that has been set up using plyRead
#arch takes a string "L" or "U" denoting the upper or lower arch
def toothVars(face, arch):
    #pdb.set_trace()
    #make copies of the dataframes so you dont edit in place
    faceC = face.copy()
    #color and tooth number associations
    
    
    
    
    # uNumCol = pd.DataFrame(
    #     {
    #      "toothNum": ["16","15","14","13","12","11","10","3","2","1","4","9","5","6","8","7","gum"], 
    #      "color": ['155-048-255', '255-099-071', '255-211-155','131-111-255','255-106-106',
    #                '060-179-113', '255-246-143', '255-000-255', '030-144-255', '000-255-127',
    #                '000-255-255', '127-255-000', '255-255-000', '000-255-000', '255-000-000',
    #                '000-000-255', '255-255-255']
    #     }
    #     )
    # lNumCol = pd.DataFrame(
    #     {
    #     "toothNum": ["25","24","26","23","27","22","28","21","29","20","30","19","31","18","32","17","gum"],
    #     "color": ['139-000-000', '255-048-048', '144-238-144', '000-191-255', '000-139-139',
    #               '255-165-000', '000-000-139','202-255-112', '139-000-139', '200-255-255',
    #               '255-105-180', '255-228-255',  '230-230-250', '255-155-255', '255-228-181',
    #               '255-069-000', '255-255-255']
    #     }
    #     )
    
    
    
    
    #identify if we have an upper or lower arch
    #get color and tooth number associations
    #merge with face
    #create variable in face distinguishing arch
    if arch == "U":
        
        uNumCol = colorNumFrame("U")
        uNumCol = uNumCol[["toothNum", "color"]] #subset for historical compatibility reasons
        
        faceC = faceC.merge(uNumCol, on="color", how = "left", validate = "many_to_one")
        faceC["arch"] = "upper"
    elif arch == "L":
        
        lNumCol = colorNumFrame("L")
        lNumCol = lNumCol[["toothNum", "color"]] #subset for historical compatibility reasons
        
        faceC = faceC.merge(lNumCol, on="color", how = "left", validate = "many_to_one")
        faceC["arch"] = "lower"
    else:
        raise ValueError("arch arguement must be either 'L' or 'U'")
    #return updated dataframe
    return faceC
#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l07 = plyRead("007_L.ply")
# l07["face"] = toothVars(l07["face"], arch = "L")

###############################################################################

#plot face and vertex dataframes, works with above but also raw reads from PlyData
#takes the ply data as data frames so you can manipulate it beforehand
#faces is a df of the faces
#vertices is a df of the vertices
def plotPly(face, vertex):
    
    #copy the dataframes so it doesnt edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    
    #faces
    #get the number of vertexes for each shape (3 here bc triangles)
    faceC["nVert"] = faceC["vertex_indices"].apply(len)
    #make the vertices for the shape into columns
    faceCExpand = pd.DataFrame(faceC["vertex_indices"].tolist(),
                 columns=["v1", "v2", "v3"])
    faceC = faceC.join(faceCExpand)
    #order the data in the way that pyvista expects it and remove extra pieces
    #the color codes come at a different step
    faceCPV = faceC[["nVert", "v1", "v2", "v3"]]
    faceCPV = faceCPV.to_numpy()
    
    #vertices
    #extract the relavent columns
    #this could also be done with the normalized coordinates if you want
    vertexC = vertexC[["x", "y", "z"]]
    #make it how pyvista likes
    vertexC = vertexC.to_numpy()
    
    #use the vertex and face information to form the mesh
    surf = pv.PolyData(vertexC, faceCPV)
    
    #colors
    colors_ = faceC[["red", "green", "blue", "alpha"]]
    colors_ = colors_.to_numpy()
    #add the color information to the mesh
    surf.cell_data["rgba"] = colors_
    
    return surf.plot(scalars = "rgba", rgb = True)


#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")
# l76["face"] = toothVars(l76["face"], arch = "L")
# plotPly(face = l76["face"], vertex = l76["vert"])

###############################################################################

#a comination of the first two functions that does them both at the same time
#takes file as a string
#arch takes a string "L" or "U" denoting which arch we are looking at
def readAndFormat(file, arch):
    pat = plyRead(file)
    pat["face"] = toothVars(pat["face"], arch=arch)
    return pat

#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\test")
# readAndFormat("001_L.ply", arch = "L")




###############################################################################

#a combination of the three above functions that simply reads in the ply file,
#formats it, and the plots it. 
#fileName is a string
#arch takes a string "L" or "U" denoting which arch we are looking at
def readAndPlot(file, arch):
    pat = readAndFormat(file = file, arch = arch)
    return plotPly(face = pat["face"], vertex = pat["vert"])

#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\test")
# readAndPlot("001_L.ply", arch = "L")


###############################################################################


#a function like plotPly that returns the surface instead of the image so that
#you can manipulate it and plot it later
def giveSurf(face, vertex):
    
    #copy the dataframes so it doesnt edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    
    #faces
    #get the number of vertexes for each shape (3 here bc triangles)
    faceC["nVert"] = faceC["vertex_indices"].apply(len)
    #make the vertices for the shape into columns
    faceCExpand = pd.DataFrame(faceC["vertex_indices"].tolist(),
                 columns=["v1", "v2", "v3"])
    faceC = faceC.join(faceCExpand)
    #order the data in the way that pyvista expects it and remove extra pieces
    #the color codes come at a different step
    faceCPV = faceC[["nVert", "v1", "v2", "v3"]]
    faceCPV = faceCPV.to_numpy()
    
    #vertices
    #extract the relavent columns
    #this could also be done with the normalized coordinates if you want
    vertexC = vertexC[["x", "y", "z"]]
    #make it how pyvista likes
    vertexC = vertexC.to_numpy()
    
    #use the vertex and face information to form the mesh
    surf = pv.PolyData(vertexC, faceCPV)
    
    #colors
    colors_ = faceC[["red", "green", "blue", "alpha"]]
    colors_ = colors_.to_numpy()
    #add the color information to the mesh
    surf.cell_data["rgba"] = colors_
    
    return surf

#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")
# l76["face"] = toothVars(l76["face"], arch = "L")
# s1 = giveSurf(face = l76["face"], vertex = l76["vert"])
# plotTest = pv.Plotter()
# plotTest.add_mesh(s1, scalars = "rgba", rgb = True)
# plotTest.show()


###############################################################################

#function that highlights a series of tooth numbers highlights them in color
#this does not check to make sure the number requested is in that arch, but
#that would be pretty easy to add
#it would also be nice to have a version of this that also returned just the surface
def toothHigh(face, vertex, toothNums):
    #make copies of the dataframes so you dont edit in place
    faceC = face.copy()
    vertexC = vertex.copy()
    #make all colors besides the one we are looking at white
    faceC["red"] = np.where(faceC["toothNum"].isin(toothNums), faceC["red"], 255)
    faceC["green"] = np.where(faceC["toothNum"].isin(toothNums), faceC["green"], 255)
    faceC["blue"] = np.where(faceC["toothNum"].isin(toothNums), faceC["blue"], 255)
    return plotPly(face = faceC, vertex = vertexC)
#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")
# l76["face"] = toothVars(l76["face"], arch = "L")
# toothHigh(l76["face"], l76["vert"], ["17", "30"])


###############################################################################

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
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train")
# l76 = plyRead("076_L.ply")
# l76["face"] = toothVars(l76["face"], arch = "L")
# tc = toothCentroids(face = l76["face"], vertex = l76["vert"])
# #can then be visualized via
# s1 = giveSurf(face = l76["face"], vertex = l76["vert"])
# plotTest = pv.Plotter()
# plotTest.add_mesh(s1, scalars = "rgba", rgb = True)
# plotTest.add_points(np.array(tc.iloc[:,range(1,4)]),
#                     color = "black", point_size=10,
#                     render_points_as_spheres=True)
# plotTest.show()



###############################################################################
#GET CENTROIDS FOR TWO SCANS
###############################################################################

#scan1File, scan2File are file paths to segmented (and ideally registered) meshes
def  twoScanCentroids(scan1File, scan2File):
    
    #hard coding arch to U, can change later
    scan1 = readAndFormat(scan1File, arch = "U")
    scan2 = readAndFormat(scan2File, arch = "U")
    
    #calculate each scans centroids
    scan1Cent = toothCentroids(face = scan1["face"], vertex = scan1["vert"])
    scan2Cent = toothCentroids(face = scan2["face"], vertex = scan2["vert"])
    
    #give the two time points suffixes for their centroid location columns
    toModify = scan1Cent.columns[range(1,4)] #this will be the same for both data frames
    scan1Cent = scan1Cent.rename(columns={i: f"{i}_pre" for i in toModify})
    scan2Cent = scan2Cent.rename(columns={i: f"{i}_post" for i in toModify})
    
    #it is possible based on incorrect segmentation that one scan will have a different
    #number of segmented classes than another, i believe the arithmatic below will
    #be able to handle missing values
    
    #since it is possible to have a different number of classes in the two data 
    #frames, we must set the join up so that all of the classes are represented
    if (len(scan1Cent) >=  len(scan1Cent)):
        joinDirection = "left"
    elif (len(scan1Cent) <  len(scan1Cent)):
        joinDirection = "right"
    else:
        raise ValueError("Error in determining data frame lenghts prior to join")
    
    #join the two time points together
    #pat055uBothCent = pat055u_01Cent.merge(pat055u_16Cent, on = "toothNum", how = "left")
    scanBothCent = scan1Cent.merge(scan2Cent, on = "toothNum", how = joinDirection)
    
    #calculate the values for the distance vector between the two time points
    scanBothCent["xDiff"] = scanBothCent["x_pre"] - scanBothCent["x_post"]
    scanBothCent["yDiff"] = scanBothCent["y_pre"] - scanBothCent["y_post"]
    scanBothCent["zDiff"] = scanBothCent["z_pre"] - scanBothCent["z_post"]
    
    #find the l2 norm of the distance vector
    scanBothCent["l2Norm"] = (scanBothCent["xDiff"]**2 + scanBothCent["yDiff"]**2 + scanBothCent["zDiff"]**2) ** (1/2)
    
    #unit vector values for the distance vector
    scanBothCent["xDiffUnit"] = scanBothCent["xDiff"]/scanBothCent["l2Norm"]
    scanBothCent["yDiffUnit"] = scanBothCent["yDiff"]/scanBothCent["l2Norm"]
    scanBothCent["zDiffUnit"] = scanBothCent["zDiff"]/scanBothCent["l2Norm"]
    
    return scanBothCent













