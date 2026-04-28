import os
os.chdir("H:\\schoolFiles\\dissertation\\intraoralSegmentation\\tools")
import plyFunctions as pf
import pandas as pd
import re
import pyvista as pv
import numpy as np


#for running exampls
# trainPath = "P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\train"
# os.chdir(trainPath)



###############################################################################


#function that outputs some basic information about the file including if any
#teeth are labeled as NA
#pass fileName as a string
def numExtract(fileName):
    #extract patient information
    #the r here means raw, avoids double escaping
    #the function natrually outputs a list, so taking the first (and only) element
    patNum = re.findall(pattern = r"^[0-9]{3}", string = fileName)[0]
    patArch = re.findall(pattern = r"_([A-Z]{1})\.", string = fileName)[0]
    #read in file 
    pat = pf.readAndFormat(fileName, arch = patArch)
    #extract summary information
    numTeeth = len(pat["face"]["toothNum"].unique())-1 #subtract 1 for the gum group
    numVerts = pat["vert"].shape[0]
    numFaces = pat["face"].shape[0] 
    anyNaTeeth = pat["face"]["toothNum"].isna().any()
    return [fileName, patNum, patArch, numTeeth, numVerts, numFaces, anyNaTeeth]


#examples
# numExtract("036_L.ply")
# #this is like an apply
# datList = [numExtract(i) for i in os.listdir(trainPath)]
# dat = pd.DataFrame(datList, columns=['fileName', 'pat', 'arch', 'numTeeth', 'verts', 'faces', 'anyNaTeeth'])
# print(dat.to_string())

###############################################################################


#function that plots the points associated with any surfaces that have an NA value
#for the toothNum column
#dat takes a object from pf.readAndFormat()
def plotIssue(dat):
    #making a copy just for saftey
    datC = dat.copy()
    
    #get original surface
    datSurf = pf.giveSurf(face = datC["face"], vertex = datC["vert"])
    
    #identify problem faces
    datNaFace = datC["face"][datC["face"]["toothNum"].isna()]
    
    #get the points associated with the coloring
    #the explode function essentially flattens this column of lists and set keeps
    #only the unique values. made to a list for easy usage
    datVertInd = list(set(datNaFace["vertex_indices"].explode()))
    
    #get the points in the original scan that correspond to these indicies
    datVert = datC["vert"].iloc[datVertInd,]
    
    #plot these on top of the mesh
    plot1 = pv.Plotter()
    plot1.add_mesh(datSurf, scalars = "rgba", rgb = True)
    plot1.add_points(np.array(datVert.iloc[:,range(0,3)]),
                        color = "black", point_size=10,
                        render_points_as_spheres=True)
    return plot1.show()


#example
# os.chdir(trainPath)
# dat=pf.readAndFormat("036_L.ply", "L")
# plotIssue(dat)


###############################################################################



#function that finds all faces with NA tooth number and assigns it the tooth number
#and color of the closest color in that arch
#dat will take in an object created by pf.readAndFormat()
#outputs the same dat with the color and tooth number corrected
#should return an error if the data has already been cleaned
def colorCleaner(dat):
    #make a copy of the data for safe keeping
    datC = dat.copy()
    
    #extract if this is an upper or a lower arch
    arch = datC["face"]["arch"].iloc[0] #just taking the first one as they will all be the same
  #extract the faces that have missing toothNum
    naFaces = datC["face"][datC["face"]["toothNum"].isna()]
    
    #if there is no missing toothNum, this function should stop
    if len(naFaces.index) == 0:
        raise Exception("No missing faces missing a toothNum value")
    
    #create a dictionary with datasets for each color that has a missing toothNum
    #this will often be just a single 
    naFacesGrp = {g: df for g, df in naFaces.groupby("color")}
    
    #get the color and tooth number data frames
    if arch == "upper":
        numCol = pf.colorNumFrame("U")
    elif arch == "lower":
        numCol = pf.colorNumFrame("L")
    else:
        raise Exception("The arch column in the face data is not 'lower' or 'upper' and it must be one of these two things")
    
    
    
    #for each of these groups in the dictionary, we want to extract the color values and compare them to 
    #each of the colors for their arch
    for i in range(len(naFacesGrp)):
        
        groupi = list(naFacesGrp.values())[i]
        groupiCol = groupi[["red", "green", "blue"]].iloc[0,:] #just taking the first one as they will all be the same
        groupiCol = pd.to_numeric(groupiCol) #these are strings, switch to numeric
        
        #grab the index for these faces in the face data
        naInd = groupi.index
        
        #check against each color that should be in this arch
        #initialize a norm holder in color dataframe
        numCol["diffNorm"] = pd.NA
        for j in range(len(numCol)):
            #extract color from mapping df
            colj = numCol[["red", "green", "blue"]].iloc[j,:]
            #find the difference vector
            #this does not work...
            diffVec = groupiCol - colj
            #take the l2 norm
            numCol.loc[j, "diffNorm"] = np.linalg.norm(diffVec, ord = 2)
        
        
        #which tooth "should" it be
        corInd = numCol["diffNorm"].idxmin()
        corNum = numCol["toothNum"].iloc[corInd]
        corCol = numCol["color"].iloc[corInd]
        #overwrite the color and the tooth number in the original data and export it
        datC["face"].loc[naInd, "toothNum"] = corNum
        datC["face"].loc[naInd, "color"] = corCol
        datC["face"].loc[naInd, "red"] = pd.to_numeric(re.findall(r"(^[0-9]{3})", corCol)[0], errors='raise')
        datC["face"].loc[naInd, "green"] = pd.to_numeric(re.findall(r"^[0-9]{3}-([0-9]{3})-[0-9]{3}$", corCol)[0], errors='raise')
        datC["face"].loc[naInd, "blue"] = pd.to_numeric(re.findall(r"([0-9]{3}$)", corCol)[0], errors='raise')
        
        
    return datC

#example
# dat=pf.readAndFormat("036_L.ply", "L")
# plotIssue(dat)
# dat = colorCleaner(dat)
# pf.plotPly(dat["face"], dat["vert"])





###############################################################################
#formatting to be in same form as original ply files
def faceFormatter(dat):
    datC = dat.copy()
    datC["face"] = datC["face"][["vertex_indices", "red", "green", "blue", "alpha"]]
    return datC
#example
#see colorCleanProcess.py

###############################################################################
#exporting the cleaned and formatted files
#thanks chatgpt
def writePly(fileName, dat):
    """
    dat["vert"] : pandas DataFrame with columns:
        x, y, z, nx, ny, nz
        
    dat["face"] : pandas DataFrame with columns:
        vertex_indices (list of ints), red, green, blue, alpha
    """
    #copying for safe keeping
    datC = dat.copy()

    verts = datC["vert"]
    faces = datC["face"]

    n_verts = len(verts)
    n_faces = len(faces)

    with open(fileName, "w") as f:
        # ----- HEADER -----
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment color corrected via colorCleanProcess.py\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property uchar alpha\n")
        f.write("end_header\n")

        # ----- VERTEX DATA -----
        for _, row in verts.iterrows():
            f.write(f"{row.x} {row.y} {row.z} {row.nx} {row.ny} {row.nz}\n")

        # ----- FACE DATA -----
        for _, row in faces.iterrows():
            indices = row.vertex_indices  # must be list-like
            f.write(
                f"{len(indices)} " +
                " ".join(str(int(i)) for i in indices) + " " +
                f"{int(row.red)} {int(row.green)} {int(row.blue)} {int(row.alpha)}\n"
            )
#example
#see colorCleanProcess.py



###############################################################################
#function for full colorClean exporting all in one go
def fullExport(fileName, dat):
    #copying for saftey
    datC = dat.copy()
    
    datC = faceFormatter(dat = datC)
    writePly(fileName = fileName, dat = datC)
    #print("export completed")
    return "export completed"
    
    