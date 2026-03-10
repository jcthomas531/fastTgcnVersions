



import os
import pandas as pd
import trimesh
import json
import numpy as np
import sys
#local
#sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
#hpc
sys.path.append("/Users/jthomas48/dissModels/fastTgcnVersions/tools")
from plyFunctions import colorNumFrame
from plyfile import PlyData, PlyElement
sys.path.append("/Users/jthomas48/fastTgcnVersions/tools/decimation")
import meshDecimation as d
import pyvista as pv

###############################################################################
#VALIDATION FUNCTION
###############################################################################
#with scan data, we want to go into each subject folder and validate that there
#is a single .json file and a single .obj file. If there is not, we want to note
#that subject. Also, if there is already a ply file in there, that means we 
#have operated on this subject before and can skip

#validate whether teeth3DS conversion to ply should happen for each subject
#fullPath argument takes the path to the directory holding subdirectories for 
#each subject of the form "K:/teeth3DS/scanData/upper/" (including final slash)

#this will need to be edited each time it is used to reflect current state
def fileValidate(fullPath, arch):
    
    #get all subject ids
    fullPathCont = os.listdir(fullPath)
    
    #loop thru each subject and create validation data frame
    valiRows = []
    for i in range(len(fullPathCont)):
        #subject id
        idi = fullPathCont[i]
        #directory for specific subject
        subPath = fullPath + idi
        #directory contents for specific subject
        subPathCont = pd.Series(os.listdir(subPath))
        #data frame with validation
        if arch == "U":
            vali = pd.DataFrame({
                "id": idi,
                "numFiles": [len(subPathCont) == 3],
                "singleObj": [subPathCont.str.contains(r"\.obj$", regex = True).sum() == 1],
                "singleJson": [subPathCont.str.contains(r"\.json$", regex = True).sum() == 1],
                "fullPly": [subPathCont.str.contains(r"U\.ply$", regex = True).sum() == 1],
                #Decim016 means decimated to 16000 faces
                "noDecPly": [subPathCont.str.contains(r"UDecim016\.ply$", regex = True).sum() == 0]
                })
        elif arch == "L":
            vali = pd.DataFrame({
                "id": idi,
                "numFiles": [len(subPathCont) == 3],
                "singleObj": [subPathCont.str.contains(r"\.obj$", regex = True).sum() == 1],
                "singleJson": [subPathCont.str.contains(r"\.json$", regex = True).sum() == 1],
                "fullPly": [subPathCont.str.contains(r"\L.ply$", regex = True).sum() == 1],
                #Decim016 means decimated to 16000 faces
                "noDecPly": [subPathCont.str.contains(r"\LDecim016.ply$", regex = True).sum() == 0]
                })
        else:
            ValueError("arch must be 'U' or 'L'")
        
        vali["proceed"] = vali.all(axis=1)
        valiRows.append(vali)
    
    #concat validation rows
    valiationFrame = pd.concat(valiRows, ignore_index=True)
    
    return(valiationFrame)

#EXAMPLE
# a = fileValidate("K:/teeth3DS/scanData/upper/")


###############################################################################
#CONVERSION FUNCTION
###############################################################################
#NOTE: this process is not deterministic, there must be a seed set


#####
#HELPER FUNCTIONS
#####
#function to look up label for a particular vertex reference in the face data
#using a for loop like this is probably massively inefficient
def labelLookup(x, labels):
    #set up empty list
    labelHolder = []
    for i in range(len(x)):
        #get index of vertex label to find
        lookupIndex = x.iloc[i]
        #find and store vertex label in list
        labelHolder.append(labels[lookupIndex]) 
    return(labelHolder)



#####
#MAIN FUNCTION
#####
#nFace will not matter when decimate = False

def convert3DS(subPath, arch, rng, decimate = False, nFace = 16000):
    
    
    #extract id string from subPath
    idStr = subPath.strip("/").split("/")[-1]
    
    
    #some validation
    assert arch in {"U", "L"}, "arch argument must be either 'U' or 'L'"
    
    #directory contents for specific subject
    subPathCont = pd.Series(os.listdir(subPath))
    
    #.obj file
    objInd = subPathCont.str.contains(r"\.obj$", regex = True)
    objFile = subPathCont[objInd].iloc[0]
    objPath = subPath + objFile
    #interestingly, it seems that the indexing for the verticies within the face data
    #starts at 0 and not 1, needing no offset for python numbering system
    #actually, it seems that in the raw file this is not the case but when it is brought
    #into python it is changed
    #when it is exported, it changes back, that is convienent
    mesh = trimesh.load_mesh(objPath, process = False)
    
    #.json file
    jsonInd = subPathCont.str.contains(r"\.json$", regex = True)
    jsonFile = subPathCont[jsonInd].iloc[0]
    jsonPath = subPath + jsonFile
    #json.load() expects the json data to already be a string
    #if you just do json.load(filename) it's trying to interpret the filename as actual json data
    #so it must be done like this or after a similar fashion
    with open(jsonPath) as fp:
        labelDat = json.load(fp)
    
    #ensure that the number of labels match up to the points
    assert len(labelDat["labels"]) == len(mesh.vertices), idStr + "_" + arch + ": label length not equal to number of vertices"
    
    
    
    if decimate == True:
        
        #make mesh pyvista object
        meshPv = pv.wrap(mesh)
        #add label data to pyvista object
        meshPv.point_data["labels"] = labelDat["labels"]
        #decimate
        meshPvDec=d.decimate3DS(meshPv, nFace = nFace)
        #make mesh trimesh object like before
        meshDec = pv.to_trimesh(meshPvDec)
        #make vertex data into data frame
        vert = pd.DataFrame(meshDec.vertices, columns=["x", "y", "z"])
        vert["labels"] = meshPvDec.point_data["labels"]
        
        #make face data into data frame
        face = pd.DataFrame(meshDec.faces, columns=["v1", "v2", "v3"])
        face[["v1Label", "v2Label", "v3Label", "fdiNum"]] = np.nan
        #get labels for each vertex associated with the face
        face["v1Label"] = labelLookup(face["v1"], vert["labels"])
        face["v2Label"] = labelLookup(face["v2"], vert["labels"])
        face["v3Label"] = labelLookup(face["v3"], vert["labels"])
        
        #create file name
        #Decim016 means decimated to 16000 faces
        outFile = subPath + idStr + "_" + arch + "Decim016.ply"
        
    elif decimate == False:
        
        #make a dataframe of the vertices
        vert = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])
        vert["labels"] = labelDat["labels"]
        
        #make a data frame of the faces
        face = pd.DataFrame(mesh.faces, columns=["v1", "v2", "v3"])
        face[["v1Label", "v2Label", "v3Label", "fdiNum"]] = np.nan
        
        #use labelLookup function to get the label for each of the vertices in each face
        face["v1Label"] = labelLookup(face["v1"], labelDat["labels"])
        face["v2Label"] = labelLookup(face["v2"], labelDat["labels"])
        face["v3Label"] = labelLookup(face["v3"], labelDat["labels"])
        
        #create file name
        outFile = subPath + idStr + "_" + arch + ".ply"
        
    else:
        raise ValueError("decimate must be True or False")
    
    
    #get the overall label, majority rules 
    overallLabHolder = []
    for i in range(len(face)):
        #extract the labels for a face
        labsi = face.iloc[i][["v1Label", "v2Label", "v3Label"]]
        #get the number of unique labels for that face
        uniqLabCount = len(set(labsi))
        #impliment majority rules rationale
        #this will work when there is only 1 or 2 label choices
        if uniqLabCount in [1,2]:
            overallLabHolder.append(pd.Series(labsi).value_counts().idxmax().astype(int))
        elif uniqLabCount == 3:
            #randomly select one of the three labels 
            overallLabHolder.append(rng.choice(labsi, 1, replace = False)[0].astype(int)) #selecting first object here so it is single dimesnional
        else:
            raise ValueError("unique label counts not 1, 2, or 3")
    
    #put overall label into face data
    face["fdiNum"] = overallLabHolder
    
    #these labels are according to the FDI tooth numbering system, need to convert to 
    #the universial labels and colors that I have been using
    #get the color match ups
    fdiRgb = colorNumFrame(arch)[["fdiNum", "red", "green", "blue"]]
    
    #join on fdi labels
    faceJoin = pd.merge(face, fdiRgb, how = "left", on = "fdiNum", validate="many_to_one")
    
    #start getting things in the set up they should be for the ply export
    #vertex infromation
    vertExport = vert.drop(columns=["labels"])
    #face information
    faceExport = faceJoin[["v1", "v2", "v3", "red", "green", "blue"]].copy()
    faceExport.loc[:,"alpha"] = 255
    faceExport["vertex_indices"] = faceExport.apply(lambda row: [row["v1"], row["v2"], row["v3"]], axis=1)
    faceExport = faceExport[["vertex_indices", "red", "green", "blue", "alpha"]]
    
    
    
    #calculate vertex normals and add them to vertex infromation
    # Extract only xyz (not normals)
    vertices = vertExport[["x", "y", "z"]].to_numpy(dtype=np.float64)
    # Stack the vertex_indices column
    faces = np.vstack(faceExport["vertex_indices"].values).astype(np.int64)
    meshN = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = meshN.vertex_normals
    #put them in data frame
    vertExport["nx"] = normals[:, 0]
    vertExport["ny"] = normals[:, 1]
    vertExport["nz"] = normals[:, 2]
    
    #set up the ply file for export
    #vertex information
    vertex_dtype = [
            ('x',  'f4'),
            ('y',  'f4'),
            ('z',  'f4'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4'),
        ]
    vertPly = np.empty(len(vertExport), dtype=vertex_dtype)
    for name in ["x", "y", "z", "nx", "ny", "nz"]:
        vertPly[name] = vertExport[name].astype(np.float32).values
    vertReady = PlyElement.describe(vertPly, 'vertex')
    
    #face infromation
    face_dtype = [
        ('vertex_indices', 'O'),  # 'O' for object (list of ints)
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('alpha', 'u1')
    ]
    facesPly = np.empty(len(faceExport), dtype=face_dtype)
    #fill in values
    facesPly['vertex_indices'] = faceExport['vertex_indices'].values
    facesPly['red']   = faceExport['red'].values.astype(np.uint8)
    facesPly['green'] = faceExport['green'].values.astype(np.uint8)
    facesPly['blue']  = faceExport['blue'].values.astype(np.uint8)
    facesPly['alpha'] = faceExport['alpha'].values.astype(np.uint8)
    
    faceReady = PlyElement.describe(facesPly, 'face')
    
    
    #make ply file
    convPly = PlyData([vertReady, faceReady], text = True)
    #file name create in if statement
    convPly.write(outFile)
    
    if decimate == True:
        print(idStr + "_" + arch + " decimated")
    elif decimate == False:
        print(idStr + "_" + arch + " converted")
    else:
        raise ValueError("print error: decimate must be True or False")
    
    
    
    return(True)
    


#EXAMPLE
# fp = "K:/testDir/test3DS/"
# rng = np.random.default_rng(826)
# convert3DS(subPath = fp, arch = "U", rng = rng)




###############################################################################
#
###############################################################################

#nFace will not matter when decimate = False

def convertAll3DS(path, arch, rng, decimate = False, nFace = 16000):
    
    #validation on the files in the directory
    validFrame = fileValidate(fullPath = path, arch = arch)
    print(validFrame)
    #extract the ids of those that ready to be converted
    validId = validFrame.query("proceed == True")["id"].tolist()
    
    #set up data frame for conversion tracking
    convTrack = pd.DataFrame({
        "id": validId,
        "convComplete": False})
    
    #conversion
    for i in range(len(validId)):
        #construct subject path
        subPath = path + validId[i] + "/"
        #perform conversion and track success (function outputs true when its finished)
        convTrack.loc[convTrack["id"] == validId[i], "convComplete"] = \
            convert3DS(subPath = subPath, arch = arch, rng = rng,
                       decimate = decimate, nFace = nFace)
    
    #return validation frame and conversion tracker
    return validFrame, convTrack

# EXAMPLE
# path = "K:/testDir/convTest/"
# arch = "U"
# rng = np.random.default_rng(826)
# convertAll3DS(path, arch, rng)
    
    
































