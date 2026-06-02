import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import pyvista as pv
import pyacvd  
import trimesh
import random
import numpy as np
import json
from scipy.spatial import cKDTree
import pandas as pd
import colorNumFrame as cnf
import trimeshToDf_labels as ttdl
import dfToPlyExport as dtpe

#seed setting
import os
os.environ["OMP_NUM_THREADS"] = "1"
seed = 826
random.seed(seed)
np.random.seed(seed)




#pull variables from snakemake
objFile = sys.argv[1]
jsonFile = sys.argv[2]
outFile = sys.argv[3]



#TESTING
# objFile = "K:/teeth3DS/scanData/upper/019TUUZD/019TUUZD_upper.obj"
# jsonFile = "K:/teeth3DS/scanData/upper/019TUUZD/019TUUZD_upper.json"
# outFile = "K:/testDir/test2.ply"

#load in obj file
meshTri = trimesh.load_mesh(objFile, process = False)

#convert to pyvista object
meshObj = pv.wrap(meshTri)

#load in json file
with open(jsonFile) as fp:
    labelDat = json.load(fp)

print("files loaded")

#extarct important pieces for use later
labelList = labelDat["labels"]
labelArray = np.array(labelList)

#give the pyvista object the appropriate labels
meshObj.point_data["labels"] = labelList

#make point data into a cKDTree so the remesh object can map to it
#NEED TO FIGURE OUT WHAT THIS ACTUALLY IS
tree = cKDTree(meshObj.points)

print("objects prepared pre-remesh")

#remesh
meshIso = meshObj.acvd.remesh(8500, subdivide=3)

print("remesh finished")

#map to original lables
#NEED TO FIGURE OUT WHAT THIS ACTUALLY IS
dist, idx = tree.query(meshIso.points)

print("remeshed object labeled")

#majority rules implimentation
#set up
faces = meshIso.faces.reshape(-1, 4)[:, 1:]
vertexLabs = labelArray[idx]
vertexLabsArray = vertexLabs[faces]
#apply rule
rng = np.random.default_rng(826)
overallLabHolder = []
for i in range(len(vertexLabsArray)):
    labsi = vertexLabsArray[i]
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

print("majority rules face labeling finished")

#get color mapping data frame
colorRefDef = cnf.colorNumFrame("U")

#function thing for mapping each label number to a color
#chatgpt wrote this piece, not exactly use what it is doing
labelToRgba = {
    row["fdiNum"]: [row["red"], row["green"], row["blue"], 255]
    for _, row in colorRefDef.iterrows()
}

#perform mapping
rgba = np.array([labelToRgba[l] for l in overallLabHolder], dtype=np.uint8)

print("label to color matching finished")

#make array into data frame
colorDf = pd.DataFrame({"red": rgba[:, 0],
                        "green": rgba[:, 1],
                        "blue": rgba[:, 2],
                        "alpha": rgba[:, 3]})

#convert pyvista object to trimesh
meshIsoTri = pv.to_trimesh(meshIso)

#convert trimesh to data frames
vDat, fDat = ttdl.trimeshToDf_labels(x = pv.to_trimesh(meshIso), colorDf = colorDf)

print("data to data frames")

#export
dtpe.dfToPlyExport(vertDf = vDat, faceDf = fDat, outFile = outFile)

print("export finished")