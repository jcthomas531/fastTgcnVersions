import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.append("tools")
import readAndFormat as raf

#for testing
meshPath = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh.ply"
ldPath = "K:/iowaExpTest/localDescriptors/rugAnnotForm_cSOriMastRemesh_localDescr/pre/pat001Pre_localDescr.csv"
outPath = "K:/iowaExpTest/testDir/test.csv"

#arguements from snakemake
meshPath = sys.argv[1] 
ldPath = sys.argv[2]
outPath = sys.argv[3]

#read in data
meshDat = raf.readAndFormat(file = meshPath, arch = "U")
vDat = meshDat["vert"]
fDat = meshDat["face"]
nVert = vDat.shape[0]

#this is majority rules for all faces associated with the vertex
#combine the vertex indices for the face and color information into a list of tuples
#iterate through each of the tuples
#within each tuple, iterate through the three vertex indices for the face
#for each vertex index within the face, if the face is labeld, go to the vertexs entry in the 
#labeled dictionary and add one. if the face is not labeled, go to the vertexs entry in the 
#non-labeled dictionary and add one. This keeps a running tally for each vertex of the number
#of faces it is associated with that are both labeled and non-labled. Thus, as we iterate
#through the tuples and that particular vertex index comes up again, we keep adding to its
#value in the dictionary, notice calling labelCounts[55] finds the value for the 
#dictionary key 55 rather than the value for the 55th index
#NOTE THAT THIS IS HARD CODED WITH THE COLOR RGB VALUE AND IF THAT CHANGES THIS WILL BREAK

#create dictionaries
labelCounts = defaultdict(int)
notCounts = defaultdict(int)

#create tuple
faceColorTuple = zip(fDat["vertex_indices"], fDat["color"])

#double iteration
for verts, color in faceColorTuple:
    for v in verts:
        if color == "255-000-127":
            labelCounts[v] += 1
        elif color == "255-255-255":
            notCounts[v] += 1

#fill in the vertex lables in an efficent manner
#notice calling labelCounts[55] finds the value for the dictionary key 55 rather than the value for the 55th index
vDat["label"] = [
    1 if labelCounts[i] >= notCounts[i] else 0
    for i in range(nVert)
]

#remove normals in favor of other normals which will be joined
vDat = vDat.drop(["nx", "ny", "nz"], axis = 1)

#read in local descriptors
ld = pd.read_csv(ldPath)
#switch data type to accomidate merge
ld[["x", "y", "z"]] = ld[["x", "y", "z"]].astype(np.float32)

#merge with labeled vertex data
vld = pd.merge(left=vDat, right = ld, how = "left", on = ["x", "y", "z"])

#export
vld.to_csv(outPath, index = False)