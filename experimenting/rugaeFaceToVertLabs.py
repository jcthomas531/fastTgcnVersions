import numpy as np
from collections import defaultdict
prePath = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh.ply"


#select all faces that include vertex i
#extract number of faces labeled 0 and number of faces labeled 1
#if there are more labeled 0 vs 1 then the vertex gets that labal
#there is a bit of a problem here because the faces are labeled with rgba values and not numbers
#however, there are only two of them so i think we can weasel a to make this easy

#or perhaps it is better to actually just read this in as two data frames as we have in the past and operate that way
import sys
sys.path.append("tools")
import readAndFormat as raf
import pandas as pd
meshDat = raf.readAndFormat(file = prePath, arch = "U")
vDat = meshDat["vert"]
fDat = meshDat["face"]
nVert = vDat.shape[0]

#import plotArch 
#plotArch.plotArch(face = meshDat["face"], vertex = meshDat["vert"])

#approach
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








#

#this is majority rules for all faces associated with the vertex
#remove normals in favor of other normals which will be joined
vDat = vDat.drop(["nx", "ny", "nz"], axis = 1)


#read in local descriptors
ld = pd.read_csv("K:/iowaExpTest/localDescriptors/rugAnnotForm_cSOriMastRemesh_localDescr/pre/pat001Pre_localDescr.csv")
#switch data type to accomidate merge
ld[["x", "y", "z"]] = ld[["x", "y", "z"]].astype(np.float32)


#merge with labeled vertex data

vld = pd.merge(left=vDat, right = ld, how = "left", on = ["x", "y", "z"])
# vld.to_csv("test.csv", index = False)




# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(
#     vld["x"],
#     vld["y"],
#     vld["z"],
#     c=vld["label"],
#     cmap="coolwarm",
#     s=5
# )
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# plt.show()

