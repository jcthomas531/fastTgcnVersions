import sys
from collections import defaultdict
sys.path.append("tools")
import readAndFormat as raf


#details
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


#takes in a file path to a ply formatted in the usual manner
#label and non label colors are changable but the defaults are the currently set colors
#returns a dictionary with "vert" and "face", but noted this format is not the same as the usual format
#because "vert" contains another column called label

#see usage in

def faceToPointLabel_2Color (inFile, labelColor = "255-000-127", nonlabelColor = "255-255-255"):

    #read in data
    meshDat = raf.readAndFormat(file = inFile, arch = "U")
    vDat = meshDat["vert"]
    fDat = meshDat["face"]
    nVert = vDat.shape[0]

    #create dictionaries
    labelCounts = defaultdict(int)
    notCounts = defaultdict(int)

    #create tuple
    faceColorTuple = zip(fDat["vertex_indices"], fDat["color"])

    #double iteration
    for verts, color in faceColorTuple:
        for v in verts:
            if color == labelColor:
                labelCounts[v] += 1
            elif color == nonlabelColor:
                notCounts[v] += 1

    #fill in the vertex labels in an efficent manner
    #notice calling labelCounts[55] finds the value for the dictionary key 55 rather than the value for the 55th index
    vDat["label"] = [
        1 if labelCounts[i] >= notCounts[i] else 0
        for i in range(nVert)
    ]

    return {"vert": vDat, "face": fDat}


