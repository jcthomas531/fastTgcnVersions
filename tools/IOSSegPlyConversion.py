import trimesh
import os
os.chdir("K:/teeth3DS/dataAllParts/upper/00OMSZGW")
trimesh.exchange.load.mesh_formats()
m1 = trimesh.load_mesh("00OMSZGW_upper.obj", process = False)
#interestingly, it does not load in the color information that is stored with the vertices
m1.vertices
m1.faces



import json
#json.load() expects the json data to already be a string
#if you just do json.load(filename) it's trying to interpret the filename as actual json data
#so it must be done like this or after a similar fashion
with open("00OMSZGW_upper.json") as fp:
    labDat = json.load(fp)


#the labels are given to each point, not each face
len(labDat["labels"]) == len(m1.faces)
len(labDat["labels"]) == len(m1.vertices)

#majority rule point to face labels
#first, must attach the labels to the points
#we will do a sort of look up procedure
import numpy as np
import pandas as pd
#make a dataframe of the vertices
vert1 = pd.DataFrame(m1.vertices, columns=["x", "y", "z"])
vert1["labels"] = labDat["labels"]
#make a data frame of the faces
face1 = pd.DataFrame(m1.faces, columns=["v1", "v2", "v3"])
face1[["v1Label", "v2Label", "v3Label", "fdiNum"]] = np.nan


#####################################################################################################################
#interestingly, it seems that the indexing for the verticies within the face data
#starts at 0 and not 1, needing no offset for python numbering system
#actually, it seems that in the raw file this is not the case but when it is brought
#into python it is changed
#this is something to be aware of when outputting, need to check behavior and compare it 
#to the ply files we already have
#####################################################################################################################


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

#use the function to get the label for each of the vertices in each face
face1["v1Label"] = labelLookup(face1["v1"], labDat["labels"])
face1["v2Label"] = labelLookup(face1["v2"], labDat["labels"])
face1["v3Label"] = labelLookup(face1["v3"], labDat["labels"])


#get the overall label
overallLabHolder = []
for i in range(len(face1)):
    #extract the labels for a face
    labsi = face1.iloc[i][["v1Label", "v2Label", "v3Label"]]
    #get the number of unique labels for that face
    uniqLabCount = len(set(labsi))
    #impliment majority rules rationale
    #this will work when there is only 1 or 2 label choices
    if uniqLabCount in [1,2]:
        overallLabHolder.append(pd.Series(labsi).value_counts().idxmax().astype(int))
    elif uniqLabCount == 3:
        #just gonna take the max, that way if it is possibly identifying a tooth
        #we error on the side of it identifying a tooth (bc i believe 0 is gums, worth a check tho)
        #there seems to be very few of these so its probably not the biggest deal but it is
        #still good to have proper logic for it. it would probably be better to have
        #it randomly select the value but that would make this process non-deterministic
        #and I am not confident enough yet in the seeding system to play that game
        overallLabHolder.append(int(max(labsi)))
    else:
        raise ValueError("unique label counts not 1, 2, or 3")


face1["fdiNum"] = overallLabHolder
#these labels are according to the FDI tooth numbering system, need to convert to 
#the labels and colors that I have been using


set(face1["fdiNum"])



import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
from plyFunctions import colorNumFrame
colorNumFrame("U")
fdiRgbU = colorNumFrame("U")[["fdiNum", "red", "green", "blue"]]



face2 = pd.merge(face1, fdiRgbU, how = "left", on = "fdiNum", validate="many_to_one")



#start getting things in the set up they should be for the ply export
vert2 = vert1.drop(columns=["labels"])


face2 = face2[["v1", "v2", "v3", "red", "green", "blue"]]
face2.loc[:,"alpha"] = 255
face2["vertex_indices"] = face2.apply(lambda row: [row["v1"], row["v2"], row["v3"]], axis=1)
face2 = face2[["vertex_indices", "red", "green", "blue", "alpha"]]




#calculate vertex normals and add them to vertex infromation
import trimesh
# Extract only xyz (not normals)
vertices = vert2[["x", "y", "z"]].to_numpy(dtype=np.float64)
# Stack the vertex_indices column into (16000, 3)
faces = np.vstack(face2["vertex_indices"].values).astype(np.int64)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
normals = mesh.vertex_normals
#put them in data frame
vert2["nx"] = normals[:, 0]
vert2["ny"] = normals[:, 1]
vert2["nz"] = normals[:, 2]







from plyfile import PlyData, PlyElement
vertex_dtype = [
        ('x',  'f4'),
        ('y',  'f4'),
        ('z',  'f4'),
        ('nx', 'f4'),
        ('ny', 'f4'),
        ('nz', 'f4'),
    ]
    
newVert = np.empty(len(vert2), dtype=vertex_dtype)
    
for name in ["x", "y", "z", "nx", "ny", "nz"]:
    newVert[name] = vert2[name].astype(np.float32).values
    
newVertReady = PlyElement.describe(newVert, 'vertex')
    
# 1. Define structured dtype for PLY face element
face_dtype = [
    ('vertex_indices', 'O'),  # 'O' for object (list of ints)
    ('red', 'u1'),
    ('green', 'u1'),
    ('blue', 'u1'),
    ('alpha', 'u1')
]

# 2. Create empty structured array
faces_np = np.empty(len(face2), dtype=face_dtype)

# 3. Fill in structured array from DataFrame
faces_np['vertex_indices'] = face2['vertex_indices'].values
faces_np['red']   = face2['red'].values.astype(np.uint8)
faces_np['green'] = face2['green'].values.astype(np.uint8)
faces_np['blue']  = face2['blue'].values.astype(np.uint8)
faces_np['alpha'] = face2['alpha'].values.astype(np.uint8)

# 4. Wrap as PlyElement
faces_element = PlyElement.describe(faces_np, 'face')


os.chdir("K:/testDir")
convPly = PlyData([newVertReady, faces_element], text = True)
convPly.write("3dsConvTest.ply")
