import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement


#function that takes a trimesh object and creates the vertex and face information
#formatted as data frames so they are easy to use in other formats or export
#this will only work if the trimesh has vertex normals, but this is the most
#common case that i will be working in
#do not use this with labeled scans, this sets each face rgba value to 255 which
#is what we are using in the unlabeled scans
#
#x is a trimesh object
def trimeshToDfNoLabels(x):
    
    #create vertex data frame
    vertDf = pd.DataFrame(x.vertices, columns = ["x", "y", "z"])
    #add vertex normal information
    vertNorms = x.vertex_normals
    vertDf["nx"] = vertNorms[:, 0]
    vertDf["ny"] = vertNorms[:, 1]
    vertDf["nz"] = vertNorms[:, 2]
    
    #create unlabeled face data frame
    faceDf = pd.DataFrame(x.faces, columns=["v1", "v2", "v3"])
    faceDf[["red", "green", "blue", "alpha"]] = 255
    #group vertex indices
    faceDf["vertex_indices"] = faceDf.apply(lambda row: [row["v1"], row["v2"], row["v3"]], axis=1)
    faceDf = faceDf[["vertex_indices", "red", "green", "blue", "alpha"]]
    
    return vertDf, faceDf


#function to export mesh as a ply in the standard format with the vertex and face
#information stored as pandas data frames. The idea is that if we can get any representation
#of a mesh into a data frame format, we can export it in the standrard format used 
#throughout the analysis
#
#vertDf, a pandas data frame with columns x, y, z, nx, ny, nz in that order
#faceDf, a pandas data frame with columns vertex_indices, red, green, blue, alpha in that order
#where vertex_indices is of format ["v1", "v2", "v3"]
def dfToPlyExport(vertDf, faceDf, outFile):
    
    #vertex infromation
    vertex_dtype = [
            ('x',  'f4'),
            ('y',  'f4'),
            ('z',  'f4'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4'),
        ]
    #prepare vertex object
    vertPly = np.empty(len(vertDf), dtype=vertex_dtype)
    #put in values in the correct format
    for name in ["x", "y", "z", "nx", "ny", "nz"]:
        vertPly[name] = vertDf[name].astype(np.float32).values
    #prepatre as ply element
    vertReady = PlyElement.describe(vertPly, 'vertex')
    

    #face information
    face_dtype = [
        ('vertex_indices', 'O'),  # 'O' for object (list of ints)
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('alpha', 'u1')
    ]
    #prepare face object
    facesPly = np.empty(len(faceDf), dtype=face_dtype)
    #put in values in the correct format
    facesPly['vertex_indices'] = faceDf['vertex_indices'].values
    facesPly['red']   = faceDf['red'].values.astype(np.uint8)
    facesPly['green'] = faceDf['green'].values.astype(np.uint8)
    facesPly['blue']  = faceDf['blue'].values.astype(np.uint8)
    facesPly['alpha'] = faceDf['alpha'].values.astype(np.uint8)
    #prepare as ply element
    faceReady = PlyElement.describe(facesPly, 'face')
    
    #combine the vertex and face information
    plyReady = PlyData([vertReady, faceReady], text = True)
    #export
    plyReady.write(outFile)