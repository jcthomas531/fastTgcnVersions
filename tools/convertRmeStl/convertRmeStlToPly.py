import os
from  plyfile import PlyData
import pandas as pd
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh

#inFile, an RME scan stored as an stl
#outFile, where the formatted ply file should be stored

def convertRmeStlToPly(inFile, outFile):
    
    #read in as mesh
    mesh1 = trimesh.load_mesh(inFile,
                              process = True,
                              validate = True 
                              )
    
    #vertices
    #calculate vertex normals
    vertNorms = mesh1.vertex_normals
    #prepare vertex information for export
    vertExport = pd.DataFrame(mesh1.vertices, columns=["x", "y", "z"])
    vertExport["nx"] = vertNorms[:, 0]
    vertExport["ny"] = vertNorms[:, 1]
    vertExport["nz"] = vertNorms[:, 2]
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

    #faces
    face = pd.DataFrame(mesh1.faces, columns=["v1", "v2", "v3"])
    face[["red", "green", "blue", "alpha"]] = 255
    face["vertex_indices"] = face.apply(lambda row: [row["v1"], row["v2"], row["v3"]], axis=1)
    faceExport = face[["vertex_indices", "red", "green", "blue", "alpha"]]
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
    #export
    convPly.write(outFile)



#EXAMPLE
# os.chdir("K:/iowaRme/preDelivAndFinalScans/pat001/final")
# convertRmeStlToPly(inFile = "106923640_shell_occlusion_u.stl",
#             outFile = "K:/iowaRme/testDir/testStlConv/pat001uFinal3.ply")


