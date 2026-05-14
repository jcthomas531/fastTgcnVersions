import trimesh
import pandas as pd

#function that takes a mesh with face colors and exports a vertex and face dataframe
def trimeshToDf_labels(x):
    #create vertex data frame
    vertDf = pd.DataFrame(x.vertices, columns = ["x", "y", "z"])
    #add vertex normal information
    vertNorms = x.vertex_normals
    vertDf["nx"] = vertNorms[:, 0]
    vertDf["ny"] = vertNorms[:, 1]
    vertDf["nz"] = vertNorms[:, 2]
    
    #create unlabeled face data frame
    faceDf = pd.DataFrame(x.faces, columns=["v1", "v2", "v3"])
    #get face metadata 
    faceMeta = x.metadata["_ply_raw"]["face"]["data"]
    faceDf["red"] = faceMeta["red"]
    faceDf["green"] = faceMeta["green"]
    faceDf["blue"] = faceMeta["blue"]
    faceDf["alpha"] = faceMeta["alpha"]
    #group vertex indices
    faceDf["vertex_indices"] = faceDf.apply(lambda row: [row["v1"], row["v2"], row["v3"]], axis=1)
    faceDf = faceDf[["vertex_indices", "red", "green", "blue", "alpha"]]
    #create a column with the 3 colors together as one variable
    #padding with leading zeros
    faceDf["color"] = (faceDf["red"].astype(str).str.zfill(3) + "-" +
                       faceDf["green"].astype(str).str.zfill(3) + "-" +
                       faceDf["blue"].astype(str).str.zfill(3))
    
    return {"vert": vertDf, "face": faceDf}


#example
# x = trimesh.load("K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016OriSeg/pat001u_preD_dec016Ori_seg.ply")
# trimeshToDf_labels(x)










