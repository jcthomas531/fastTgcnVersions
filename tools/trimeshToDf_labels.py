import trimesh
import pandas as pd

#when x and colorDf supplied
#function that takes an unlabeled trimesh object and a data frame of the colors
#associated with each face and exports a vertex and face (with colors) dataframe
#(for example of this see remeshFullPlyTeeth3DS.py)
#when just x is supplied FOR OLD USAGE
#function that takes a trimesh with face colors and exports a vertex and face dataframe
def trimeshToDf_labels(x, colorDf = None):
    #create vertex data frame
    vertDf = pd.DataFrame(x.vertices, columns = ["x", "y", "z"])
    #add vertex normal information
    vertNorms = x.vertex_normals
    vertDf["nx"] = vertNorms[:, 0]
    vertDf["ny"] = vertNorms[:, 1]
    vertDf["nz"] = vertNorms[:, 2]


    if colorDf is not None:
        #create unlabeled face data frame
        faceDf = pd.DataFrame(x.faces, columns=["v1", "v2", "v3"])
        #get face metadata 
        faceDf["red"] = colorDf["red"]
        faceDf["green"] = colorDf["green"]
        faceDf["blue"] = colorDf["blue"]
        faceDf["alpha"] = colorDf["alpha"]
        #group vertex indices
        faceDf["vertex_indices"] = faceDf.apply(lambda row: [row["v1"], row["v2"], row["v3"]], axis=1)
        faceDf = faceDf[["vertex_indices", "red", "green", "blue", "alpha"]]
        return vertDf, faceDf
    else:
        #THIS IS A SPECIAL CASE FOR COMPATIBILITY WITH OLD USE
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










