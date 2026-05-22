import pandas as pd



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