#face and vertex are the formatted data frames as we commonly use
#toothNums is a list of tooth numbers stored as strings ie ["1", "16"]
def toothGroupCent(face, vertex, toothNumList):
    
    faceC = face.copy()
    vertexC = vertex.copy()
    
    inds = faceC["toothNum"].isin(toothNumList)
    vertInds = faceC[inds]["vertex_indices"].explode().unique().tolist()
    vertVals = vertexC.iloc[vertInds,][["x", "y", "z"]]
    centDf = vertVals.mean().to_frame().T
    
    return centDf