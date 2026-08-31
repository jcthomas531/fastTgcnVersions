import numpy as np
#function for extracting the vertex data for each tooth label
#vertDat is the formatted vertex data frame
#facedat is the formatted face data frame
def verticesByToothLabel(vertDat, faceDat):
    #list of unique tooth numbers in the scan
    uniqToothNums = list(set(faceDat["toothNum"]))

    #create dictionary for each tooths face infromation
    toothFaceDfs = {}
    #create dictionary for vertex indices of each toothNum
    #this allows an individual point to be in multiple categories
    #this may or may not be what we want, can also have some sort of majority rules criteria
    toothVertexIndices = {}
    #create dictionary of the vertex infromation for each tooth
    #one with all vertex info and one with just xyz info
    toothVertexDfs = {}
    toothVertexDfsXYZ = {}
    
    for i in uniqToothNums:
        #face information
        toothFaceDfs[i] = faceDat[faceDat["toothNum"] == i]
        #vertex indices
        toothVertexIndices[i] = list(set(np.concatenate(toothFaceDfs[i]["vertex_indices"].values)))
        #vertex infromation
        toothVertexDfs[i] = vertDat.iloc[toothVertexIndices[i]]
        toothVertexDfsXYZ[i] = vertDat.iloc[toothVertexIndices[i]][["x", "y", "z"]]
    #note that the new subset toothVertexDfs no longer match the indices in toothFaceDfs
    #but this is probably fine for now 
    
    #only returning the vertex information for now
    return {"toothVertDfs": toothVertexDfs,
            "toothVertXyz": toothVertexDfsXYZ}
    
#example
# import sys
# sys.path.append("tools")
# import readAndFormat as raf
# prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
# #postPath = "K:/iowaExpansion/segResults/segResults_remeshT3dsEpoch270/post/pat001Post_modelReady_seg.ply"
# preDf = raf.readAndFormat(file = prePath)
# preDfFace = preDf["face"]
# preDfVert = preDf["vert"]
# aaa = verticesByToothLabel(vertDat = preDfVert, faceDat = preDfFace)
