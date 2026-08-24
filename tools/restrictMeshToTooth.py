#function that takes the data as read in via readAndFormat and returns a similarly
#structured version that just restricts to tooth of interest
#dat is the data in the format from readAndFormat
#toothNum is "1" to "16" including "gum", this must be a character
def restrictMeshToTooth(dat, toothNum):
    
    #separate face and vert data
    vDat = dat["vert"]
    fDat = dat["face"]
    
    #restrict face data to only those with number of interest
    fDatRest = fDat.loc[fDat["toothNum"] == toothNum]
    
    #index of associated vertices
    vertIndRest = list({
        vert
        for sublist in fDatRest["vertex_indices"]
        for vert in sublist
        })
    
    #restrict vertex data to only those associated with tooth of interest
    vDatRest = vDat.iloc[vertIndRest]
    
    return {"vert": vDatRest, "face": fDatRest}