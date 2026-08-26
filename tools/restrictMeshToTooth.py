#function that takes the data as read in via readAndFormat and returns a similarly
#structured version that just restricts to tooth of interest
#dat is the data in the format from readAndFormat
#toothNum is "1" to "16" including "gum", this must be a character
def restrictMeshToTooth(dat, toothNum):
    
    #separate face and vert data
    vDat = dat["vert"]
    fDat = dat["face"]
    
    #restrict face data to only those with number of interest
    fDatRest = fDat.loc[fDat["toothNum"] == toothNum].copy()
    
    #index of associated vertices
    vertIndRest = [
        vert
        for sublist in fDatRest["vertex_indices"]
        for vert in sublist
        ]
    #only unique values, no duplicate vertices
    #this would also work with list(set(vertIndRest)), it just orders things a bit differently
    vertIndRest = list(dict.fromkeys(vertIndRest)) 
    #restrict vertex data to only those associated with tooth of interest
    vDatRest = vDat.iloc[vertIndRest].copy()





    #change vertex indices in face data to correspond to newly reset vertex ordering
    aaa = dict(zip(vDatRest.index, range(len(vDatRest.index))))
    fDatRest["vertex_indices"] = fDatRest["vertex_indices"].apply(
        lambda inds: [aaa[i] for i in inds]
        )
    
    #reset indices of both data frames
    fDatRest = fDatRest.reset_index(drop = True)
    vDatRest = vDatRest.reset_index(drop = True)
    

    return {"vert": vDatRest, "face": fDatRest}
