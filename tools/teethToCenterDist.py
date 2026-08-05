import numpy as np

#function to calculate distance from each tooth to a central centroid
#x is the df output from toothCentroids()
#teeth is either "all" or a list of numebers for selecting certain teeth ie [5, 7]
#center is the centroid to measure to, "noGum", "allScan", or "gum"
def teethToCenterDist(x, center = "noGum"):

    teethNums = list(map(str, list(range(1,17))))
    #filter centroid data to only teeth of interest
    toothDat = x.loc[x["toothNum"].isin(teethNums)].copy()
    centerDat = x.loc[x["toothNum"] == center].copy()
    
    #distance from each tooth centroid to the center centroid
    toothDat["xDistCent"] = toothDat["x"] - centerDat["x"].iloc[0]
    toothDat["yDistCent"] = toothDat["y"] - centerDat["y"].iloc[0]
    toothDat["zDistCent"] = toothDat["z"] - centerDat["z"].iloc[0]
    
    #distance calculated as l2 norm
    toothDat["l2Norm"] = np.linalg.norm(
        toothDat[["xDistCent", "yDistCent", "zDistCent"]],
        axis = 1,
        ord = 2
        ) 
    
    return toothDat

#example
# import sys
# sys.path.append("tools")
# import readAndFormat as raf
# import toothCentroids as toCe
# prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
# datPre = raf.readAndFormat(file = prePath, arch = "U")
# tcPre = toCe.toothCentroids(face = datPre["face"], vertex = datPre["vert"])
# teethToCenterDist(tcPre)
# teethToCenterDist(tcPre, center = "gum")
