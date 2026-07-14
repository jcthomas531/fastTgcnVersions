
import numpy as np

#function to calculate centroid size of all teeth on arch or specific teeth
#regardless of the teeth selected, the centroid will always be the centroid for 
#overall arch not just these specific teeth


#x is the output from teethToCenterDist()
#teeth is either "all" or a list of numeric teeth values ie [5,6]

def centriodSize(x, teeth = "all"):
    #create list of all teeth
    if teeth == "all":
        teeth = list(range(1,17))

    #convert tooth numbers to string like in centroid data frame
    teeth = list(map(str, teeth))
    
    #subset data to only teeth of interest
    toothDat = x.loc[x["toothNum"].isin(teeth)].copy()
    
    #calculate centroid size, just l2 norm of the l2 norms
    centSize = np.linalg.norm(toothDat["l2Norm"], ord = 2)
    
    return centSize

#example
# import sys
# sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
# import readAndFormat as raf
# import toothCentroids as toCe
# import teethToCenterDist as ttcd
# prePath = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh_seg.ply"
# datPre = raf.readAndFormat(file = prePath, arch = "U")
# tcPre = toCe.toothCentroids(face = datPre["face"], vertex = datPre["vert"])
# distPre = ttcd.teethToCenterDist(tcPre)
# centriodSize(distPre)
# centriodSize(distPre, teeth = [1, 2, 4, 14, 15, 16])
