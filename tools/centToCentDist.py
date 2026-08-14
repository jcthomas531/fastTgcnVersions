import numpy as np
import pandas as pd

#x is output from toothCentroids()
#tooth1 and tooth2 are tooth numbers from the toothNum column, as strings
def centToCentDist(x, tooth1, tooth2):
    
    #make sure the teeth are in the centroid data 
    if ((tooth1 in x["toothNum"].values) & (tooth2 in x["toothNum"].values)):
        #extract teeth
        t1 = x[x["toothNum"] == tooth1]
        t2 = x[x["toothNum"] == tooth2]
        
        #find difference
        centDiff = t1[["x", "y", "z"]].iloc[0] - t2[["x", "y", "z"]].iloc[0]
        
        #take l2 norm
        dist = np.linalg.norm(centDiff, axis = 0, ord = 2)
    else:
        #if one or both of the teeth non in the cetroid data
        dist = np.nan
    
    
    return dist

#centToCentDist(tci, tooth1 = "3", tooth2 = "20")
