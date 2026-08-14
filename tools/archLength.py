import pandas as pd
import centToCentDist as ctcd

#x is output from toothCentroids()
def archLength(x):
    
    xC = x.copy()
    teethNums = list(map(str, list(range(1,17))))
    
    #take only the teeth centroids
    xC = xC[xC["toothNum"].isin(teethNums)]
    #make the teeth numbers numeric
    xC["toothNum"] = pd.to_numeric(xC["toothNum"])
    #sort teeth in numerical order
    xC = xC.sort_values("toothNum").reset_index(drop = True)
    
    #calculate distances
    distList = []
    for i in range(0,len(xC)):
        #distance between each successive tooth, in numerical order
        #this will skip missing teeth
        #at the largest value, it will calculate the distance across the pallate
        
        #current tooth
        toothCurrent = xC["toothNum"].iloc[i]
        
        #check if this is the larges tooth number
        maxTooth = max(xC["toothNum"])
        
        #determine tooth to measure to
        if toothCurrent != maxTooth:
            toothNext = xC["toothNum"].iloc[i+1]
        else:
            toothNext = xC["toothNum"].iloc[0]
        
        #store distances
        distList.append(
            ctcd.centToCentDist(xC, tooth1 = toothCurrent, tooth2 = toothNext)
            )
    
    #arch length
    archLength = sum(distList[:-1])
    #arch length and across the back
    fullPerimeter = sum(distList)
    
    return {"archLength": archLength, "fullPerimeter": fullPerimeter}

#archLength(x = tci)

