import pandas as pd
import numpy as np
import os
import re
import sys
import pickle

sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import readAndFormat as raf
import toothCentroids as toCe
import teethToCenterDist as ttcd
import centroidSize as ceSi



#testing
#dir_ = "K:/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/pre/"
#outPath = "K:/iowaExpTest/testDir/testPickle.pkl"

#take values from snakemake
dir_ = sys.argv[1] + "/" #bc of the way snakemake directory() function is set up this is necessary to add on the end
outPath = sys.argv[2]

#create dictionary of patient names and files
files = os.listdir(dir_)
pats = [re.search(pattern = r"^pat[0-9]{3}", string = i).group() for i in files]
pathDict = dict(zip(pats, files))

#prepare centroid size data frame
centSize = pd.DataFrame(np.nan, index=range(len(files)),
                          columns=["patNum", "centSize"])
centSize["patNum"] = list(pathDict.keys())

#loop thru files and calculate centriod size
for i in list(centSize["patNum"]):
    
    #read in data
    pathi = dir_ + pathDict[i]
    dati = raf.readAndFormat(file = pathi, arch = "U") 
    
    #get centroids
    tci = toCe.toothCentroids(face = dati["face"], vertex = dati["vert"])
    
    #get distance to center
    disti = ttcd.teethToCenterDist(tci)
    
    #calculate centroid size
    csi = ceSi.centriodSize(x = disti)
    
    #calculate centroid size
    centSize.loc[centSize["patNum"] == i, "centSize"] = csi 

#export
centSize.to_csv(outPath)


