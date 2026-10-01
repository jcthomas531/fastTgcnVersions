import sys
from pathlib import Path

import pandas as pd

#testing
outPath = "K:/iowaExpTest/testDir/testCombined.csv"
inFiles = [
    "K:/iowaExpTest/rugaePipeOptim/labeledDescriptors/post/decim8500/norm0.02_desc1.1/pat001Post.csv",
    "K:/iowaExpTest/rugaePipeOptim/labeledDescriptors/post/decim8500/norm0.02_desc1.1/pat004Post.csv"
]

#variables from snakemake
outPath = sys.argv[1]
inFiles = sys.argv[2:]


#create a to put all of the data frames in to
allDat = []
#read in files and attatch patient number 
for i in inFiles:
    #read in data
    dati = pd.read_csv(i)
    #add patient id
    dati["patId"] = Path(i).stem
    #add to data frame list
    allDat.append(dati)


#stack data frames
stackedDat = pd.concat(allDat, ignore_index=True)
#export
stackedDat.to_csv(outPath, index = False)
