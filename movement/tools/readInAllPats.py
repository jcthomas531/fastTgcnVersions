from pathlib import Path
import pickle

def readInAllPats(pklPaths):
    #copy input so the caller's list is not modified
    pklPathsC = list(pklPaths)

    #patient number is the name of the folder containing the pkl
    patNums = [Path(i).parent.name for i in pklPathsC]

    #load in all files for all patients
    allPatRes = {}
    for i in range(len(pklPathsC)):
        pati = patNums[i]
        pathi = pklPathsC[i]
        with open(pathi, "rb") as f:
            allPatRes[pati] = pickle.load(f)
    #object structure
    #allPatRes.keys()
    #allPatRes["pat004"].keys()
    #allPatRes["pat004"]["mainResults"]["3"].keys()
    
    return allPatRes, patNums