import pickle
import random
import os
import numpy as np
import open3d as o3d
import sys
sys.path.append("tools")
import getRotToMaster as grtm

#testing
# inPath = "K:/teeth3DS/scanData/upperPly_cS/BIIEY91S_cS_U.ply"
# outPath = "K:/testDir/rotTest.pkl"

#pull variables from snakemake
inPath = sys.argv[1]
outPath = sys.argv[2]
masterArch = sys.argv[3]


#set seed/randomness
os.environ["OMP_NUM_THREADS"] = "1"
seed = 826
random.seed(seed)
np.random.seed(seed)
o3d.utility.random.seed(seed)

#obain rotation matrix for orientation to master arch
mat = grtm.getRotToMaster(filePath = inPath, masterArchPath = masterArch)

#export rotation matrix
filePath = open(outPath, "wb")
pickle.dump(obj = mat,
            file = filePath)
filePath.close()
#can be read in like: 
# with open(outPath, "rb") as f:
#     obj = pickle.load(f)