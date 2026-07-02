import os
import sys
from pathlib import Path
import numpy as np
import trimesh
import pickle
import random
import open3d as o3d
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import getRotToMaster as grtm

#pull variables from snakemake
dirPath = sys.argv[1]
outDir = sys.argv[2]

#testing
dirPath = "K:/teeth3DS/scanData/upperPly/"
# outDir = "K:/teeth3DS/randomRotations/"

#set seed/randomness
os.environ["OMP_NUM_THREADS"] = "1"
seed = 826
random.seed(seed)
np.random.seed(seed)
o3d.utility.random.seed(seed)

#file names
files = os.listdir(dirPath)
n = len(files)
names = [Path(i).stem for i in files]

#produce random rotation matrices
matrices = trimesh.transformations.random_rotation_matrix(num = n)

#get rotation matrices to master arch
#NEED CENTER AND SCALING
matDict = {}
for i in range(len(files)):
    matDict[names[i]] = grtm.getRotToMaster(dirPath + files[i])

#write out random matrices
for i in range(len(matrices)):
    namei = names[i]
    filePath = open(outDir + namei + "_rot" + ".pkl", "wb")
    pickle.dump(obj = matrices[i],
                file = filePath)
    filePath.close()
    print(namei + " random rotation matrix produced")


