import os
import sys
from pathlib import Path
import numpy as np
import trimesh
import pickle

#pull variables from snakemake
dirPath = sys.argv[1]
outDir = sys.argv[2]

#testing
# dirPath = "K:/teeth3DS/scanData/upperPly/"
# outDir = "K:/teeth3DS/randomRotations/"

#set seed
seed = 826
np.random.seed(seed)

#file names
files = os.listdir(dirPath)
n = len(files)
names = [Path(i).stem for i in files]

#produce random rotation matrices
matrices = trimesh.transformations.random_rotation_matrix(num = n)


#write out random matrices
for i in range(len(matrices)):
    namei = names[i]
    filePath = open(outDir + namei + "_rot" + ".pkl", "wb")
    pickle.dump(obj = matrices[i],
                file = filePath)
    filePath.close()
    print(namei + " random rotation matrix produced")

