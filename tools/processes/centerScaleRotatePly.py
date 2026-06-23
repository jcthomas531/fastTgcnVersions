from pathlib import Path
import trimesh
import os
import pickle
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimeshExtractFaceLabels as tefl
import numpy as np
import trimeshToDf_labels as ttdl
import dfToPlyExport as dtpe

#pull variables from snakemake
inPath = sys.argv[1]
rotDir = sys.argv[2]
fileSuffix = sys.argv[3]
outPath = sys.argv[4]

#testing
# inPath = "K:/IOSSeg/clean/allCleanU/002_U.ply"
# rotDir = "K:/IOSSeg/randomRotations/"
# fileSuffix = ".ply"
# outPath = "K:/aaa.ply"




#extract name
name = Path(inPath).name.replace(fileSuffix, "")

#find random roation associated with this mesh
allRots = os.listdir(rotDir)
rotFile = [i for i in allRots if Path(i).stem.startswith(name)]
#check to ensure just one match
if len(rotFile) != 1:
    raise ValueError("Multiple name matches in rotation directory for " + name +", quitting")
#path for random rotation
rotPath = rotDir + rotFile[0]

#load in roation
with open(rotPath, "rb") as i:
    rotMat = pickle.load(i)


#load in mesh
mesh = trimesh.load(inPath, process = False)

#extract face color information from trimesh
colorDf = tefl.trimeshExtractFaceLabels(mesh)

#center mesh
mesh.apply_translation(-mesh.centroid)
#obtain scaling factor
scaleFac = 1/np.max(mesh.extents)
#scale mesh
mesh.apply_scale(scaleFac)
#apply random rotation
mesh.apply_transform(rotMat)

#export
vertDf, faceDf = ttdl.trimeshToDf_labels(mesh, colorDf = colorDf)
dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = outPath)







