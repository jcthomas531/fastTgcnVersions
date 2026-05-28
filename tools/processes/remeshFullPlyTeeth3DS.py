import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import pyvista as pv
import pyacvd  
import trimesh
import trimeshToDfNoLabels as tdnl
import dfToPlyExport as dpe
import random
import numpy as np

#seed setting
import os
os.environ["OMP_NUM_THREADS"] = "1"
seed = 826
random.seed(seed)
np.random.seed(seed)

#pull variables from snakemake
inFile = sys.argv[1]
outFile = sys.argv[2]

#read in file 
meshOrig = pv.read(inFile)
print("file read in")

#remesh
#this decimates and gives an isotropic remesh
#this will give approx 8500*2 faces, not excatly sure how it work but the subdivide value doesnt matter for resulting faces
meshIso = meshOrig.acvd.remesh(8500, subdivide=3)
print("decimated and remeshed")

#make into trimesh
meshIsoTri = pv.to_trimesh(meshIso)

#format and export
isoVert, isoFace = tdnl.trimeshToDfNoLabels(meshIsoTri)
print("formated")
dpe.dfToPlyExport(vertDf = isoVert, faceDf = isoFace, outFile = outFile)
print("exported")