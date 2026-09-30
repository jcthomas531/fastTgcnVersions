import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.append("tools")
import readAndFormat as raf
from faceToPointLabel_2Color import faceToPointLabel_2Color

#for testing
meshPath = "K:/iowaExpTest/scanData/rugAnnotForm_cSOriMastRemesh/pre/pat001Pre_formCSOriMastRemesh.ply"
ldPath = "K:/iowaExpTest/localDescriptors/rugAnnotForm_cSOriMastRemesh_localDescr/pre/pat001Pre_localDescr.csv"
outPath = "K:/iowaExpTest/testDir/test.csv"

#arguements from snakemake
meshPath = sys.argv[1] 
ldPath = sys.argv[2]
outPath = sys.argv[3]


#change face labels to point labels
meshDat = faceToPointLabel_2Color(inFile = meshPath, labelColor='255-000-127', nonlabelColor='255-255-255')
vDat = meshDat["vert"]
fDat = meshDat["face"]

#remove normals in favor of other normals which will be joined
vDat = vDat.drop(["nx", "ny", "nz"], axis = 1)

#read in local descriptors
ld = pd.read_csv(ldPath)
#switch data type to accomidate merge
ld[["x", "y", "z"]] = ld[["x", "y", "z"]].astype(np.float32)

#merge with labeled vertex data
vld = pd.merge(left=vDat, right = ld, how = "left", on = ["x", "y", "z"])

#export
vld.to_csv(outPath, index = False)