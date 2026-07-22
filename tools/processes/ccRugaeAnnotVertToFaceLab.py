import trimesh
import numpy as np
import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import trimeshToDfNoLabels as ttdnl
import dfToPlyExport as dtpe

#testing
# inPath = "K:/iowaExpTest/scanData/rugaeAnnot/pre/pat001Pre_annot.ply"
# outPath = "K:/iowaExpTest/testDir/testLabs.ply"

#bring in snakemake variables
inPath = sys.argv[1]
outPath = sys.argv[2]


#load in mesh
mesh = trimesh.load(inPath, process = False)
#extract vertex labels
vertLabs = mesh.metadata["_ply_raw"]["vertex"]["data"]["scalar_Classification"]

#loop through each face and assign face label based on if any vertex is classified as 1 (rugae)
#create empty list for face labels
faceLabs = []
for i in mesh.faces:
    labs = vertLabs[i]
    if np.any(labs == 1):
        faceLabs.append(1)
    else:
        faceLabs.append(0)
#make face labels in an array
faceLabs = np.array(faceLabs)

#format mesh into data frames as if it had no labels
vDat, fDat = ttdnl.trimeshToDfNoLabels(mesh)
#change color for rugae labeled faces to black
fDat.loc[faceLabs == 1, ["red", "green", "blue"]] = [255,0,127]
#export
dtpe.dfToPlyExport(vertDf = vDat, faceDf = fDat, outFile = outPath)

