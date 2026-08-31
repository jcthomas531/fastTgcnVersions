import trimesh
import numpy as np
import sys
sys.path.append("tools")
import trimeshToDfNoLabels as ttdnl
import dfToPlyExport as dtpe

#testing
# inPath = "K:/iowaExpTest/scanData/rugAnnot/pre/pat001Pre_annot.ply"
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
    #extract the point labels for a face
    labs = vertLabs[i]
    #how many of the points are labeled as rugae
    countRugLabeled = np.sum(labs == 1)
    #impliment majority rules to map vertex labels to face labels
    if countRugLabeled >= 2:
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

