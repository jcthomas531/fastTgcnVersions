import trimesh
import sys
sys.path.append("tools")
import trimeshToDfNoLabels as ttdnl
import dfToPlyExport as dtpe

#testing
# inPath = "K:/iowaExpTest/scanData/orig/post/pat001Post.ply"
# outPath = "K:/iowaExpTest/testDir/outMesh2.ply"

#pull variables from snakemake
inPath = sys.argv[1]
outPath = sys.argv[2]

#load mesh
#skip materials ignores texture information that is contained in other file
mesh = trimesh.load_mesh(inPath, process = False, skip_materials = True)

#convert to dfs
vertDf, faceDf = ttdnl.trimeshToDfNoLabels(mesh)

#export in desired format
dtpe.dfToPlyExport(vertDf = vertDf, faceDf = faceDf, outFile = outPath)