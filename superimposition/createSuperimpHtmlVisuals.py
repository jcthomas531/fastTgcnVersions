import pyvista as pv
import sys


#pull variables from snakemake
prePath = sys.argv[1]
postPath = sys.argv[2]
color_ = sys.argv[3]
visHtml = sys.argv[4]


#testing
# patNum = "015"
# prePath = "K:/iowaExpansion/fullRugaeAnnotScans/pre/pat" + patNum + "Pre_annot.ply"
# postPath = "K:/iowaExpansion/superimposition/transPostScan/annotRugaeTransPostScan/pat" + patNum + "Post_annotRugaeSuperimp.ply"
# color_ = "green"
# visHtml = "K:/iowaExpansion/superimposition/visuals/annotRugaeSuperimp/testRegist.html"

#read in meshes
preMesh = pv.read(prePath)
postMesh = pv.read(postPath)

#create plotting environment
superimpPlot = pv.Plotter()
#add meshes
superimpPlot.add_mesh(preMesh, color = "white")
superimpPlot.add_mesh(postMesh, color=color_, opacity = .6)
#superimpPlot.show() #do not run show() this prior to using export_html()
#export
superimpPlot.export_html(visHtml)
print("superimposition html exported")









