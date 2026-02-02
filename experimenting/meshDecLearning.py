



###############################################################################
#tutorial from pyvista
###############################################################################
# https://docs.pyvista.org/examples/01-filter/decimate.html
import numpy as np
import pyvista as pv
from pyvista import examples

mesh1 = examples.download_face()
#Define a camera position that shows this mesh properly
#i dont believe i will need this with my process
cpos = [(0.4, -0.07, -0.31), (0.05, -0.13, -0.06), (-0.1, 1, 0.08)]
dargs = dict(show_edges=True, color=True)
#Preview the mesh
mesh1.plot(cpos=cpos, **dargs)

#set decimation porportion
target_reduction = 0.7

#perform decimation
decimated1 = mesh1.decimate(target_reduction)
#view
decimated1.plot(cpos=cpos, **dargs)

#now using "pro decimation", just another algorothm
decimated2 = mesh1.decimate_pro(target_reduction, preserve_topology=True)
#view
decimated2.plot(cpos=cpos, **dargs)


###############################################################################
#testing with an iosSeg file
###############################################################################
#gonna start with a mesh from iosSeg bc i know it can fit in memory
import os
#setting this up to run locally, these file paths can get annoying
#need to find the equivalent of the here package
os.chdir("K:/IOSSegData/clean/testClean")

#read in file and plot
#colors not important at this step
l001 = pv.read("001_L.ply")
l001.n_faces_strict
l001.plot(show_edges=True)

#decimate and plot
l001Dec = l001.decimate(.25)
l001Dec.n_faces_strict
l001Dec.plot(show_edges=True)

#using decimate_pro
l001DecP = l001.decimate_pro(.25, preserve_topology=True)
l001DecP.n_faces_strict
l001DecP.plot(show_edges = True)

#show all together
pl = pv.Plotter(shape=(1, 3))
pl.add_mesh(l001, **dargs)
pl.subplot(0, 1)
pl.add_mesh(l001Dec, **dargs)
pl.subplot(0, 2)
pl.add_mesh(l001DecP, **dargs)
pl.link_views()
pl.show()


###############################################################################
#trying process on a RME file
###############################################################################
os.chdir("K:/iowaRme")
rmeTest = pv.read("convertedPlyTestU.ply")
origFaceN = rmeTest.n_faces_strict
rmeTest.plot(show_edges = True)

#want to reduce to about 16000 faces so it is like the iosSeg files
reduct = 1-(16000/origFaceN)

#decimation
rmeTestDec = rmeTest.decimate(reduct)
decFaceN = rmeTestDec.n_faces_strict
rmeTestDec.plot(show_edges = True)

#decimation_pro
rmeTestDecP = rmeTest.decimate_pro(reduct, preserve_topology=True)
decPFaceN = rmeTestDecP.n_faces_strict
rmeTestDecP.plot(show_edges = True)

#viewing
testPlot = pv.Plotter(shape=(1, 3))
testPlot.add_mesh(rmeTest, **dargs)
testPlot.subplot(0, 1)
testPlot.add_mesh(rmeTestDec, **dargs)
testPlot.subplot(0, 2)
testPlot.add_mesh(rmeTestDecP, **dargs)
testPlot.link_views()
testPlot.show()

#outputting
rmeTestDec.save("convertedPlyTestUDec.ply", binary = False)



