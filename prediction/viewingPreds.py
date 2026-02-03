import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
import plyFunctions as pf
import os




#RME prediction
os.chdir("K:/iowaRme/test2Pred/")
pf.readAndPlot(file = "convFunTestDec.ply", 
               arch = "U")


#iosSeg prediction
os.chdir("K:/iowaRme/test1Pred/")
pf.readAndPlot(file = "001_U.ply", 
               arch = "U")



#prediction used in modeling process, iosSeg
os.chdir("Y:/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/pred_global")
pf.readAndPlot(file = "001_U.ply", 
               arch = "U")


import pyvista as pv
dargs = dict(show_edges=False, rgb=True)
m1 = pv.read("K:/iowaRme/test2Pred/convFunTestDec.ply")
m2 = pv.read("K:/iowaRme/test1Pred/001_U.ply")
m3 = pv.read("Y:/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_01_27 full upper/pred_global/001_U.ply")

pl = pv.Plotter(shape=(1, 3))
pl.add_mesh(m1, **dargs)
pl.subplot(0, 1)
pl.add_mesh(m2, **dargs)
pl.subplot(0, 2)
pl.add_mesh(m3, **dargs)
pl.link_views()
pl.show()