import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import plyFunctions as pf
import os



#RME prediction

os.chdir("K:/iowaRme/testDir/test1")
pf.readAndPlot(file = "pat058u_01CONV2.ply", 
               arch = "U")

os.chdir("K:/iowaRme/testDir/test1PredD/")
pf.readAndPlot(file = "pat058u_01CONV2.ply", 
               arch = "U")

os.chdir("Y:/dissModels/intraoralSegmentation/fastTgcnEasy/modelOutputs/2026_04_21 teeth3dsFullTrainDecim016/pred_global")
pf.readAndPlot(file = "pat058u_01CONV2.ply", 
               arch = "U")







os.chdir("K:/trainTestSets/teeth3dsDecim016/test")
pf.readAndPlot(file = "0JN50XQR_UDecim016.ply", 
               arch = "U")




import pyvista as pv
dargs = dict(show_edges=False, rgb=True)
m1 = pv.read("K:/iowaRme/test1Pred/decimIowa.ply")
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