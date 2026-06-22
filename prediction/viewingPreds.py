import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import plyFunctions as pf
import os





dir1 = "K:/iowaExpansion/segResults/segResults_remeshT3dsEpoch270/post/"

os.chdir(dir1)
segFiles = os.listdir()
for i in range(len(segFiles)):
    filei = segFiles[i]
    if filei.endswith("seg.ply"):
        pf.readAndPlot(file = filei, 
                       arch = "U")



os.chdir("K:/iowaRme/testDir/segTestInput")
pf.readAndPlot(file = "pat058u_01CONV2.ply", 
               arch = "U")

os.chdir("K:/iowaRme/testDir/segTestOutput/")
pf.readAndPlot(file = "pat055u_01_dec016Form.ply", 
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