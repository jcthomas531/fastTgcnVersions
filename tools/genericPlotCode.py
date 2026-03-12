import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
import plyFunctions as pf
import os



dir1 = "K:/teeth3DS/scanData/upper/"
pf.readAndPlot(dir1 + "6X24ILNE/6X24ILNE_U.ply", arch = "U")
pf.readAndPlot(dir1 + "6X24ILNE/6X24ILNE_UDecim016.ply", arch = "U")

dir2 = "K:/IOSSegData/clean/testCleanU/"
pf.readAndPlot(dir2 + "100_U.ply", arch = "U")
