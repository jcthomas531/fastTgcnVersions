import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
import plyFunctions as pf
import os

modOutDir = "Y:/dissModels/fastTgcnVersions/fastTgcnEasy/ModelOutputs"
os.chdir(modOutDir + "/2026_01_27 full upper/pred_global")



allFiles = os.listdir()

for i in allFiles:
    pf.readAndPlot(i, arch = "U")
