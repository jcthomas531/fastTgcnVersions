import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools")
import plyFunctions as pf
import os



dir1 = "Y:/dissModels/fastTgcnVersions/fastTgcnEasy/modelOutputs/2026_03_12 teeth3dsPartialTrain/pred_global"
os.chdir(dir1)
pf.readAndPlot("00OMSZGW_UDecim016.ply", "U")


aaa = os.listdir()
for i in range(len(aaa)):
    pf.readAndPlot(aaa[i], "U")
    
dir2 = "K:/testDir/warmStartTestData/train" 
os.chdir(dir2)
aaa = os.listdir()
for i in range(len(aaa)):
    pf.readAndPlot(aaa[i], "U")

pf.readAndPlot("01A91JH6_UDecim016.ply", arch = "U")
pf.readAndPlot(dir1 + "6X24ILNE/6X24ILNE_UDecim016.ply", arch = "U")

dir2 = "K:/IOSSegData/clean/testCleanU/"
pf.readAndPlot(dir2 + "100_U.ply", arch = "U")
