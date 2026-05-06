import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import registrationFuns as regi
# import decimationFuns as d
# import formatAndExportFuns as fe
import os


#DOING FULL REGISTRATION IN ORDER TO ORIENT THE RME DATA IN THE SAME DIRECTION
#AS THE TEETH3DS TRAINING DATA IS PROBABLY OVERKILL BUT IT IS THE TOOLS I HAVE
#BUILT ALREADY
teeth3dsTarget = "K:/trainTestSets/teeth3dsDecim016/train/00OMSZGW_UDecim016.ply"


#for preD files
preDInDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016Scans/"
preDInNames = os.listdir(preDInDir)
preDOutDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016OriScans/"
newPreDNames = [i[:-4] + "Ori.ply" if i.endswith(".ply") else i
               for i in preDInNames]

#loop thru all preD and orient them, separate loops for preD and final bc not the same number of obs
for i in range(len(preDInNames)):
    inPath = preDInDir + preDInNames[i]
    outPath = preDOutDir + newPreDNames[i]
    regi.fullRegistFlow(targetFile=teeth3dsTarget, sourceFile=inPath, registerFile=outPath)
    print(preDInNames[i] + " done")


#for final scans
finInDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016Scans/"
finInNames = os.listdir(finInDir)
finOutDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriScans/"
newFinNames = [i[:-4] + "Ori.ply" if i.endswith(".ply") else i
               for i in finInNames]

#loop thru all fin and orient them, separate loops for preD and final bc not the same number of obs
for i in range(len(finInNames)):
    inPath = finInDir + finInNames[i]
    outPath = finOutDir + newFinNames[i]
    regi.fullRegistFlow(targetFile=teeth3dsTarget, sourceFile=inPath, registerFile=outPath)
    print(finInNames[i] + " done")