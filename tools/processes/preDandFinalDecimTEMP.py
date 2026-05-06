import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import decimationFuns as d
import formatAndExportFuns as fe
import os

#SET SEED NEXT TIME

#set up function for full workflow
def decimFlow(inFile, outFile, nFace = 16000):
    #perform decimation 
    mDec = d.decim(inFile=inFile, nFace=nFace)
    #tranform to data frames
    mVert, mFace = fe.trimeshToDfNoLabels(mDec)
    #export
    fe.dfToPlyExport(vertDf = mVert,
                     faceDf = mFace,
                     outFile = outFile)

#directories and naming for final scans
preDFullDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/fullScans/"
allPreDNames = os.listdir(preDFullDir)
preDDecDir = "K:/iowaRme/preDelivAndFinalScans/preDelivScanU/dec016Scans/"
newPreDNames = [i[:-4] + "_dec016.ply" if i.endswith(".ply") else i
               for i in allPreDNames]

#loop thru, separate loops for preD and final bc not the same number of obs
for i in range(len(allPreDNames)):
    inPath = preDFullDir + allPreDNames[i]
    outPath = preDDecDir + newPreDNames[i]
    decimFlow(inFile = inPath, outFile = outPath, nFace = 16000)
    print(allPreDNames[i] + " done")



#directories and naming for pre delivery scans
finalFullDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/fullScans/"
allFinalNames = os.listdir(finalFullDir)
finalDecDir = "K:/iowaRme/preDelivAndFinalScans/finalScanU/dec016Scans/"
newFinalNames = [i[:-4] + "_dec016.ply" if i.endswith(".ply") else i
               for i in allFinalNames]

#loop thru, separate loops for preD and final bc not the same number of obs
for i in range(len(allFinalNames)):
    inPath = finalFullDir + allFinalNames[i]
    outPath = finalDecDir + newFinalNames[i]
    decimFlow(inFile = inPath, outFile = outPath, nFace = 16000)
    print(allFinalNames[i] + " done")


