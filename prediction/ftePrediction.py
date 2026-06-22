import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/prediction/")
import fastTgcnEasyPredictFun as ftep


predNote = "iowaRme fin scans, using t3ds model trained on remeshed data, remeshT3dsEpoch270"
print(predNote)


ftep.fastTgcnEasyPredict(inDir = "/Shared/gb_lss/Thomas/iowaRme/preDelivAndFinalScans/finalScanU/segReadyScans",
                         outDir = "/Shared/gb_lss/Thomas/iowaRme/segResults/segResults_remeshT3dsEpoch270/fin",
                         modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/trainedModels/2026_16_12 remeshT3dsEpoch270.pth"
                         )



print(predNote)