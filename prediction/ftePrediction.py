import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/prediction/")
import fastTgcnEasyPredictFun as ftep


predNote = "iowaExp pre scans, using t3ds model trained on remeshed data, some pred scans not corrently oriented"
print(predNote)


ftep.fastTgcnEasyPredict(inDir = "/Shared/gb_lss/Thomas/iowaExpansion/segModelReadyScans/pre",
                         outDir = "/Shared/gb_lss/Thomas/iowaExpansion/segResults/segResults_remeshT3dsEpoch270/pre",
                         modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/trainedModels/2026_16_12 remeshT3dsEpoch270.pth"
                         )



print(predNote)
