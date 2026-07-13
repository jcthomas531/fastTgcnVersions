import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/prediction/")
import fastTgcnEasyPredictFun as ftep


predNote = "iowaExpansion origForm_cSOriMastRemesh post scans, using model t3dsIosseg_cSOriMastEpoch300"
print(predNote)


ftep.fastTgcnEasyPredict(inDir = "/Shared/gb_lss/Thomas/iowaExpTest/scanData/origForm_cSOriMastRemesh/post",
                         outDir = "/Shared/gb_lss/Thomas/iowaExpTest/segResults/segResults_t3dsIosseg_cSOriMastEpoch300/origForm_cSOriMastRemesh/post",
                         modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/trainedModels/2026_07_09 t3dsIosseg_cSOriMastEpoch300.pth"
                         )



print(predNote)