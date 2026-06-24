import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/prediction/")
import fastTgcnEasyPredictFun as ftep


predNote = "iowaExpansion post scans, using interim csRot model "
print(predNote)


ftep.fastTgcnEasyPredict(inDir = "/Shared/gb_lss/Thomas/iowaExpansion/segReadyScans2/post",
                         outDir = "/Shared/gb_lss/Thomas/iowaExpansion/segResults/TEMP/post",
                         modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/trainedModels/TEMP.pth"
                         )



print(predNote)