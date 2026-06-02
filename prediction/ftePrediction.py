import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/prediction/")
import fastTgcnEasyPredictFun as ftep


predNote = "using t3ds model that was trained on decimated data to predict on remeshed data"
print(predNote)


ftep.fastTgcnEasyPredict(inDir = "/Shared/gb_lss/Thomas/iowaExpansion/segModelReadyScans/post",
                         outDir = "/Shared/gb_lss/Thomas/iowaExpansion/remeshSegResults_t3dsDec016Epoch140/post",
                         modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/trainedModels/2026_04_21 fullT3dsDec016Epoch140.pth"
                         )



print(predNote)
