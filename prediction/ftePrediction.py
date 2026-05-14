import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/prediction/")
import fastTgcnEasyPredictFun as ftep


ftep.fastTgcnEasyPredict(inDir = "/Shared/gb_lss/Thomas/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriScans",
                         outDir = "/Shared/gb_lss/Thomas/iowaRme/preDelivAndFinalScans/finalScanU/dec016OriSeg",
                         modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/trainedModels/2026_04_21 fullT3dsDec016Epoch140.pth"
                         )
