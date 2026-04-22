import sys
sys.path.append("/Users/jthomas48/dissModels/intraoralSegmentation/prediction/")
import fastTgcnEasyPredictFun as ftep



ftep.fastTgcnEasyPredict(inDir = "/Shared/gb_lss/Thomas/iowaRme/testDir/segTestInput",
                         outDir = "/Shared/gb_lss/Thomas/iowaRme/testDir/segTestOutput",
                         modelPath = "/Users/jthomas48/dissModels/intraoralSegmentation/fastTgcnEasy/modelOutputs/2026_04_21 fullT3dsDec016/checkpointsAndLogs/checkpoints/coordinate_140_0.939648.pth"
                         )