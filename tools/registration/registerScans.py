import sys
sys.path.append("Y:/dissModels/fastTgcnVersions/tools/registration")
import registrationFuns as rg
import os
import open3d as o3d
import copy
import trimesh
import numpy as np

import random
random.seed(826)


#target file
iossegPath = "K:/IOSSegData/clean/trainCleanU/007_U.ply"
targetCloud = o3d.io.read_point_cloud(iossegPath)


#loop for train files
outDirTrain = "K:/testDir/warmstartTestDataReg/train/"
inDirTrain = "K:/testDir/warmstartTestData/train/"
trainToReg = os.listdir(inDirTrain)
for i in range(len(trainToReg)):
    #obtaining transformation
    sourceCloudi = o3d.io.read_point_cloud(inDirTrain + trainToReg[i])
    regi = rg.getRegistration(source = sourceCloudi, target=targetCloud)
    #register the meshes and export
    rg.registerAndExport(inFile = inDirTrain + trainToReg[i],
                         outFile = outDirTrain + trainToReg[i],
                         trans = regi.transformation) 
    print("Done: " + str(i))



#loop for test files
outDirTest = "K:/testDir/warmstartTestDataReg/test/"
inDirTest = "K:/testDir/warmstartTestData/test/"
testToReg = os.listdir(inDirTest)
for i in range(len(testToReg)):
    #obtaining transformation
    sourceCloudi = o3d.io.read_point_cloud(inDirTest + testToReg[i])
    regi = rg.getRegistration(source = sourceCloudi, target=targetCloud)
    #register the meshes and export
    rg.registerAndExport(inFile = inDirTest + testToReg[i],
                         outFile = outDirTest + testToReg[i],
                         trans = regi.transformation) 
    print("Done: " + str(i))










