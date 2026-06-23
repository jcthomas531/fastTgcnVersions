import numpy as np
import os
import shutil

#set seed
seed = 826
np.random.seed(seed)

#directory for new test and train set
newDir = "K:/trainTestSets/teeth3dsIosseg_cSRot/"


###############################################################################
#teeth3ds
#files in teeth3ds remesh center, scale, random rotate directory
t3dsDir = "K:/teeth3DS/scanData/upperPlyRemeshCSRot/"
t3dsFiles = os.listdir(t3dsDir)

#generate random order of files
t3dsAssignOrder = np.argsort(np.random.rand(900))

#group assignments
#test (180 files)
t3dsTestIdx =  t3dsAssignOrder[:180]
t3dsTestFiles = [t3dsFiles[i] for i in t3dsTestIdx]
t3dsOldTestPath = [t3dsDir + i for i in t3dsTestFiles]
t3dsNewTestPath = [newDir + "test/" + i for i in t3dsTestFiles]
t3dsTestFilePairs = zip(t3dsOldTestPath, t3dsNewTestPath)
#train (720 files)
t3dsTrainIdx = t3dsAssignOrder[180:]
t3dsTrainFiles = [t3dsFiles[i] for i in t3dsTrainIdx]
t3dsOldTrainPath = [t3dsDir + i for i in t3dsTrainFiles]
t3dsNewTrainPath = [newDir + "train/" + i for i in t3dsTrainFiles]
t3dsTrainFilePairs = zip(t3dsOldTrainPath, t3dsNewTrainPath)

#move t3ds files
#test data
for source, destination in t3dsTestFilePairs:
    shutil.copy2(source, destination)
#train data
for source, destination in t3dsTrainFilePairs:
    shutil.copy2(source, destination)
    
    

###############################################################################
#iosseg
iosDir = "K:/IOSSegData/cleanCSRot/upper/"
iosFiles = os.listdir(iosDir)

#generate random order of files
iosAssignOrder = np.argsort(np.random.rand(89))

#group assignements
#test (17 files)
iosTestIdx = iosAssignOrder[:17]
iosTestFiles = [iosFiles[i] for i in iosTestIdx]
iosOldTestPath = [iosDir + i for i in iosTestFiles]
iosNewTestPath = [newDir + "test/" + i for i in iosTestFiles]
iosTestFilePairs = zip(iosOldTestPath, iosNewTestPath)
#train (72 files)
iosTrainIdx = iosAssignOrder[17:]
iosTrainFiles = [iosFiles[i] for i in iosTrainIdx]
iosOldTrainPath = [iosDir + i for i in iosTrainFiles]
iosNewTrainPath = [newDir + "train/" + i for i in iosTrainFiles]
iosTrainFilePairs = zip(iosOldTrainPath, iosNewTrainPath)

#move ios files
#test data
for source, destination in iosTestFilePairs:
    shutil.copy2(source, destination)
#train data
for source, destination in iosTrainFilePairs:
    shutil.copy2(source, destination)