
#
runNote = "warmstartTest3, half and half base and no base teeth3ds scans, oriented like iosseg, 100 epochs, upper arch"
#


print(runNote)
#should alredy be in the proper working directory
import train
train.fastTgcnEasy(arch = "u",
                   testPath = "/Shared/gb_lss/Thomas/testDir/warmstartTestDataReg/test",
                   trainPath = "/Shared/gb_lss/Thomas/testDir/warmstartTestDataReg/train",
                   batch_size = 1,
                   k = 32,
                   numWorkers = 8,
                   epochs = 100)
print(runNote)


#import os
#print(os.getcwd())





