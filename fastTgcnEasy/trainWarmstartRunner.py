
#
runNote = "first warm start attempt, using 30 teeth3ds to train and 10 to test, 100 epochs, upper arch"
#


print(runNote)
#should alredy be in the proper working directory
import train
train.fastTgcnEasy(arch = "u",
                   testPath = "/Shared/gb_lss/Thomas/testDir/warmstartTestData/test",
                   trainPath = "/Shared/gb_lss/Thomas/testDir/warmstartTestData/train",
                   batch_size = 1,
                   k = 32,
                   numWorkers = 8,
                   epochs = 100)
print(runNote)


#import os
#print(os.getcwd())





