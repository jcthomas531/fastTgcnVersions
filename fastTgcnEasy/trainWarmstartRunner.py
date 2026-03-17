
#
runNote = "warmstartTest4_2, same test data as model trained on, new train data, 30 registered teeth3ds, 15 with base 15 without"
#


print(runNote)
#should alredy be in the proper working directory
import trainWarmstart as tws
tws.fastTgcnWarm(arch = "u",
                   testPath = "/Shared/gb_lss/Thomas/IOSSegData/clean/testCleanU",
                   trainPath = "/Shared/gb_lss/Thomas/testDir/warmstartTestData/train",
                   batch_size = 1,
                   k = 32,
                   numWorkers = 8,
                   epochs = 101)
print(runNote)


#import os
#print(os.getcwd())





