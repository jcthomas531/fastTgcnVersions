

#note 1
#in the future i want to move this to just a bash script where python is opened
#and train.py is imported and the function is ran but now to keep things easy
#i am going to use the same framework that I had set up in fastTgcn


#
runNote = "testing new output messages, file storage location, and clean plys"
#


print(runNote)
#should alredy be in the proper working directory
import train
train.fastTgcnEasy(arch = "l",
                   testPath = "/Shared/gb_lss/Thomas/IOSSegData/clean/testCleanMiniL",
                   trainPath = "/Shared/gb_lss/Thomas/IOSSegData/clean/trainCleanMiniL",
                   batch_size = 1,
                   k = 32,
                   numWorkers = 8,
                   epochs = 31)
print(runNote)


#import os
#print(os.getcwd())





