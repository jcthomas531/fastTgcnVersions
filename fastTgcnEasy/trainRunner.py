

#note 1
#in the future i want to move this to just a bash script where python is opened
#and train.py is imported and the function is ran but now to keep things easy
#i am going to use the same framework that I had set up in fastTgcn


#
runNote = "full training on teeth3ds data, 720 train 180 test"
#


print(runNote)
#should alredy be in the proper working directory
import train
train.fastTgcnEasy(arch = "u",
                   testPath = "/Shared/gb_lss/Thomas/trainTestSets/teeth3dsBase/test",
                   trainPath = "/Shared/gb_lss/Thomas/trainTestSets/teeth3dsBase/train",
                   batch_size = 1,
                   k = 32,
                   numWorkers = 8,
                   epochs = 301)
print(runNote)


#import os
#print(os.getcwd())





