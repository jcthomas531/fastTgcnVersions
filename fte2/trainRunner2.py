

#note 1
#in the future i want to move this to just a bash script where python is opened
#and train.py is imported and the function is ran but now to keep things easy
#i am going to use the same framework that I had set up in fastTgcn


#
runNote = "attempting constant learning rate"
#


print(runNote)
#should alredy be in the proper working directory
import train2
train2.fastTgcnEasy(arch = "u",
                   testPath = "/Shared/gb_lss/Thomas/trainTestSets/remeshT3dsIos_csRot_smaller/test",
                   trainPath = "/Shared/gb_lss/Thomas/trainTestSets/remeshT3dsIos_csRot_smaller/train",
                   batch_size = 1,
                   k = 32,
                   numWorkers = 8,
                   epochs = 101)
print(runNote)


#import os
#print(os.getcwd())





