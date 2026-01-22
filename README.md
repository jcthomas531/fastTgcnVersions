# Overview

Various versions of the fastTgcn model taken from https://github.com/MIVRC/Fast-TGCN comprising minor fixes, edits for ease of use, and introduction of new ideas to improve model performance.

The code as written in the original fastTgcn repository does not work out of the box. The directory fastTgcnBase is a minimally updated version that fixes these small issues and gets the model running according to how it is described in the README of the original repository. Details of these changes can be found in the readme of fastTgcnBase. All other versions of fastTgcn in this repository are built up from fastTgcnBase.

fastTgcnBaby is a shrunk-down version of the original model modified to run on my local machine for proof of concept.

fastTgcnEasy is a more use friendly version of fastTgcn formatted as a function with options for changing data direectories, arches, ect.



# Models

## fastTgcnBase

A minimally updated version of the original that has had some basic error fixed so that it runs. Original: https://github.com/MIVRC/Fast-TGCN. This version does not impliment any improvements or corrections other than what it takes to get things running.

## fastTgcnBaby

The same as fastTgcn just updated to run on a smaller set of data.

## fastTgcnEasy

My version of fastTgcn with the same changes necessary to run as fastTgcn needed as well as fixing other error. Main idea is the functionization of the process so you can specify filepaths, upper vs lower jaw, ect without actually having to edit the code. Major changes listed in README.md in the directory.


# Major plans

* try new loss functions that utilize information we know about the problem, multimodal learning. See notes from 2025-11-07 and 2025-11-14 for specifics. Also need to investigate the loss function they are using.
* adding in a layer prior to the segmentation that calssifies how many teeth there are, like a unet
* trying this model on different data set, something without so many "problem" cases

# Known issues

* in tools/colorClean/colorCleanProcess.py, when a .ply file is color corrected and exported, there are too many values after the decimal point. Somehow the granularity increased between the input data and the export data which is not possible. I believe that is happening is some sort of data storage issue (ie something being stored as a float32 vs float64). Probably a simple fix but will take some time to identify where it is. As it wont make much different in the long run, I am leaving it for now.





