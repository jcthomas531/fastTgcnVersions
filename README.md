# Structure

## fastTgcn

A minimally updated version of the original that has had some basic error fixed so that it runs. Original: https://github.com/MIVRC/Fast-TGCN. This version does not impliment any improvements or corrections other than what it takes to get things running.

## fastTgcnBaby

The same as fastTgcn just updated to run on a smaller set of data.

## fastTgcnEasy

My version of fastTgcn with the same changes necessary to run as fastTgcn needed as well as fixing other error. Main idea is the functionization of the process so you can specify filepaths, upper vs lower jaw, ect without actually having to edit the code. Major changes listed in README.md in the directory.




# Known issues

* in tools/colorClean/colorCleanProcess.py, when a .ply file is color corrected and exported, there are too many values after the decimal point. Somehow the granularity increased between the input data and the export data which is not possible. I believe that is happening is some sort of data storage issue (ie something being stored as a float32 vs float64). Probably a simple fix but will take some time to identify where it is. As it wont make much different in the long run, I am leaving it for now.





