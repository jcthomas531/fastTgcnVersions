# General

There is an issue with the IOSSeg data. All tooth classification in the model is done by color (rgd code) which is 1 to 1 associated with a tooth number. However, there are some faces in the ply files that have a color that is not associated with a tooth number. These are likely the result of reducing the granularity of the scans. Since there are groups that shouldnt exist, the model sees these as new teeth classes. My process here is find the faces that are not associated with a tooth and change their color and classification to the closest color by L2 norm of the RGB value. I will save this as IOSSegCC for color clean.

# Process in colorCleanProcess.py

 This process has been completed. Running the file colorCleanProcess.py will do it again (which is not harmful) and you may change the output location if you want to save it somewhere else. As of now, all of my data is being stored in one of Grants drives.


