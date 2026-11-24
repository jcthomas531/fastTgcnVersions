import pandas as pd
import math
import numpy as np

#produce a data frame with the desired directional movement
#values produced by sara rodgers


#set value for range of acceptable directions around desired direction
#value is for either side, thus the total width is 2x this value
accRange = math.pi/2

desiredDirec = pd.DataFrame({
    "toothNum": list(range(2, 16)),
    "degree": 
    [180 - i for i in [8.1, 10.2, 21.2, 28.5, 31.5, 48.5, 80.3]] +
    [80.3, 48.5, 31.5, 28.5, 21.2, 10.2, 8.1]
})

#to ensure these radians are always positive applying modulus function
#thus they will always indicating rotation from postive x axis (wouldnt be a worry in this particular case bc all vectors in quadrants 1 and 2)
desiredDirec["radian"] = np.mod([i * math.pi / 180 for i in desiredDirec["degree"]], 2*math.pi)
desiredDirec["x_u"] = [math.cos(i) for i in desiredDirec["radian"]]
desiredDirec["y_u"] = [math.sin(i) for i in desiredDirec["radian"]]
#for a reference magnitude, 2.5mm is used though this can be changed later
#this can be also set for each specific tooth
magVal = 2.5
desiredDirec["x_mag"] = [i * magVal for i in desiredDirec["x_u"]]
desiredDirec["y_mag"] = [i * magVal for i in desiredDirec["y_u"]]

#
#setting up range of acceptable directions, currently set to within 90 degrees either side of desired direction
#
#i am not putting the modulus operation on these radian measures
#thier sign is important to the correct direction of the plotting operation
#however, this makes them a bit confusing as far as interpretation when compared with other radian measures
#thus, they can be used for plotting but they should not be used for other tasks
desiredDirec["radian_lwrPLOT"] = desiredDirec["radian"] - accRange
desiredDirec["radian_uprPLOT"] = desiredDirec["radian"] + accRange


