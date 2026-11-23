import pandas as pd
import math

#produce a data frame with the desired directional movement
#values produced by sara rodgers

desiredDirec = pd.DataFrame({
    "toothNum": list(range(2, 16)),
    "degree": 
    [180 - i for i in [8.1, 10.2, 21.2, 28.5, 31.5, 48.5, 80.3]] +
    [80.3, 48.5, 31.5, 28.5, 21.2, 10.2, 8.1]
})
desiredDirec["radian"] = [i * math.pi / 180 for i in desiredDirec["degree"]]
desiredDirec["x_u"] = [math.cos(i) for i in desiredDirec["radian"]]
desiredDirec["y_u"] = [math.sin(i) for i in desiredDirec["radian"]]
#i beleive that the desired movement is 5mm, scaling x and y movement to reflect this
#this can be also set for each specific tooth
#this needs to be dicussed with the group
desiredDirec["x_mag"] = [i * 5 for i in desiredDirec["x_u"]]
desiredDirec["y_mag"] = [i * 5 for i in desiredDirec["y_u"]]
