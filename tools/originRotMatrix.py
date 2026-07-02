import math
import numpy as np
#create rotation matrix around a particular axis at the orgin
#degress in the amount to turn in degrees (numeric)
#axis is the desired axis to trun around, character "x", "y", "z"
#returns a numpy array
def originRotMatrix(degrees, axis):
    
    #set up
    rad = math.radians(degrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    
    #build matrix by rows
    #composition is based on axis to rotate around
    if axis == "x":
        r1 = [1, 0, 0, 0]
        r2 = [0, cos, sin, 0]
        r3 = [0, -sin, cos, 0]
        r4 = [0, 0, 0, 1]
    elif axis == "y":
        r1 = [cos, 0, -sin, 0]
        r2 = [0, 1, 0, 0]
        r3 = [sin, 0, cos, 0]
        r4 = [0, 0, 0, 1]
    elif axis == "z":
        r1 = [cos, -sin, 0, 0]
        r2 = [sin, cos, 0, 0]
        r3 = [0, 0, 1, 0]
        r4 = [0, 0, 0, 1]
    else:
        raise ValueError("axis arguement must be 'x', 'y', or 'z'")
    
    #stack rows into matrix
    mat = np.array([r1, r2, r3, r4])
    
    return mat

# example
# originRotMatrix(180, "z")