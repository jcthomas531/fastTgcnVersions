import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import toothVars
import plyRead

#a comination of the first two functions that does them both at the same time
#takes file as a string
#arch takes a string "L" or "U" denoting which arch we are looking at
def readAndFormat(file, arch = "U"):
    pat = plyRead.plyRead(file)
    pat["face"] = toothVars.toothVars(pat["face"], arch=arch)
    return pat

#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\test")
# readAndFormat("001_L.ply", arch = "L")