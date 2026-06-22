import sys
sys.path.append("Y:/dissModels/intraoralSegmentation/tools")
import readAndFormat as raf
import plotArch

#a combination of the three above functions that simply reads in the ply file,
#formats it, and the plots it. 
#fileName is a string
#arch takes a string "L" or "U" denoting which arch we are looking at
def readAndPlot(file, arch):
    pat = raf.readAndFormat(file = file, arch = arch)
    return plotArch.plotArch(face = pat["face"], vertex = pat["vert"])

#example
# import os
# os.chdir("P:\\cph\\BIO\\Faculty\\gown\\research\\ThesisProjects\\Thomas\\IOSSegData\\test")
# readAndPlot("001_L.ply", arch = "L")