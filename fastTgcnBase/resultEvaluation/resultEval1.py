import os
#os.getcwd()
dirPath = "Y:\\dissModels\\intraoralSegmentation"
toolExt = "\\tools"
predExt = "\\fastTgcn\\pred_global"
os.chdir(dirPath + toolExt)
import plyFunctions as pf





os.chdir(dirPath + predExt)
#lets start by investigating 001_L
l01 = pf.plyRead("059_L.ply")
l01["face"] = pf.toothVars(l01["face"])
pf.plotPly(face = l01["face"], vertex = l01["vert"])


pf.toothHigh(l01["face"], l01["vert"], ["32"])


#002 is a great example
#003 has some issues
#004 is a child, too few teeth i think, performs poorly on molars
#perhaps inclusion of age or number of teeth/presence or absence of adult teeth
#would be informative, check the truth for 004, why does the prediction do so poorly
#are there children in the training set?
#is it possible to split the model when there are less teeth, meaning, can we still
#use the information gained via all scans for all the teeth that are in common
#but then when it comes to specific back teeth in kids, use only what is common to all
#it would be interesting to see how good the accuracy is on just the teeth, is the
#gum identification articially inflating the success?

#023 is interesting
#025 we see a smattering of what I assume should be wisdom teeth coloring at the back
#this is very interesting and also points to a more informed model being able to segement
#better, something as simple as number of teeth may make a big difference let alone
#indicators for teeth present
#might be picking up 3rd molar in back, red

#026 is a kid
#not a kid, an adult missing lateral incisors
#027 also seems to be a kid with a large gap in teeth
#missing 19 and 30, wrong coloring on 18 which is falling into 19s place, wisdom tooth in back
#029 is perfect
#031 is a mess
#037 is a mess and has weird tooth placement
#has a torus maybe, gums covering molar? second molar but that is usally a characteristic
#of the third molar, unless its a kid which is unlikely
#052 has 13 teeth
#missing both lateral incisors 
#059 is a kid with a large gap, with some baby teeth
#077 has 15 teeth
#096 is a kid with 11 teeth, still has 3 baby teeth, missing a tooth as well (but thats ok)
#below 12, about 9 or 10
#097 is pretty good
#100 is a kid



#when thinking about things like calculating centroids when there are some things
#that are colored that arent the correct tooth, could use just the major connected
#component like we do for networks

