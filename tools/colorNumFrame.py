import pandas as pd
#functions that just give the tooth number and color information for each arch
#there is probably a better way to do this but right now this will work
#arch is either "U" or "L"
#
#copy of function by the same name that was originally in plyFunctions.py
#keeping that one there for legacy reasons but use this one moving forward
#
def colorNumFrame(arch):
    if arch == "U":
        #the matchups
        numCol = pd.DataFrame(
            {
             "toothNum": ["16","15","14","13","12","11","10","3","2","1","4","9","5","6","8","7","gum"], #universal numbering system
             "fdiNum": [28, 27, 26, 25, 24, 23, 22, 16, 17, 18, 15, 21, 14, 13, 11, 12, 0], #FDI numbering system
             "color": ['155-048-255', '255-099-071', '255-211-155','131-111-255','255-106-106',
                       '060-179-113', '255-246-143', '255-000-255', '030-144-255', '000-255-127',
                       '000-255-255', '127-255-000', '255-255-000', '000-255-000', '255-000-000',
                       '000-000-255', '255-255-255']
            }
            )
        #making it nice to use
        numCol = numCol.assign(
            red = pd.to_numeric(numCol["color"].str.extract(r"(^[0-9]{3})")[0], errors='raise'),
            green = pd.to_numeric(numCol["color"].str.extract(r"^[0-9]{3}-([0-9]{3})-[0-9]{3}$")[0], errors='raise'),
            blue = pd.to_numeric(numCol["color"].str.extract(r"([0-9]{3}$)")[0], errors='raise')
            )
    elif arch == "L":
        #the matchups
        numCol = pd.DataFrame(
            {
            "toothNum": ["25","24","26","23","27","22","28","21","29","20","30","19","31","18","32","17","gum"], #universal numbering system
            "fdiNum": [41, 31, 42, 32, 43, 33, 44, 34, 45, 35, 46, 36, 47, 37, 48, 38, 0], #FDI numbering system
            "color": ['139-000-000', '255-048-048', '144-238-144', '000-191-255', '000-139-139',
                      '255-165-000', '000-000-139','202-255-112', '139-000-139', '200-255-255',
                      '255-105-180', '255-228-255',  '230-230-250', '255-155-255', '255-228-181',
                      '255-069-000', '255-255-255']
            }
            )
        #make individual columns for the colors for easy use and make them numeric
        numCol = numCol.assign(
            red = pd.to_numeric(numCol["color"].str.extract(r"(^[0-9]{3})")[0], errors='raise'),
            green = pd.to_numeric(numCol["color"].str.extract(r"^[0-9]{3}-([0-9]{3})-[0-9]{3}$")[0], errors='raise'),
            blue = pd.to_numeric(numCol["color"].str.extract(r"([0-9]{3}$)")[0], errors='raise')
            )
    else: 
        #for some reason this is not working
        raise ValueError("arch arguement must be either 'L' or 'U'")
        
        
        
    return numCol

# example
# colorNumFrame("L")
# colorNumFrame("U")