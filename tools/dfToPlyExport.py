import numpy as np
from plyfile import PlyData, PlyElement


#function to export mesh as a ply in the standard format with the vertex and face
#information stored as pandas data frames. The idea is that if we can get any representation
#of a mesh into a data frame format, we can export it in the standrard format used 
#throughout the analysis
#
#vertDf, a pandas data frame with columns x, y, z, nx, ny, nz in that order
#faceDf, a pandas data frame with columns vertex_indices, red, green, blue, alpha in that order
#where vertex_indices is of format ["v1", "v2", "v3"]
#pointLabCol, name of column that contains point labels if they exist, None is default
#see "Y:\dissModels\intraoralSegmentation\superimposition\rugaeAnnotRegistartion.py" for specifics on pointLabCol usage
def dfToPlyExport(vertDf, faceDf, outFile, pointLabCol = None):
    
    
    #vertex information
    if pointLabCol is None:
        vertex_dtype = [
                ('x',  'f4'),
                ('y',  'f4'),
                ('z',  'f4'),
                ('nx', 'f4'),
                ('ny', 'f4'),
                ('nz', 'f4'),
            ]
        #prepare vertex object
        vertPly = np.empty(len(vertDf), dtype=vertex_dtype)
        #put in values in the correct format
        for name in ["x", "y", "z", "nx", "ny", "nz"]:
            vertPly[name] = vertDf[name].astype(np.float32).values
        #prepatre as ply element
        vertReady = PlyElement.describe(vertPly, 'vertex')
    else: 
        vertex_dtype = [
                ('x',  'f4'),
                ('y',  'f4'),
                ('z',  'f4'),
                ('nx', 'f4'),
                ('ny', 'f4'),
                ('nz', 'f4'),
                (pointLabCol, 'f4'),
            ]
        #prepare vertex object
        vertPly = np.empty(len(vertDf), dtype=vertex_dtype)
        #put in values in the correct format
        for name in ["x", "y", "z", "nx", "ny", "nz", pointLabCol]:
            vertPly[name] = vertDf[name].astype(np.float32).values
        #prepatre as ply element
        vertReady = PlyElement.describe(vertPly, 'vertex')
    
    
    

    #face information
    face_dtype = [
        ('vertex_indices', 'O'),  # 'O' for object (list of ints)
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('alpha', 'u1')
    ]
    #prepare face object
    facesPly = np.empty(len(faceDf), dtype=face_dtype)
    #put in values in the correct format
    facesPly['vertex_indices'] = faceDf['vertex_indices'].values
    facesPly['red']   = faceDf['red'].values.astype(np.uint8)
    facesPly['green'] = faceDf['green'].values.astype(np.uint8)
    facesPly['blue']  = faceDf['blue'].values.astype(np.uint8)
    facesPly['alpha'] = faceDf['alpha'].values.astype(np.uint8)
    #prepare as ply element
    faceReady = PlyElement.describe(facesPly, 'face')
    
    #combine the vertex and face information
    plyReady = PlyData([vertReady, faceReady], text = True)
    #export
    plyReady.write(outFile)