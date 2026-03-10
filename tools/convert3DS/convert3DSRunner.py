import sys
sys.path.append("/Users/jthomas48/dissModels/fastTgcnVersions/tools/convert3DS")
import convert3DSFuns as cf_
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None)


#
runNote = "decimation to 16000 faces"
#

fp = "/Shared/gb_lss/Thomas/teeth3DS/scanData/upper/"
rng = np.random.default_rng(826)

print(runNote)

print(cf_.convertAll3DS(path = fp, arch = "U", rng = rng, decimate = True, nFace = 16000))

print(runNote)
