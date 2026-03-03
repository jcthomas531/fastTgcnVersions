import sys
sys.path.append("/Users/jthomas48/dissModels/fastTgcnVersions/tools/convert3DS")
import conver3DSFuns as cf_
import numpy as np

#
runNote = "first attempt on HPC, U arch"
#

fp = "/Shared/gb_lss/Thomas/teeth3DS/upper/"
rng = np.random.default_rng(826)

print(runNote)

cf_.convertAll3DS(path = fp, arch = "U", rng = rng)

print(runNote)
