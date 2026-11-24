import pandas as pd
import numpy as np


def allPatRes_process1(x, patNums):
    # stack all patient median tables
    medAll = pd.concat(
        [x[i]["medianTable"].assign(patNum=i) for i in patNums], ignore_index=True
    )[["patNum", "toothNum", "medX", "medY", "medZ", "medL2"]]

    # convert movement to expansion
    #teeth 1-8 are on the other side of the midline so x is flipped to make outward movement positive
    medAll_expand = medAll.assign(
        # this is like ifelse in r
        # converting tooth num to numeric
        medXExpand=lambda d: np.where(
            pd.to_numeric(d["toothNum"]) <= 8, -d["medX"], d["medX"]
        )
    )

    # tooth number to numeric for ease of plotting
    medAll_expand["toothNum"] = pd.to_numeric(medAll_expand["toothNum"])

    return medAll_expand