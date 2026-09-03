from xlrd import colname
import xgboost as xgb
import pandas as pd





#testing
inDir = "K:/iowaExpTest/localDescriptors/rugAnnotForm_cSOriMastRemesh_localDescr/"



d1 = pd.read_csv(inDir + "preLabeledCsv/pat001Pre_localDescrLabel.csv")
d1["pat"] = "001"
d2 = pd.read_csv(inDir + "preLabeledCsv/pat004Pre_localDescrLabel.csv")
d2["pat"] = "004"
dAll = pd.concat([d1, d2])


#following this guide
#https://xgboost.readthedocs.io/en/stable/python/python_intro.html





# https://www.youtube.com/watch?v=aLOQD66Sj0g



# https://www.youtube.com/watch?v=GrJP9FLV3FE
X = dAll.drop(columns=["label", "pat"]).copy()
y = dAll["label"].copy()

#data checking
X.dtypes.unique()
X.dtypes.value_counts()
y.dtypes
