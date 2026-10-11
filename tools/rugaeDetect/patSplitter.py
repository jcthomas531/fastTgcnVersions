#make a dictionary where each entry has a data frame for a single hold out patient
#and a data frame for the rest of the patients

#x is a data frame of combined labeled patient descriptors that has column patId

def patSplitter(x):
    # Unique patient numbers
    uniqPatNums = x["patId"].str.extract(r"([0-9]{3})")[0].unique()

    # Create output dictionary
    leftOutList = {}

    for pati in uniqPatNums:
        # Identify rows belonging to this patient number
        logicInd = x["patId"].str.contains(pati)

        # Patient and non-patient data frames
        patiDf = x[logicInd].copy()
        nonPatiDf = x[~logicInd].copy()

        # Store them
        leftOutList[f"pat{pati}"] = {
            "pat": patiDf,
            "nonPat": nonPatiDf
        }

    return leftOutList