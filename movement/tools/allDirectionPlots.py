from plotnine import ggplot, aes, geom_histogram, labs, theme_bw, geom_vline


#designed specifically for the patient movement quarto summaries
#dat is rigidResults output from rigidMoveSummary.py
def allDirectionPlots(dat):
    #extract main res
    mainRes = dat["mainResults"]

    #extract pat number
    allTeeth = list(mainRes.keys())
    arbitTooth = allTeeth[0]
    patNum = mainRes[arbitTooth]["patNum"]

    toothPlots = {}
    for i in allTeeth:
        toothDati = mainRes[i]["diffDat"]
        sumi = mainRes[i]["diffSummary"]

        plotDicti = {}
        plotDicti["x"] = (
            ggplot(toothDati, aes(x = "x")) + 
            geom_histogram(fill = "dodgerblue", color = "black") +
            labs(
                title =  "Movement in X",
                subtitle= "Patient: "+ str(patNum) + ", tooth: " + i + ", median plotted",
                x = "Movement in units",
                y = "Count"
                ) +
            theme_bw() +
            geom_vline(xintercept = sumi["medX"], linetype = "dashed", size = 1)
        )
        plotDicti["y"] = (
            ggplot(toothDati, aes(x = "y")) + 
            geom_histogram(fill = "firebrick", color = "black") +
            labs(
                title =  "Movement in Y",
                subtitle= "Patient: "+ str(patNum) + ", tooth: " + i + ", median plotted",
                x = "Movement in units",
                y = "Count"
                ) +
            theme_bw() +
            geom_vline(xintercept = sumi["medY"], linetype = "dashed", size = 1)
        )
        plotDicti["z"] = (
            ggplot(toothDati, aes(x = "z")) + 
            geom_histogram(fill = "purple", color = "black") +
            labs(
                title =  "Movement in Z",
                subtitle= "Patient: "+ str(patNum) + ", tooth: " + i + ", median plotted",
                x = "Movement in units",
                y = "Count"
                ) +
            theme_bw() +
            geom_vline(xintercept = sumi["medZ"], linetype = "dashed", size = 1)
        )

        toothPlots[i] = plotDicti
    
    return toothPlots










    