library(dplyr)
library(stringr)
library(ggplot2)
set.seed(826)

#this will need to be modified when this file finds a new home
#testing
perfDir <- "K:/iowaExpTest/rugaePipeOptim/performance/run001"
#this should work on hpc but i cannot navigate like this when running locally
perfDir <- "../../../../../Shared/gb_lss/Thomas/iowaExpTest/rugaePipeOptim/performance/run001"

setwd(perfDir)


#------------------------------------------------------------------------------------------------------
#function for reading in and formatting data
#note that this is hard coded thus if the structure of the file system changes, this
#will no longer work. this must always be ran from the same working directory
reader = function(x) {
  #split paths on folder separator
  folderVecFull <- strsplit(x, .Platform$file.sep)[[1]]
  folderVecFullLen <- length(folderVecFull)

  #make new entries for the "normX_descY" names
  normIdx <- which(str_detect(folderVecFull, "^norm"))
  splitNorm <- strsplit(folderVecFull[normIdx], "_")[[1]]

  #separating decimation technique and resolution
  resoIdx <- which(
    str_detect(folderVecFull, "^decim") | str_detect(folderVecFull, "^remesh")
  )
  resoVal <- str_extract(folderVecFull[resoIdx], "[0-9]+")
  techVal <- str_extract(folderVecFull[resoIdx], "^[a-zA-Z]+")


  #create vector with pertinant info
  folderVec <- c(
    folderVecFull[-c(1, normIdx, resoIdx, folderVecFullLen)],
    splitNorm,
    techVal,
    resoVal
  )

  #make into data frame
  folderDf <- as.matrix(folderVec) |>
    t() |>
    as.data.frame()
  #pretty up data frame
  colnames(folderDf) <- c(
    "folds", "model", "descriptor", "normRadMult", "descRadMult", "reductTech", "resol"
  )
  folderDf <- folderDf |>
    mutate(
      resol = as.numeric(resol),
      folds = str_extract(folds, "[0-9]+$") |>
        as.numeric(),
      normRadMult = str_remove(normRadMult, "^[a-zA-Z]+")
        |> as.numeric(),
      descRadMult = str_remove(descRadMult, "^[a-zA-Z]+")
        |> as.numeric(),
      runId = paste0(folds, model, descriptor, normRadMult, descRadMult, reductTech, resol)
    )

  #read in results
  dati <- read.csv(x) 

  #combine with folder infor
  datFulli <- cbind(folderDf, dati)

  return(datFulli)
}
#------------------------------------------------------------------------------------------------------

#perform data read
filePaths <- list.files(
  path = ".",
  pattern = "\\.csv$",
  recursive = TRUE,
  full.names = TRUE
) |>
  as.list()

datList <- lapply(filePaths, reader)
datDf <- do.call(rbind, datList)
datDf_no15 <- datDf |>
  filter(patId != "pat015")
datDf_remesh <- datDf |>
  filter(reductTech == "remesh")

runSummariser <- function(x) {
  x |>
    group_by(runId) |>
    summarise(
      folds = first(folds),
      model = first(model),
      descriptor = first(descriptor),
      normRadMult = first(normRadMult),
      descRadMult = first(descRadMult),
      reductTech = first(reductTech),
      resol = first(resol),
      meanGamma = mean(gamma),
      meanLR = mean(learning_rate),
      meanDepth = mean(max_depth),
      meanN = mean(n_estimators),
      medianIou = median(iou),
      meanIou = mean(iou),
      meanRocAuc = mean(rocAuc)
    )

}

runSum <- datDf |>
  runSummariser()

runSum_no15 <- datDf_no15 |>
  runSummariser()

runSum_remesh <- datDf_remesh |>
  runSummariser()

ggIou <- function(x, title) {
  x +
    geom_jitter(width = 200, alpha = .7) +
    ylim(0,1) + 
    theme_bw() +
    labs(
      title = title,
      x = "Scan resolution",
      y = "IoU",
      color = "Reduction Tech."
    )
}


#iou results
datDf |>
  ggplot(aes(x = resol, y = iou, color = reductTech)) |>
  ggIou(title = "All LOOCV iterations")

datDf_no15 |>
  ggplot(aes(x = resol, y = iou, color = reductTech)) |>
  ggIou(title = "All LOOCV iterations, no patient 15")


runSum |>
  ggplot(aes(x = resol,  y = meanIou, color = reductTech)) |>
  ggIou(title = "Mean validation IoU")

runSum_no15 |>
  ggplot(aes(x = resol, y = meanIou, color = reductTech)) |>
  ggIou(title = "Mean validation IoU, no patient 15")


#auc results
runSum_no15 |>
  ggplot(aes(x = resol, y = meanRocAuc, color = reductTech)) |>
  ggIou(title = "Mean validation AUC, no patient 15") +
  labs(y = "ROC AUC")






table(datDf$n_estimators, datDf$resol)

#the remeshed data seems to benefit from low gamma, high n_estimators, high learning_rate, high max_depth
table(datDf_remesh$gamma)
table(datDf_remesh$n_estimators)
table(datDf_remesh$learning_rate)
table(datDf_remesh$max_depth)

datDf |>
  filter(patId != "pat015") |>
  group_by(reductTech) |>
  summarise(meanIou = mean(iou))

runSum |>
  filter(reductTech == "decim") |>
  group_by(resol) |>
  summarise(meanIou = mean(meanIou))


#finding best overall setting (ignoring pat015)
bestSetting <- which.max(runSum_no15$meanIou)
runSum_no15[bestSetting,]
