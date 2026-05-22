library(dplyr)
library(stringr)

#movement files
moveDir <- "K:/iowaRme/movement/preDFin_dec016/"
moveFileList <- list.files(moveDir) |> as.list()

#function to read in and format data
moveReader <- function(x) {
  #extract patient number
  patNum <- str_extract(x, "^pat[0-9]{3}")
  
  #read in file
  dat <- read.csv(paste0(moveDir, x)) |>
    mutate(patNum = patNum)
  
  return(dat)
}

#read in all movement files
moveReadList <- lapply(moveFileList, moveReader)

#merge all patients
#pad tooth number with leading zeros
allMoveDat <- do.call(rbind, moveReadList) |>
  mutate(toothNum = str_pad(toothNum, 2, pad = "0"))

#remove gum
moveDat <- allMoveDat |>
  filter(toothNum != "gum")

#pca on l2 norm
library(tidyr)
pcaDat <- moveDat |>
  select(patNum, toothNum, l2Norm) |>
  pivot_wider(names_from = toothNum, values_from = l2Norm,
              names_prefix = "tooth") |>
  #remove tooth 1 and 16 for now
  select(-tooth01, -tooth16)

#complete cases
pcaDatCompl <- pcaDat[complete.cases(pcaDat),]


pcaRes <- prcomp(pcaDatCompl[,-1], center = TRUE)
loading1 <- pcaRes$rotation[,1] |> round(3)


library(sparsepca)
aaa <- spca(pcaDatCompl[,-1], center = TRUE)
bbb <- aaa$loadings[,1] |> round(3)
names(bbb) <- names(loading1)
bbb
