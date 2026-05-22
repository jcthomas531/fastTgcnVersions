here::i_am("movement/visualization/centroidMovement/centroidMoveVis.R")
library(here)
visPath <- here("movement/visualization/centroidMovement")
library(dplyr)
library(stringr)
library(ggplot2)
library(plotly)
library(ggbeeswarm)

#get arguments from snakemake
snakeArgs <- commandArgs(trailingOnly = TRUE)
patLinesPath <- snakeArgs[1]
beePlotPath <- snakeArgs[2]




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
  mutate(toothNumChar = str_pad(toothNum, 2, pad = "0"))

#remove gums and make tooth number numeric
#also make a character version with padded zeros
moveDat <- allMoveDat |>
  filter(toothNum != "gum") |>
  mutate(
    toothNum = as.numeric(toothNum),
    toothNumChar = sprintf("%02d", toothNum)
    )



#lines for each patient
moveLines <- moveDat |>
  ggplot(aes(x = toothNum, y = l2Norm, color = patNum)) +
  geom_point() +
  geom_line() +
  theme_bw() +
  scale_x_continuous(breaks = 1:17) +
  labs(title = "Centroid movement across teeth for each patient")
#make into a plotly and export
moveLinesPlotly <- ggplotly(moveLines)
htmlwidgets::saveWidget(moveLinesPlotly,
                        file = here(patLinesPath))



#bee swarm plot
moveBee <- allMoveDat  |>
  ggplot(aes(x = toothNumChar, y = l2Norm, fill = toothNumChar)) +
  geom_beeswarm(color = "black", shape = 21) +
  theme_bw() +
  labs(title = "L2 centroid movement bee swarm plot") +
  theme(legend.position = "none")
ggsave(filename = here(beePlotPath),
       plot = moveBee,
       width = 6,
       height = 3.5,
       units = "in")








