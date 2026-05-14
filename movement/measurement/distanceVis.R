setwd("H:/schoolFiles/dissertation/movementModeling/measurement")
library(dplyr)
datList <- list(
  read.csv("pat055uDist_dec016.csv") |>
    mutate(pat = "p055"),
  read.csv("pat056uDist_dec016.csv") |>
    mutate(pat = "p056"),
  read.csv("pat057uDist_dec016.csv") |>
    mutate(pat = "p057"),
  read.csv("pat058uDist_dec016.csv") |>
    mutate(pat = "p058")
)

datRaw <- do.call(rbind, datList)
dat <- datRaw |>
  filter(toothNum != "gum") |>
  mutate(toothNum = as.numeric(toothNum))

library(ggplot2)

dat |>
  filter(pat != "p056") |>
  ggplot(aes(x = toothNum, y = l2Norm, color = pat)) +
    geom_point() +
  ylim(0, NA) +
  theme_bw() +
  scale_x_continuous(breaks = 0:17)


#take a look at pca on each individuals 15 tooth vector to see if trends are detectable
#multiple sources of error being fed into the model is an interesting statistical question
#thinking about models that address this type of error coming from measurements 
#(robustness to segmentation error, multiple effects, misture models, are we observing something that is nicely behaved or poorly behaved)



