here::i_am("centroidSize/autoCentroidVsManualDistanceComp.R")
library(here)
library(dplyr)
library(openxlsx)
library(ggplot2)
library(stringr)
set.seed(826)



#helper functions
longFormat <- function(x) {
  x |>
    select(patient, starts_with("u"), starts_with("X")) |>
  tidyr::pivot_longer(
    cols = !patient,
    names_to = "measure",
    values_to = "value"
  ) |>
  mutate(
    method = case_when(
      measure %in% c("u3_3","u4_4","u5_5","u6_6") ~ "manual",
      measure %in% c("X1.16","X2.15","X3.14","X4.13","X5.12","X6.11","X7.10","X8.9") ~ "auto",
      measure %in% c("u3_3Diff", "u4_4Diff", "u5_5Diff", "u6_6Diff") ~ "diff"
    ),
    toothGroup = case_when(
      measure %in% c("u3_3Diff", "u3_3", "X6.11") ~ "dist3_3",
      measure %in% c("u4_4Diff", "u4_4", "X5.12") ~ "dist4_4",
      measure %in% c("u5_5Diff", "u5_5", "X4.13") ~ "dist5_5",
      measure %in% c("u6_6Diff", "u6_6", "X3.14") ~ "dist6_6"
    )
  )

}

measurePlot <- function(x, title) {
  x |>
    ggplot(aes(x = value, y = toothGroup, color = method)) +
    geom_jitter(height = 0.1, width = 0) +
    labs(
      title = title, x = "Distance (scaled)", y = "Group", color = "method"
    ) +
    theme_bw()
}

#

manDir <- "K:/iowaExpTest/centroidSize/"
autoDir <- "K:/iowaExpTest/centroidSize/centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/"

manPre <- read.xlsx(paste0(manDir, "manualToothDistances.xlsx"), sheet = "pre") |>
  rename_with(~ gsub("-", "_", .x))
manPost <- read.xlsx(paste0(manDir, "manualToothDistances.xlsx"), sheet = "post") |>
  rename_with(~ gsub("-", "_", .x))

autoPre <- read.csv(paste0(autoDir, "archLengthPre.csv")) |>
  select(-X)
autoPost <- read.csv(paste0(autoDir, "archLengthPost.csv"))|>
  select(-X)



allPre <- left_join(manPre, autoPre, by = join_by(patient == patNum)) |>
  mutate(
    u3_3Diff = u3_3 - X6.11,
    u4_4Diff = u4_4 - X5.12,
    u5_5Diff = u5_5 - X4.13,
    u6_6Diff = u6_6 - X3.14
  )
allPost <- left_join(manPost, autoPost, by = join_by(patient == patNum)) |>
  mutate(
    u3_3Diff = u3_3 - X6.11,
    u4_4Diff = u4_4 - X5.12,
    u5_5Diff = u5_5 - X4.13,
    u6_6Diff = u6_6 - X3.14
  )




allPreLong <- allPre |>
  longFormat()
allPostLong <- allPost |>
  longFormat()

distPlotPre <- allPreLong |>
  filter(
    !is.na(toothGroup),
    method != "diff"
  ) |>
  measurePlot(title = "Distances, pre")
diffPlotPre <- allPreLong |>
  filter(
    !is.na(toothGroup),
    method == "diff"
  ) |>
  measurePlot(title = "Patient-level differences, pre") +
  geom_vline(xintercept = 0, linetype = 2)


distPlotPost <- allPostLong |>
  filter(
    !is.na(toothGroup),
    method != "diff"
  ) |>
  measurePlot(title = "Post")
diffPlotPost <- allPostLong |>
  filter(
    !is.na(toothGroup),
    method == "diff"
  ) |>
  measurePlot(title = "Patient-level differences, post") +
  geom_vline(xintercept = 0, linetype = 2)





preTest <- allPre |>
  select(contains("Diff")) |>
  apply(2, t.test)

postTest <- allPost |>
  select(contains("Diff")) |>
  apply(2, t.test)


