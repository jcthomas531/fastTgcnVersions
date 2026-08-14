library(dplyr)
library(ggplot2)
library(ggbeeswarm)

#data directory
centDir <- "K:/iowaExpTest/centroidSize/centSize_t3dsIosseg_cSOriMastEpoch300/rugAnnotForm_cSOriMastRemesh/"

#bring in centroid data
preCentDat <- read.csv(file = paste0(centDir, "/centSizePre.csv")) |>
  select(-X) |>
  rename_with(.cols = fullCs:anteriorCs, .fn = function(x){paste0("pre_", x)})
postCentDat <- read.csv(file = paste0(centDir, "/centSizePost.csv")) |>
  select(-X) |>
  rename_with(.cols = fullCs:anteriorCs, .fn = function(x){paste0("post_", x)})

centDat <- left_join(preCentDat, postCentDat, by = join_by(patNum)) |>
  mutate(
    diff_fullCs = post_fullCs - pre_fullCs,
    diff_molarCs = post_molarCs - pre_molarCs,
    diff_premolarCs = post_premolarCs - pre_premolarCs,
    diff_posteriorCs = post_posteriorCs - pre_posteriorCs,
    diff_canineCs = post_canineCs - pre_canineCs,
    diff_incisorCs = post_incisorCs - pre_incisorCs,
    diff_anteriorCs = post_anteriorCs - pre_anteriorCs
  )

#bring in arch length data
preLenDat <- read.csv(file = paste0(centDir, "/archLengthPre.csv")) |>
  select(-X) |>
  rename_with(.cols = archLength:X8.9, .fn = function(x){paste0("pre_", x)})
postLenDat <- read.csv(file = paste0(centDir, "/archLengthPost.csv"))|>
  select(-X) |>
  rename_with(.cols = archLength:X8.9, .fn = function(x){paste0("post_", x)})

lenDat <- left_join(preLenDat, postLenDat, by = join_by(patNum)) |>
  mutate(
    diff_archLength = post_archLength - pre_archLength,
    diff_fullPerimeter = post_fullPerimeter - pre_fullPerimeter,
    diff_1_16 = post_X1.16 - pre_X1.16,
    diff_2_15 = post_X2.15 - pre_X2.15,
    diff_3_14 = post_X3.14 - pre_X3.14,
    diff_4_13 = post_X4.13 - pre_X4.13,
    diff_5_12 = post_X5.12 - pre_X5.12,
    diff_6_11 = post_X6.11 - pre_X6.11,
    diff_7_10 = post_X7.10 - pre_X7.10,
    diff_8_9 = post_X8.9 - pre_X8.9
  )



#summary of centroid data
centroidTypes = paste0(
  c("full", "molar", "premolar", "posterior", "canine", "incisor", "anterior"),
  "Cs"
  )
columnLevels = paste0(c("pre_", "post_"), rep(centroidTypes, each = 2))
vis1 <- centDat |>
  select(-starts_with("diff")) |>
  tidyr::pivot_longer(!patNum, names_to = "centroid", values_to = "centroidSize") |>
  mutate(centroid = factor(centroid, levels = columnLevels)) |>
  ggplot(aes(x = centroid, y = centroidSize, color = centroid)) +
  geom_beeswarm() +
  theme_bw() +
  labs(title = "Distribution of centroid size") +
  ylim(0, NA) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))
  

sum1 <- centDat |>
  select(patNum, starts_with("diff")) |>
  tidyr::pivot_longer(!patNum, names_to = "centroid", values_to = "centroidSize") |>
  group_by(centroid) |>
  summarise(
    meanDiff = mean(centroidSize, na.rm = TRUE),
    lwrCi = t.test(centroidSize)$conf.int[1],
    uprCi = t.test(centroidSize)$conf.int[2]
  ) |>
  mutate(
    centroid = stringr::str_remove(centroid, pattern = "^diff_") |>
      stringr::str_remove(pattern = "Cs")
  )

#summary of arch length data
lengthTypes = c("archLength", "fullPerimeter")
lengthLevels = paste0(c("pre_", "post_"), rep(lengthTypes, each = 2))
lenVis1 <- lenDat |>
  select(patNum, -starts_with("diff") & (contains("archLength") | contains("fullPerimeter"))) |>
  tidyr::pivot_longer(!patNum, names_to = "measureName", values_to = "value") |>
  mutate(measureName = factor(measureName, levels = lengthLevels)) |>
  ggplot(aes(x = measureName, y = value, color = measureName)) +
  geom_beeswarm() +
  theme_bw() +
  labs(title = "Distribution of various lengths") +
  ylim(0, NA) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))


lenSum1 <- lenDat |>
  select(patNum, starts_with("diff") & (contains("archLength") | contains("fullPerimeter"))) |>
  tidyr::pivot_longer(!patNum, names_to = "measureName", values_to = "value") |>
  group_by(measureName) |>
  summarise(
    meanDiff = mean(value, na.rm = TRUE),
    lwrCi = t.test(value)$conf.int[1],
    uprCi = t.test(value)$conf.int[2]
  ) |>
  mutate(
    measureName = stringr::str_remove(measureName, pattern = "^diff_")
  )



lenDat |>
  select(patNum, starts_with("diff")) |>
  tidyr::pivot_longer(!patNum, names_to = "measureName", values_to = "value") |>
  group_by(measureName) |>
  summarise(
    meanDiff = mean(value, na.rm = TRUE),
    #lwrCi = t.test(value)$conf.int[1],
    #uprCi = t.test(value)$conf.int[2]
  ) |>
  mutate(
    measureName = stringr::str_remove(measureName, pattern = "^diff_")
  )
