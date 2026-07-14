library(dplyr)
library(ggplot2)

centDir <- "K:/iowaExpTest/centroidSize/centSize_t3dsIosseg_cSOriMastEpoch300"

preDat <- read.csv(file = paste0(centDir, "/centSizePre.csv")) |>
  select(-X) |>
  rename(
    preCentSize = centSize
  )
postDat <- read.csv(file = paste0(centDir, "/centSizePost.csv")) |>
  select(-X) |>
  rename(
    postCentSize = centSize
  )



dat <- left_join(preDat, postDat, by = join_by(patNum)) |>
  mutate(
    diffPostPre = postCentSize - preCentSize
  )

vis1 <- dat |>
  select(-diffPostPre) |>
  tidyr::pivot_longer(!patNum, names_to = "prePost", values_to = "centroidSize") |>
  mutate(prePost = factor(prePost, levels = c("preCentSize", "postCentSize"))) |>
  ggplot(aes(x = prePost, y = centroidSize, fill = prePost)) +
  geom_boxplot() +
  theme_bw() +
  labs(title = "Distribution of centroid size for pre and post scans") +
  ylim(0, NA)

res1 <- t.test(x = dat$diffPostPre)
