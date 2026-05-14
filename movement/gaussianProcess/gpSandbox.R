set.seed(826)
library(rstan)

#lets take a look at gaussian process using 1d vector movements

#generate data
#start with a correlation matrix, 16x16 describing the relationship of each tooth
#in the top arch to each other tooth in the top arch
#to simplify things for the first attempt, we will assume that there is only correlation
#on the basis of proximity, not symmetry
#this is a pretty standard set up, we will use AR1 to show decaying relationship
#this is probably an oversimplification of the decaying relationship but will work 

library(simstudy)
#https://cran.r-project.org/web/packages/simstudy/vignettes/corelationmat.html
cor1 <- genCorMat(16, rho = .7, corstr = "ar1")
genCorGen(100, nvars = 16, dist = "normal", params1 = c(8:1, 1:8), params2 = 1, corMatrix = cor1)
library(mvtnorm)
#draw 100 samples of 16 teeth, ie 100 patients
aaa <- rmvnorm(100, mean = c(8:1, 1:8), sigma = cor1)
library(corrplot)
corrplot(cor(aaa))
corrplot(cor1)
corrplot(cor(aaa) - cor1)
barplot(aaa[1,])


#how do we feed this data to stan?
#how does the model know what the "location" of an observation is?
#how is that encoded in the data?
#perhaps should work through the tutorial from the beginning to see how things are formatted
#and then try it with my data

#following tutorial
#https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#1_Modeling_Functional_Relationships
library(dplyr)
library(stringr)
aaa <- as.data.frame(aaa)
aaa$id <- 1:nrow(aaa)
longDat <- aaa |>
  tidyr::pivot_longer(names_to = "tooth",
                      cols = !id,
                      values_to = "dist") |>
  mutate(tooth = str_extract(tooth, "[0-9]+$") |> as.numeric())

#example of entire observed data
plot(dist ~ tooth, data = longDat, main = "all observed data")

#subset to a few observations
sampIds <- sample(1:nrow(aaa), 10)
datSmall <- longDat |>
  filter(id %in% sampIds)
plot(dist ~ tooth, data = datSmall, main = "subset of observed data")
library(ggplot2)
datSmall |> 
  ggplot(aes(x = tooth, y = dist, color = as.character(id))) +
  geom_line() +
  guides(color="none") +
  labs(title = "subset of observed data") +
  theme_bw()


#from here I will be following section 3.2.3 from betancourt tutorial
#in his work, he has what seems to be the equivalent of a single sample
#i will try it this way first and they try to expand it
datSingle <- longDat |>
  filter(id== 1)
#this assumes an exponentiated quadtratic covariance function, which is probably
#fine for the true covariance i have set up

#we want to have an informative prior on rho like he did
#the informativeness of rho was based on the max and min distance measurements
#in this case, there can be no distance less than 1 and no distance greater than 15
#to represent that, i will mirror what he did and find an inv gamma such that the
#probability > 15 is 0.01 and less than 1 is 0.01

#in terms of informtiveness of other priors, I will need to consider on what they
#represent. he did not bother too much with those beyond setting a reasonable sd 
#on their normal priors so i will see if that works




