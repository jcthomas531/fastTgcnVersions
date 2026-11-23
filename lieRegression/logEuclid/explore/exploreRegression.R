here::i_am("lieRegression/logEuclid/explore/exploreRegression.R")
library(here)
library(dplyr)
library(ggplot2)
library(expm) #for handling matrix exponential and log
library(mvtnorm)
set.seed(826)

#tools
toolDir <- here("lieRegression/tools")
source(here(toolDir, "makeRigidTrans_2d.R"))
source(here(toolDir, "checkAssum_se2_grp.R"))
source(here(toolDir, "checkAssum_se2_alg.R"))
source(here(toolDir, "vee_se2.R"))
source(here(toolDir, "hat_se2.R"))


#NOTES
#grant note
#projecting down bayesian samples instead of the confidence interval


#----------------------------------------------------------------------------------------------------------------------------
#helper functions

#function to make the matrix into a data frame for graphing and descriptives
matToDf <- function(mat) {
  df <- mat |>
    t() |>
    as.data.frame()
  colnames(df) <- c("x", "y", "one")
  df$vert <- paste0("v", 1:nrow(df))
  return(df)
}

#plotting function for triangles, see application later in file
plotTri <- function(x, data, color) {
  x +
    geom_polygon(
      data = data,
      fill = color,
      color = color
    ) +
    geom_point(
      data = data,
      color = color
    )
}

plotTri2 <- function(x, data, color, alpha = .4) {
  alpha2 <- alpha + .1
  x +
    geom_polygon(
      data = data,
      fill = color,
      color = alpha(color, alpha2),
      alpha = alpha
    ) +
    geom_point(
      data = data,
      color = color,
      alpha = alpha2
    )
}


extractPositions <- function(x) {
  df <- data.frame(
    p11 = rep(NA, length(x)),
    p12 = NA,
    p13 = NA,
    p21 = NA,
    p22 = NA,
    p23 = NA
  )
  for (i in 1:length(x)) {
    transi = x[[i]]
    df$p11[i] = transi[1,1]
    df$p12[i] = transi[1,2]
    df$p13[i] = transi[1,3]
    df$p21[i] = transi[2,1]
    df$p22[i] = transi[2,2]
    df$p23[i] = transi[2,3]
  }
  return(df)
}

grpToTau <- function(T) {
  #move transformation to lie algebra
  T_alg <- logm(T)
  #vectorize with vee function
  tau <- vee_se2(T_alg)
  return(tau)
}

tauToGrp <- function(tau) {
  #make tau into matrix form in algebra
  T_alg <- hat_se2(tau)
  #move down to lie group
  T <- expm(T_alg)
  return(T)
}

#----------------------------------------------------------------------------------------------------------------------------
#adding noise to true transformation in the lie algebra
#transformation transTrueGrp in se2 lie group moves triangle from its position at tri1 to its position at tri2

#specifying the triangle vertices as a matrix
#the first row is x, second row is y, third row is that dummy 1 that points need to interact with se(2) mats
#this form allows for easy application of transformation matrices

#triangle at initial location
tri1 <- matrix(
  data = c(
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    1
  ),
  ncol = 3
)

#true transformation
transTrue_grp <- makeRigidTrans_2d(theta = 30, xMove = 0, yMove = 2)

#triangle at subsequent location
tri2 <- transTrue_grp %*% tri1

#visualize
p1 <- ggplot(mapping = aes(x = x, y = y)) |>
  plotTri(data = matToDf(tri1), color = "dodgerblue") |>
  plotTri(data = matToDf(tri2), color = "green4") +
  coord_fixed(ratio = 1) +
  theme_bw()
p1

#move transformation to lie algebra
transTrue_alg <- logm(transTrue_grp)

#vectorize with vee function
tauTrue <- vee_se2(transTrue_alg)

#make many observations of tau with normal error
randNoise <- rmvnorm(100, mean = rep(0, 3))
tauObs <- apply(randNoise, 1, function(x) {
  x + tauTrue
}) |>
  t()
#asplit makes the rows of the matrix into a list
tauObsList <- asplit(tauObs, MARGIN = 1)

#put into algebra matrix form
tauObs_alg <- lapply(tauObsList, hat_se2)
#check assumptions
all(
  sapply(tauObs_alg, function(x) {
    checkAssum_se2_alg(x)$rotSkewSymmCheck
  })
)

#move back down to group
transObs_grp <- lapply(tauObs_alg, expm)
#check assumptions
all(
  sapply(transObs_grp, function(x) {
    checkAssum_se2_grp(x)$rotOrthogCheck
  })
)


#apply transformations to triangle at original position
tri2Obs <- lapply(transObs_grp, function(x) {
  x %*% tri1
})

#stack all transformations and dfs into a large df for plotting
tri2ObsDf <- lapply(tri2Obs, matToDf) |>
  bind_rows(.id = "obs")


#noise here seems really large potentially bc the rotation and translations are on different scales
obsPlot <- ggplot(mapping = aes(x = x, y = y)) +
  geom_polygon(
    data = tri2ObsDf,
    mapping = aes(group = obs), #one polygon per observation
    fill = "dodgerblue4",
    color = alpha("dodgerblue4", .2), #geom_polygon alpha only affects fill, so fade the outline directly
    alpha = .1
  )
simPlot <- obsPlot |>
  plotTri2(data = matToDf(tri1), color = "black", alpha = .8) |>
  plotTri2(data = matToDf(tri2), color = "green4", alpha = .8) +
  coord_fixed(ratio = 1) +
  theme_bw() +
  labs(
    title = "100 simulated trans. with mv normal error in algebra",
    subtitle = "Mean  = 0, Var = Identity"
  )
simPlot


#----------------------------------------------------------------------------------------------------------------------------
#functionizing above process

simulateTrans <- function(
  transTrue_grp,
  tri1,
  n,
  mean = c(0, 0, 0),
  sigma = diag(3),
  subtitle = "subtitle"
) {
  #triangle at subsequent location
  tri2 <- transTrue_grp %*% tri1

  #move transformation to lie algebra
  transTrue_alg <- logm(transTrue_grp)

  #vectorize with vee function
  tauTrue <- vee_se2(transTrue_alg)

  #make many observations of tau with normal error
  randNoise <- rmvnorm(n, mean = mean, sigma = sigma)
  tauObs <- apply(randNoise, 1, function(x) {
    x + tauTrue
  }) |>
    t()
  #asplit makes the rows of the matrix into a list
  tauObsList <- asplit(tauObs, MARGIN = 1)

  #put into algebra matrix form
  tauObs_alg <- lapply(tauObsList, hat_se2)

  #move back down to group
  transObs_grp <- lapply(tauObs_alg, expm)

  #apply transformations to triangle at original position
  tri2Obs <- lapply(transObs_grp, function(x) {
    x %*% tri1
  })

  #stack all transformations and dfs into a large df for plotting
  tri2ObsDf <- lapply(tri2Obs, matToDf) |>
    bind_rows(.id = "obs")
  #plot
  obsPlot <- ggplot(mapping = aes(x = x, y = y)) +
    geom_polygon(
      data = tri2ObsDf,
      mapping = aes(group = obs), #one polygon per observation
      fill = "dodgerblue4",
      color = alpha("dodgerblue4", .2), #geom_polygon alpha only affects fill, so fade the outline directly
      alpha = .1
    )
  simPlot <- obsPlot |>
    plotTri2(data = matToDf(tri1), color = "black", alpha = .8) |>
    plotTri2(data = matToDf(tri2), color = "green4", alpha = .8) +
    coord_fixed(ratio = 1) +
    theme_bw() +
    labs(
      title = "100 simulated trans. with mv normal error in algebra",
      subtitle = subtitle
    )

  return(
    list(
      transTrue_grp = transTrue_grp,
      tri1 = tri1,
      tri2 = tri2,
      transObs_grp = transObs_grp,
      tri2Obs = tri2Obs,
      simPlot = simPlot,
      n = n,
      mean = mean,
      sigma = sigma
    )
  )
}

#----------------
#recreate what was done above with the function
noiseSim1 <- simulateTrans(
  transTrue_grp = transTrue_grp,
  tri1 = tri1,
  n = 100,
  subtitle = "Mean  = 0, Var = Identity"
)
noiseSim1$simPlot

#----------------
#reducing sigma
noiseSim2 <- simulateTrans(
  transTrue_grp = transTrue_grp,
  tri1 = tri1,
  n = 100,
  sigma = diag(3) * .25,
  subtitle = "Mean  = 0, Var = Diag * 0.25"
)
noiseSim2$simPlot
#because rotation is in radians, the scale is very different from translation

#looking at the "mean" transformation
noiseSim2Vals <- extractPositions(noiseSim2$transObs_grp)
noiseSim2Means <- apply(noiseSim2Vals, 2, mean)
transObsMean_grp <- matrix(
  data = c(noiseSim2Means, 0, 0, 1),
  byrow = TRUE,
  ncol = 3
)
checkAssum_se2_grp(transObsMean_grp)
#this mean transformation does NOT meet the assumptions for being se2
#and thus we see the importance of working in the algebra rather than the group

#----------------
#smaller sigma for rotation than for translation
noiseSim3 <- simulateTrans(
  transTrue_grp = transTrue_grp,
  tri1 = tri1,
  n = 100,
  sigma = matrix(c(.25,0,0, 0,.25,0, 0,0,.07), nrow = 3, byrow = TRUE),
  subtitle = "Mean  = 0, Var = Diag * (.25,.25,.07)"
)
noiseSim3$simPlot

#----------------------------------------------------------------------------------------------------------------------------
#regression
#using values from sim3


sim3TauObs <- lapply(noiseSim3$transObs_grp, grpToTau)
sim3TauDf <- do.call(rbind, sim3TauObs) |> as.data.frame()
colnames(sim3TauDf) <- c("rho1", "rho2", "theta")


sim3Mod1 <- lm(cbind(rho1, rho2, theta) ~ 1, data = sim3TauDf)
summary(sim3Mod1)
estVar(sim3Mod1)

#we can obtain an estimate and a confidence interval for each value in tau
#but because tau is used to make a matrix, what does in mean to have a confidence 
#interval for a matrix? what does it mean to be "between two matrices"?


#----------------------------------
#projecting the confidence interval down 
#treating both the upper and lower bound as a tau to be hatted and set down to group

#calculate and extract
sim3Mod1_ci <- confint(sim3Mod1)
s3m1_ciLwr <- sim3Mod1_ci[,1]
s3m1_ciUpr <- sim3Mod1_ci[,2]

#hat
s3m1_ciLwrH <- hat_se2(s3m1_ciLwr)
s3m1_ciUprH <- hat_se2(s3m1_ciUpr)
#check assumptions
checkAssum_se2_alg(s3m1_ciLwrH)
checkAssum_se2_alg(s3m1_ciUprH)

#project to group
s3m1_ciLwr_grp <- expm(s3m1_ciLwrH)
s3m1_ciUpr_grp <- expm(s3m1_ciUprH)
#check assumptions
checkAssum_se2_grp(s3m1_ciLwr_grp)
checkAssum_se2_grp(s3m1_ciUpr_grp)

#check parameter wise whether the ci captures true value
#check structurewise whether the ci captures the true matrix
checkGrpCover_se2 <- function(true, lwr, upr) {
  #there seems to be some switching of what is up lower and upper when the porjection
  #happens, so will just be checking between for the time being though this will potentially
  #need to be touched later

  #will check rotation mat diag, off diag, both tranlation params
  #the two rotation parameters may be inherently tied due to the orthog condition
  rDiag <- between(true[1,1], lwr[1,1], upr[1,1]) | between(true[1,1], upr[1,1], lwr[1,1])  
  rOffDiag <- between(true[2,1], lwr[2,1], upr[2,1]) | between(true[2,1], upr[2,1], lwr[2,1])
  tx <- between(true[1,3], lwr[1,3], upr[1,3]) | between(true[1,3], upr[1,3], lwr[1,3])
  ty <- between(true[2,3], lwr[2,3], upr[2,3]) | between(true[2,3], upr[2,3], lwr[2,3])
  
  return(
    list(
      rDiag = rDiag,
      rOffDiag = rOffDiag,
      tx = tx,
      ty = ty,
      all = all(c(rDiag, rOffDiag, tx, ty))
    )
  )
}
checkGrpCover_se2(true = transTrue_grp, lwr = s3m1_ciLwr_grp, upr = s3m1_ciUpr_grp)



#---------------------------------------------------------------------------------
#functionizing regression
#returns the regression object in the algebra
performReg <- function(sim) {
  #move simulated observations up to algebra and format
  tauObs <- lapply(sim$transObs_grp, grpToTau)
  tauDf <- do.call(rbind, tauObs) |> as.data.frame()
  colnames(tauDf) <- c("rho1", "rho2", "theta")

  #perform regression
  mod <- lm(cbind(rho1, rho2, theta) ~ 1, data = tauDf)

  return(mod)
}

#------------------------------------------------------------------------------
#functionize moving the cis back down to group as matrices
#treating both the upper and lower bound as a tau to be hatted and sent down to group
getGrpCi <- function(mod) {
  #calculate and extract
  ci <- confint(mod)
  ciLwr <- ci[,1]
  ciUpr <- ci[,2]

  #hat
  ciLwrH <- hat_se2(ciLwr)
  ciUprH <- hat_se2(ciUpr)

  #project to group
  ciLwr_grp <- expm(ciLwrH)
  ciUpr_grp <- expm(ciUprH)

  return(
    list(
      lwr = ciLwr_grp,
      upr = ciUpr_grp
    )
  )
}



#attempting with functions
s3m1 <- performReg(noiseSim3)
s3ci1 <- getGrpCi(s3m1)
checkGrpCover_se2(transTrue_grp, s3ci1$lwr, s3ci1$upr)
