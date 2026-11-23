#based on work in exploreRegression.R
here::i_am("lieRegression/logEuclid/explore/exploreCoverage.R")
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
    transi <- x[[i]]
    df$p11[i] <- transi[1, 1]
    df$p12[i] <- transi[1, 2]
    df$p13[i] <- transi[1, 3]
    df$p21[i] <- transi[2, 1]
    df$p22[i] <- transi[2, 2]
    df$p23[i] <- transi[2, 3]
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


#------------------------------------------------------------------------------------------------------------
#functions for simulation

#functionize simulation of observations
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


#functionize moving the cis back down to group as matrices
#treating both the upper and lower bound as a tau to be hatted and sent down to group
getGrpCi <- function(mod) {
  #calculate and extract
  ci <- confint(mod)
  ciLwr <- ci[, 1]
  ciUpr <- ci[, 2]

  #next two steps could also be done with tauToGrp()
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

#check parameter wise whether the ci captures true value
#check structurewise whether the ci captures the true matrix
checkGrpCover_se2 <- function(true, lwr, upr) {
  #there seems to be some switching of what is up lower and upper when the porjection
  #happens, so will just be checking between for the time being though this will potentially
  #need to be touched later

  #will check rotation mat diag, off diag, both tranlation params
  #the two rotation parameters may be inherently tied due to the orthog condition
  rDiag <- between(true[1, 1], lwr[1, 1], upr[1, 1]) |
    between(true[1, 1], upr[1, 1], lwr[1, 1])
  rOffDiag <- between(true[2, 1], lwr[2, 1], upr[2, 1]) |
    between(true[2, 1], upr[2, 1], lwr[2, 1])
  tx <- between(true[1, 3], lwr[1, 3], upr[1, 3]) |
    between(true[1, 3], upr[1, 3], lwr[1, 3])
  ty <- between(true[2, 3], lwr[2, 3], upr[2, 3]) |
    between(true[2, 3], upr[2, 3], lwr[2, 3])

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


#------------------------------------------------------------------------------------------------------------
#general set up

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


#------------------------------------------------------------------------------------------------------------
#single iteration set up

simi <- simulateTrans(
  transTrue_grp = transTrue_grp,
  tri1 = tri1,
  n = 100,
  sigma = matrix(c(.25, 0, 0, 0, .25, 0, 0, 0, .07), nrow = 3, byrow = TRUE),
  subtitle = "Mean  = 0, Var = Diag * (.25,.25,.07)"
)
modi <- performReg(simi)
cii <- getGrpCi(modi)
checkGrpCover_se2(transTrue_grp, cii$lwr, cii$upr)


#---------------------------------------------------------------------------------------------------------
#function for simulation

sim <- function(
  iters = 100,
  trueTrans = transTrue_grp,
  tri1_ = tri1,
  n = 100,
  sigma = matrix(c(.25, 0, 0, 0, .25, 0, 0, 0, .07), nrow = 3, byrow = TRUE)
) {

  holdCover <- list(
    rDiag = rep(NA, iters),
    rOffDiag = rep(NA, iters),
    tx = rep(NA, iters),
    ty = rep(NA, iters),
    all = rep(NA, iters)
  )

  for (i in 1:iters) {
    simi <- simulateTrans(
      transTrue_grp = trueTrans,
      tri1 = tri1_,
      n = n,
      sigma = sigma,
      subtitle = "SIMULATION"
    )
    modi <- performReg(simi)
    cii <- getGrpCi(modi)
    coveri <- checkGrpCover_se2(trueTrans, cii$lwr, cii$upr)

    #output
    holdCover$rDiag[i] <- coveri$rDiag
    holdCover$rOffDiag[i] <- coveri$rOffDiag
    holdCover$tx[i] <- coveri$tx
    holdCover$ty[i] <- coveri$ty
    holdCover$all[i] <- coveri$all
  }

  return(
    lapply(holdCover, mean)
  )
}

sim()


