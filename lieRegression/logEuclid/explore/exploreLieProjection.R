here::i_am("lieRegression/logEuclid/explore/exploreLieProjection.R")
library(here)
library(dplyr)
library(ggplot2)
library(expm) #for handling matrix exponential and log
set.seed(826)


#resources:
#2d rigid motion
#https://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/04transforms2d.pdf
#https://yuxng.github.io/Courses/CS6301Fall2023//lecture_04_2D_rigid_body_motions_rotation_matrices.pdf

#tools
toolDir <- here("lieRegression/tools")
source(here(toolDir, "makeRigidTrans_2d.R"))
source(here(toolDir, "checkAssum_se2_grp.R"))
source(here(toolDir, "checkAssum_se2_alg.R"))
source(here(toolDir, "vee_se2.R"))
source(here(toolDir, "hat_se2.R"))

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
      color = color,
      alpha = .4
    ) +
    geom_point(
      data = data,
      color = color
    )
}


#----------------------------------------------------------------------------------------------------------------------------

#specifying the triangle vertices as a matrix
#the first row is x, second row is y, third row is that dummy 1 that points need to interact with se(2) mats
#this form allows for easy application of transformation matrices

#initial triangle to play with
tri1 <- matrix(
  data = c(
    0,0,1,
    1,0,1,
    0,1,1
  ),
  ncol = 3
)

#create a transformation that is just translation
trans1 <- makeRigidTrans_2d(theta = 0, xMove = 2, yMove = 3)

#apply transformation to tri1
tri1Trans1 <- trans1 %*% tri1

#visualize
p1 <- ggplot(mapping = aes(x = x, y = y)) |>
  plotTri(data = matToDf(tri1), color = "dodgerblue") |>
  plotTri(data = matToDf(tri1Trans1), color = "firebrick") +
  coord_fixed(ratio = 1) +
  theme_bw()
p1


#transformation with rotation
trans2 <- makeRigidTrans_2d(theta = 30, xMove = 0, yMove = 2)
tri1Trans2 <- trans2 %*% tri1
p2 <- ggplot(mapping = aes(x = x, y = y)) |>
  plotTri(data = matToDf(tri1), color = "dodgerblue") |>
  plotTri(data = matToDf(tri1Trans2), color = "darkgreen") +
  coord_fixed(ratio = 1) +
  ylim(0,3) +
  xlim(-1,1.5) +
  theme_bw()
p2


#----------------------------------------------------------------------------
#moving the transformation from the lie group to the lie algebra
#applying the vee and hat functions
#checking all assumptions as it goes thru

#rename so it is easy to change to another trans and east to follow code
transGrp <- trans2

#transformation in lie group
transGrp
#check lie group se2 assumtions
checkAssum_se2_grp(transGrp)
#assumptions hold

#move transformation from lie group to lie algebra
transAlg <- logm(transGrp)
transAlg
#check lie algebra se2 assumptions
checkAssum_se2_alg(transAlg)
#assumptions hold

#use vee function to apply vectorizing basis to lie algebra transformation
tau <- vee_se2(transAlg)
tau

#use hat function to return vectorized transformation to lie algebra matrix form
tauHatted <- hat_se2(tau)
tauHatted
#check assumption that this should be the same as trans2_alg
Matrix::all.equal(transAlg, tauHatted)
#assumptions hold
#check lie algebra se2 assumptions
checkAssum_se2_alg(tauHatted) 
#assumptions hold

#move transformation from lie algebra to lie group
transGrp_back <- expm(tauHatted)
transGrp_back
#check lie group se2 assumptions
checkAssum_se2_grp(transGrp_back)
#assumptions hold
#check assumption that this transformation remains unchanged after this pocess
Matrix::all.equal(transGrp, transGrp_back)



#a couple of notes
#when logm() takes the group element to algebra, diagonal is not exactly 0. is this important?
#the log mapping is only exact around the identity and fails around 180 rotation, we will
#need to investigate how logm() handles these type of things and what happens behind the scenes



