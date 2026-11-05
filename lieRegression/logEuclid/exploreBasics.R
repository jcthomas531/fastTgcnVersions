library(dplyr)
library(ggplot2)
set.seed(826)


#resources:
#2d rigid motion
#https://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/04transforms2d.pdf
#https://yuxng.github.io/Courses/CS6301Fall2023//lecture_04_2D_rigid_body_motions_rotation_matrices.pdf

#----------------------------------------------------------------------------------------------------------------------------
#helper functions

#change from degrees to radians
degToRad <- function(deg) {
  return(deg * pi / 180)
}

#function for rigid transformations
#theta is rotation angle in degrees
#xMove, yMove translation amounts
#rigid transformations in 2d take a single angle theta and rotate the point in that direction like the unit circle
transMaker <- function(theta, xMove, yMove) {
  #convert degrees to radians
  thetaRad <- degToRad(theta)
  #base matrix
  transi <- diag(3)
  #translation component
  transi[1:2, 3] <- c(xMove, yMove)
  #rotation component
  rotMat <- matrix(
    data = c(cos(thetaRad), sin(thetaRad), -sin(thetaRad), cos(thetaRad)),
    nrow = 2
  )
  transi[1:2, 1:2] <- rotMat

  return(transi)
}


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

#checking assumptions of SE(2) group
#wikipedia says that proper rigid trans have det = 1 but it is not clear if this property is necessary
#for what we are doing or even to be expected, it is not listed in sola 2018
#https://en.wikipedia.org/wiki/Rigid_transformation
checkAssum_se2_group <- function(x) {

  rot <- x[1:2,1:2]

  #check orthogonality
  rotrot <- t(rot) %*% rot
  #this vectorizes the matrix, checks equality within tolerence
  rotOrthog <- Matrix::all.equal(rotrot, diag(2))
  rotOrthogLogic <- ifelse(rotOrthog == TRUE, TRUE, FALSE)

  #check determinate
  rotDet <- det(rot)
  rotDetLogic <- ifelse(
    base::all.equal(rotDet, 1),
    TRUE,
    FALSE
  )

  return(
    list(
      #test results
      rotOrthogCheck = rotOrthogLogic,
      rotDetCheck = rotDetLogic,
      #specifics
      rotOrthogOutput = rotOrthog,
      rotOrthogDiff = rotrot - diag(2),
      rotDet = rotDet
    )
  )

}
#testing
# trans2 <- transMaker(theta = 30, xMove = 0, yMove = 2)
# checkAssum_se2_group(trans2)
# test1 <- matrix(rnorm(9), nrow = 3)
# checkAssum_se2_group(test1)


#function for checking form of matrix once it is lie algebra
#see sola 2018
checkAssum_se2_algebra <- function(x) {
  
  rot <- x[1:2,1:2]
  
  
  #check rotation portion is skew symmetric
  #checking all everything at once
  rotSkewSymm <- Matrix::all.equal(t(rot), -rot)
  rotSkewSymmLogic <- ifelse(rotSkewSymm == TRUE, TRUE, FALSE)
  #check diagonal is 0
  rotDiagLogic <- base::all.equal(diag(rot), rep(0, 2))

  return(
    list(
      rotSkewSymmCheck = rotSkewSymmLogic,
      rotSkewSymmOutput = rotSkewSymm,
      diag0Check = rotDiagLogic
    )
  )
  
}

#----------------------------------------------------------------------------------------------------------------------------

#specifying the triangle vertices as a matrix
#the first row is x, second row is y, third row is that dummy 1
#this orientation allows for easy application of transformation matrices

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
trans1 <- transMaker(theta = 0, xMove = 2, yMove = 3)

#apply transformation to tri1
tri1Trans1 <- trans1 %*% tri1

#visualize
p1 <- ggplot(mapping = aes(x = x, y = y)) |>
  plotTri(data = matToDf(tri1), color = "dodgerblue") |>
  plotTri(data = matToDf(tri1Trans1), color = "firebrick") +
  theme_bw()



#transformation with rotation
trans2 <- transMaker(theta = 30, xMove = 0, yMove = 2)
tri1Trans2 <- trans2 %*% tri1
p2 <- p1 |>
  plotTri(data = matToDf(tri1Trans2), color = "darkgreen")
p2


#------------------------------------------------------------------------------------------------------------------------
#adding error to the transformation
#THIS MAY NOT BE THE PROPER WAY TO HANDLE ERROR
#this may give rise to valid transformation matrices but it is not the error for the assumed model
#that error is given in the lie algebra

simulateTrans <- function(
  n, thetaMean = 30, thetaSd = 1, 
  xMean = 3, xSd = 1,
  yMean = 0, ySd = 1
){

  #holder to the trans matrices
  transHolder <- vector(mode = "list", length = n)

  

  #generate transformation matrix with error
  for (i in 1:n) {
    #params with error
    thetai <- rnorm(1, mean = thetaMean, sd = thetaSd)
    xi <- rnorm(1, mean = xMean, sd = xSd)
    yi <- rnorm(1, mean = yMean, sd = ySd)

    #create trans matrix
    transHolder[[i]] <- transMaker(
      theta = thetai,
      xMove = xi,
      yMove = yi
    )
    print(thetai)
  }

  return(transHolder)
}


transTrue <- transMaker(theta = 30, xMove = 3, yMove = 0)
tri1TransTrue <- transTrue %*% tri1

#remeber that theta is in terms of degrees so its sd must be larger
transError <- simulateTrans(n = 2, thetaSd = 10, xSd = .1, ySd = .1) 
tri1TransError <- lapply(transError, function(x) {x %*% tri1})

pError <- ggplot(mapping = aes(x = x, y = y)) |>
  plotTri(data = matToDf(tri1), color = "black") |>
  plotTri(data = matToDf(tri1TransTrue), color = "black") |>
  plotTri(data = matToDf(tri1TransError[[1]]), color = "dodgerblue") |>
  plotTri(data = matToDf(tri1TransError[[2]]), color = "dodgerblue") +
  xlim(-5,5) +
  ylim(-5,5)
pError
#------------------------------------------------------------------------------------------------------------------------


#moving the transformation from the lie group to the lie algebra
#as the log mapping is onto but not one-to-one, we will need to investigate how this
#method handles multiple mappings
library(expm)
trans2_up <- logm(trans2)
#check properties in the lie algebra
checkAssum_se2_algebra(trans2_up)
#bringing it back down to the lie group
trans2_down <- expm(trans2_up)
trans2
#it seems that the projection up and back down reasonably preserves the transformation matrix
#we will want to investigate if that remains true with matrices farther from the identity
Matrix::all.equal(trans2, trans2_down)
#we will want to investigate these assumptions as we get farther from the identity as well
checkAssum_se2_group(trans2)
checkAssum_se2_group(trans2_down)

#a quick check with larger transformations
#seems to still be working
trans3 <- transMaker(theta = 95, xMove = 100, yMove = 100)
trans3_up <- logm(trans3)
trans3_down <- expm(trans3_up)
Matrix::all.equal(trans3, trans3_down)
checkAssum_se2_group(trans3)
checkAssum_se2_group(trans3_down)





#vee function to take matrix in lie algebra and make it into a vector
#matrices in lie algebra have specific form, see sola 2018, so the basis can be hard coded
#using form from sola 2018
vee_se2 <- function(x) {
  #basing this off SO(3) formulation in example 3 in sola 2018 and table 1
  theta <- x[2,1]
  rho <- x[1:2,3]
  return(c(rho, theta))
}
#testing
# trans3_up
# vee_se2(trans3_up)

#hat function to take vector representation in lie algebra back to matrix representation
hat_se2 <- function(x) {
  
}



