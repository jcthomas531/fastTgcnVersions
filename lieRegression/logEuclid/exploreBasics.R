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



#adding error to the transformation
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
