library(dplyr)
library(ggplot2)
set.seed(826)


#resources:
#2d rigid motion
#https://www.cs.cornell.edu/courses/cs4620/2013fa/lectures/04transforms2d.pdf
#https://yuxng.github.io/Courses/CS6301Fall2023//lecture_04_2D_rigid_body_motions_rotation_matrices.pdf

#----------------------------------------------------------------------------------------------------------------------------
#helper functions


#function for rigid transformations
#theta is rotation angle
#xMove, yMove translation amounts
#rigid transformations in 2d take a single angle theta and rotate the point in that direction like the unit circle
transMaker <- function(theta, xMove, yMove) {
  #base matrix
  transi <- diag(3)
  #translation component
  transi[1:2, 3] <- c(xMove, yMove)
  #rotation component
  rotMat <- matrix(
    data = c(cos(theta), sin(theta), -sin(theta), cos(theta)),
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

p1 <- ggplot(mapping = aes(x = x, y = y)) |>
  plotTri(data = matToDf(tri1), color = "dodgerblue") |>
  plotTri(data = matToDf(tri1Trans1), color = "firebrick")

