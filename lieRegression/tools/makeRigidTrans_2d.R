
#function for making rigid transformations in 2d
#theta is rotation angle in degrees
#xMove, yMove translation amounts
#rigid transformations in 2d take a single angle theta and rotate the point in that direction like the unit circle
makeRigidTrans_2d <- function(theta, xMove, yMove) {
  #convert degrees to radians
  thetaRad <- theta * pi / 180
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