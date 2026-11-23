#checking assumptions of SE(2) group
#wikipedia says that proper rigid trans have det = 1 but it is not clear if this property is necessary
#for what we are doing or even to be expected, it is not listed in sola 2018
#https://en.wikipedia.org/wiki/Rigid_transformation
checkAssum_se2_grp <- function(x) {

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
#SEE exploreLieProjection.R FOR USAGE
# trans2 <- makeRigidTrans_2d(theta = 30, xMove = 0, yMove = 2)
# checkAssum_se2_grp(trans2)
# test1 <- matrix(rnorm(9), nrow = 3)
# checkAssum_se2_grp(test1)