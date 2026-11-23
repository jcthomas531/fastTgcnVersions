#hat function to take vector representation in lie algebra back to matrix representation
#NOTE when a matrix is moved up to the algebra, the off diagonals of the so(2) portion are not *perfectly* 0
#this is not retained here
hat_se2 <- function(x) {
  mat <- matrix(0, nrow = 3, ncol = 3)
  mat[1,3] <- x[1]
  mat[2,3] <- x[2]
  mat[1,2] <- -x[3]
  mat[2,1] <- x[3]
  return(mat)
}
#testing
#SEE exploreLieProjection.R FOR USAGE
# trans3_up
# tau <- vee_se2(trans3_up)
# tau_hatted <- hat_se2(aaa) #same thing as the element of g (the lie algebra)
# Matrix::all.equal(trans3_up, tau_hatted)