#vee function to take matrix in lie algebra and make it into a vector
#matrices in lie algebra have specific form, see sola 2018, so the basis can be hard coded
#using form from sola 2018
#NOTE when a matrix is moved up to the algebra, the off diagonals of the so(2) portion are not *perfectly* 0
#this is not retained here
vee_se2 <- function(x) {
  #basing this off SO(3) formulation in example 3 in sola 2018 and table 1
  theta <- x[2,1]
  rho <- x[1:2,3]
  return(c(rho, theta))
}
#testing
#SEE exploreLieProjection.R FOR USAGE
# trans3_up
# tau <- vee_se2(trans3_up)
# tau