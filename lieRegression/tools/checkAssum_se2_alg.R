#function for checking form of matrix once it is lie algebra
#see sola 2018
checkAssum_se2_alg <- function(x) {
  
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
#SEE exploreLieProjection.R FOR USAGE