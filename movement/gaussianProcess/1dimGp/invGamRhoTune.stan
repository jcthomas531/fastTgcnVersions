functions {
  vector tail_delta(vector y, vector theta, real[] x_r, int[] x_i) {
    vector[2] deltas;
    //0.01 is the targeted probability, 0.2 added to param to avoid 0 value
    deltas[1] = inv_gamma_cdf(theta[1] + .2, exp(y[1]), exp(y[2])) - 0.01; 
    //0.01 is the targeted probability, 0.2 added to param to avoid 0 value
    deltas[2] = 1 - inv_gamma_cdf(theta[2] + .2, exp(y[1]), exp(y[2])) - 0.01; 
    return deltas;
  }
}

transformed data {
  real l = 1; //lower target 
  real u = 15; //upper target
  vector[2] theta = [l, u]';
    
  real delta = 1;
  real a = square(delta * (u + l) / (u - l)) + 2;
  real b =  ((u + l) / 2) * ( square(delta * (u + l) / (u - l)) + 1);
  vector[2] y_guess = [log(a), log(b)]';
  
  real x_r[0];
  int x_i[0];
  
  vector[2] y = algebra_solver(tail_delta, y_guess, theta, x_r, x_i);

  print("a = ", exp(y[1]));
  print("b = ", exp(y[2]));
}
