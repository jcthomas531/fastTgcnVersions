data {
  int<lower=1> N_predict;
  real x_predict[N_predict];

  int<lower=1> N_obs;
  real y_obs[N_obs];
  int<lower=1, upper=N_predict> observed_idx[N_obs];

  
}

parameters {
  vector[N_predict] f_tilde;
  
  real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=0> sigma;
}

transformed parameters {

  matrix[N_predict, N_predict] cov =   cov_exp_quad(x_predict, alpha, rho)
                                     + diag_matrix(rep_vector(1e-10, N_predict));
  matrix[N_predict, N_predict] L_cov = cholesky_decompose(cov);
  
  vector[N_predict] f_predict = L_cov * f_tilde;
  
}

model {
  f_tilde ~ normal(0, 1);
  y_obs ~ normal(f_predict[observed_idx], sigma);
  
  
  rho ~ inv_gamma(4.6, 22.1);
  alpha ~ normal(0, 2);
  sigma ~ normal(0, 1);
  #the following formulation is used in the analytical files, assumes f is integrated
  #out and the observational noise already is captured in L_cov, but that is not applicable
  #to us here
  #y_obs ~ multi_normal_cholesky(rep_vector(0, N_obs), L_cov);
}

generated quantities {
  real y_predict[N_predict] = normal_rng(f_predict, sigma);
}
