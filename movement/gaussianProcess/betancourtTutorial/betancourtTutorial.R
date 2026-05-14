#following tutorial
#https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#1_Modeling_Functional_Relationships
here::i_am("gaussianProcess/betancourtTutorial/betancourtTutorial.R")
library(here)
#in this tutorial we will be be using the exponentiated quadratic covariance function
#this utilizes parameters alpha and rho
betanPath <- here("gaussianProcess/betancourtTutorial")
#loading in his functions
source(here(betanPath, "stan_utility.R"))
source(here(betanPath, "gp_utility.R"))
c_dark_trans <- "maroon"
c_green_trans <- "green"
#####
#some general set up
#####
library(rstan)
rstan_options(auto_write = FALSE)
set.seed(826)

#parallelization
options(mc.cores = parallel::detectCores())
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")

#for some reason setting up an environment
util <- new.env()

#setting up graphical parameters
# par(family="CMU Serif", las=1, bty="l", cex.axis=1, cex.lab=1, cex.main=1,
#     xaxs="i", yaxs="i", mar = c(5, 5, 3, 5))

#####
#define covariate grid
#####
N = 551
x <- 22 * (0:(N - 1)) / (N - 1) - 11

#####
#set true params for covariate function
#####
alpha_true <- 3
rho_true <- 5.5
#pack all of the data together for simulation
simu_data <- list(alpha=alpha_true, rho=rho_true, N=N, x=x)


#####
#sample from the Gaussian process, 2.1
#####
#this entails constructing the Gram (ie realized) matrix and then sampling from the corresponding
#multivariate normal 
#this is done thru stan though it seems to me this is an unnecessary complication
#stan file: simu.stan


#function for creating the exponentiated quad covariance matrix
#obviously not very efficient bc i am using a nested for loop and the matrix will
#be symmetric but this gets the job done for now
expQuadCov <- function(x, alpha, rho) {
  
  #set up empty covariance matrix
  n <- length(x)
  covMat <- matrix(nrow = n, ncol = n)
  
  for (i in 1:n) {
    
    xi <- x[i]
    
    for (j in 1:n) {
      
      xj <- x[j]
      
      #produce the i, j value of the covariance matrix
      covA <- (abs(xi - xj)/rho)^2
      covB <- exp((-1/2)*covA)
      covMat[i, j] <- (alpha^2) * covB
      
      
    }
    
  }
  return(covMat)
}

#generate covariance matrix for the space of interested given the true values of 
#the covariance parameters
#the tutorial also adds a nugget (ie some jitter) to this matrix
#i would think that this would involve adding some noise, however, it seems that we 
#are just adding the same small quantity to the diags
#also in my work here, i am not doing a decomp like he did, but rmvnorm does have
#the capibility to do that
cov1 <- expQuadCov(x = x, alpha = alpha_true, rho = rho_true) + diag(rep(1e-10, N))
#draw 4000 samples of the gaussian process
library(mvtnorm)
gp1 <- rmvnorm(n = 4000, sigma = cov1)
#visualize these draws to make sure they are similar to what was done in the tutorial
plot(x, gp1[1,], type = "l", ylim = c(-10, 10))
for(i in 2:100) {
  lines(x, gp1[i,])
}



#####
#simulating data from a gaussian process model, 2.2
#####
#consider using the gaussian process as described above as the mean function of
#a normal model
#this is done thru stan though it seems to me this is an unnecessary complication
#stan file: simu.stan

sigma_true <- 2

#for each sample of the gaussian process taken above, sample points from a normal 
#model using this as the mean function

yGen <- function(gp, sigma) {
  
  #set up
  #number of pints to sample from a single gp
  n <- ncol(gp)
  #number of gps
  gpN <- nrow(gp)
  
  #generate the data in the same format as the gps are in
  yMat <- matrix(nrow = gpN, ncol = n)
  for (i in 1:gpN) {
    yMat[i,] <- rnorm(n, mean = gp[i,], sd = sigma)
  }
  
  return(yMat)
  
}
#generate some data from the normal model using the gps as the mean function
yMat <- yGen(gp = gp1, sigma = sigma_true)

#lets look at some realizations from this data generating prcess
plotGpNormSim <- function(ind = 1, gp, y) {
  #realization of the gp that serves as the mean function
  gpi <- gp[ind,]
  #realization of the y data using the gp as the mean function
  yi <- y[ind,]
  
  
  plot(x, gpi, type = "l", ylim=c(-10, 10), 
       xlab = "x", ylab = "y", main = paste("simulation", ind))
  points(x, yi, col = "dodgerblue")
}

plotGpNormSim(ind = 7, gp = gp1, y = yMat)


#lets use the first simulation as what we are interested in moving forward
y <- yMat[7,]
gp <- gp1[7,]

#in practice, we would only observe a subset of these points, not the whole line
#something like this
observed_idx  <- c(50 * (0:10) + 26)
N_obs = length(observed_idx)
x_obs <- x[observed_idx]
y_obs <- y[observed_idx]
#naming things the same as he has named them
N_predict <- N
x_predict <- x
y_predict <- y
f <- gp
#he dumps these thing into a rdump file
#i think if you just store these as named lists it will work as well
stan_rdump(c("N_obs", "x_obs", "y_obs", "N_predict", "x_predict", "y_predict", "observed_idx"),
           file=here(betanPath, "normal.data.R")) 
stan_rdump(c("f", "x"), file=here(betanPath, "gp.truth.R"))


plot(x, gp, type = "l", ylim=c(-10, 10), 
     xlab = "x", ylab = "y", main = "simulation 1, observed")
points(x_obs, y_obs, col = "firebrick")



#####
#fitting a general gaussian process posterior, 2.3
#####
#this is using fixed values for the covariance function
#due to some reason that is not clear to me, there are computational issues
#with calculating the posterior from this method and we need to use a non-cnetered
#parameterization of this, whih is not challenging given it is a normal distribution
#and cetered at 0
#stan files: fit_normal.stan, fit_normal_ncp.stan
truth <- read_rdump(here(betanPath, "gp.truth.R"))
data <- read_rdump(here(betanPath, "normal.data.R"))
data$alpha <- alpha_true
data$rho <- rho_true
data$sigma <- sigma_true

normal_fit <- stan(file = here(betanPath, "fit_normal_ncp.stan"), data = data,
                   seed = 826, refresh = 1000)

normalFitDraws <- extract(normal_fit)


plot(normalFitDraws$f_predict[1,]~x, type = "l", ylim=c(-10, 10), 
     main = "some posterior function realizations from fixed cov matrix")
for(i in 2:100) {
  lines(x, normalFitDraws$f_predict[i,])
}
lines(gp~x, col = "red", lwd = 2)



#####
#Simulating From An Analytic Gaussian Process Posterior, 2.4
#####
#i beleive that this is saying that bc we have the normal-normal conjugacy we do 
#not actually need to run a sampler but can calculate the posterior analytically
#i am not going to walk through this example

#stan file: fit_normal_anal.stan

################################################################################
#here is where we transition from make believe land where we know the params for
#the covariance function to situations more like real life where we need to estimate
#these values
################################################################################


#####
#3.1 maximum marginal likelihood estimation
#####

#this section is very interesting, the model specified by the stan file does not
#actually do anything bayesian. there are no priors set on the parameters, only
#the model structure is laid out. then by using the optimizing() function, an approximation
#of the marginal mles (not sure about this "marginal" distinction) for the parameters
#in the covariance function (and the normal model) are estimated. They are then 
#extracted and simply used in the model from the previous section similar to how the
#true values were passed in last time. results are poor

#stan file: fit_covar1.stan


#the same thing is done with a different seed to show the inconsistency of this method
#strong sensitivity of the mmle might suggest degeneracy of the mmle function, which
#we investigate in the next section with monte carlo

#####
#3.2 exploring the marginal likelihood function
#####

#he states that MCMC and especially hamiltonian MCMC are good tools for investigating
#features of a likelihood function regardless of inference desires, this is something
#good to keep in mind

#also, apparently, when no prior was specified in the above stan file, an improper
#uniform prior is assumed. not sure what effects that has on the mle-like procedure
#that is used

#essentially the issue with our estimation was that we only have information on 
#distances up to the two farthest points in our observed data and we are try to
#use this to inform about all distances

#3.2.2 has some really interesting points about how our oberserved data should
#influence the range of our covariates

#in attempt to fix the issue above, we set reasonable priors on the covariance param
#and the param for the y variance. he does this built on his analytical sampling
#script that was used before: "fit_normal_anal.stan". I will also make the corresponing
#cahnges to the non-analytical one: "fit_normal_ncp.stan" bc that is what i would 
#prefer to use as i dont want to spend the time disecting the conjugacy rn
#as long as they produce similar results, we are in good shape

#using the analytical base: fit_covar2.stan
#using the MCMC base: fit_covar2MCMC.stan

#note, when the "truth" up above is defined as yMat[7,] and gp1[7,] I observe the 
#same trends as he does in the tutorial, if the index is changed to 1, everything
#is well behaved and we do not see any issues

#analytical
fitA <- stan(file = here(betanPath, "fit_covar2.stan"), data = data, seed = 826, refresh = 1000)
drawA <- extract(fitA)
#diagnostics
#depending on what index is chosen for truth above, there may or may not be 
#divergent transisitons
check_all_diagnostics(fitA)
#visuals
plot(drawA$f_predict[1,]~x, type = "l", ylim=c(-10, 10), 
     main = "some posterior function realizations from algorithm cov mat")
for(i in 2:100) {
  lines(x, drawA$f_predict[i,])
}
lines(gp~x, col = "red", lwd = 2)
#looking at parameter values
#green points only will show up if there are divergent transitions
partition <- partition_div(fitA)
div_params <- partition[[1]]
nondiv_params <- partition[[2]]

par(mfrow=c(1, 3))
plot(log(nondiv_params$rho), log(nondiv_params$alpha),
     col=c_dark_trans, pch=16, cex=0.8, xlab="log(rho)", ylab="log(alpha)")
points(log(div_params$rho), log(div_params$alpha),
       col=c_green_trans, pch=16, cex=0.8)

plot(log(nondiv_params$rho), log(nondiv_params$sigma),
     col=c_dark_trans, pch=16, cex=0.8, xlab="log(rho)", ylab="log(sigma)")
points(log(div_params$rho), log(div_params$sigma),
       col=c_green_trans, pch=16, cex=0.8)

plot(log(nondiv_params$alpha), log(nondiv_params$sigma),
     col=c_dark_trans, pch=16, cex=0.8, xlab="log(alpha)", ylab="log(sigma)")
points(log(div_params$alpha), log(div_params$sigma),
       col=c_green_trans, pch=16, cex=0.8)
par(mfrow=c(1, 1))

#this one will only work if some sigma draws are below 0.5
plot_low_sigma_gp_post_realizations(fitA, data, truth,
                                    "Posterior Realizations with sigma < 0.5")
plot_gp_post_realizations(fitA, data, truth,
                          "Posterior Realizations")
plot_gp_post_quantiles(fitA, data, truth,
                       "Posterior Marginal Quantiles")
#i observe some of the issues that he is describing, but not as extreme
#it seems to be highly dependent on what the truth is





#mcmc
#tried with just a few iterations, could be improved
# fitMcmc <- stan(file = "fit_covar2MCMC.stan", data = data, seed = 826, refresh = 1000,
#                 chains = 1, iter = 100) #try at lower levels first
# save(fitMcmc, file = "fitMcmc.RData")
load(here(betanPath, "fitMcmc.RData"))
drawMcmc <- extract(fitMcmc)
plot(drawMcmc$f_predict[1,]~x, type = "l", ylim=c(-10, 10), 
     main = "some posterior function realizations from mcmc cov mat")
for(i in 2:nrow(drawMcmc$f_predict)) {
  lines(x, drawMcmc$f_predict[i,])
}
lines(gp~x, col = "red", lwd = 2)
#based on comparison with the version produced in the tutorial, my tweak here seems to work
plot_low_sigma_gp_post_realizations(fitMcmc, data, truth,
                                    "Posterior Realizations with sigma < 0.5")
plot_gp_post_realizations(fitMcmc, data, truth,
                          "Posterior Realizations")
plot_gp_post_quantiles(fitMcmc, data, truth,
                       "Posterior Marginal Quantiles")
#this model takes a long time to fit so I have only done a few iteratione here,
#that is potentially why some of these graphic look different than the analytic version

#I HAVE FOUND HIS FUNCTIONS: https://github.com/betanalpha/knitr_case_studies/tree/master/gaussian_processes

#the reason for the issues he is describing is the same as what was happening before
#with the distances larger than what were observed, except now it is due to distances
#being smaller than what was observed





#####
#3.2.3 informative prior model
#####

#we want to supress both infinity and zero as distances
#to do this we will use inverse gammas as the prior

#we want a particular inverse gamma distribution that suppresses how we would like
#in particular, we want to find the parameterization that puts a cumulative prob 
#of 0.01 on all values less than a value l (in this case 2) and cumulative prob of 0.01 on all values
#over a value u (in this case 20)

#we can use stans algebraic solver for this, i dont totally understand his script
#for this but it is solving for inv gamma params a and b

#stan script: prior_tune.stan

fitInvGam <- stan(file=here(betanPath, 'prior_tune.stan'), iter=1, warmup=0, chains=1,
            seed=5838298, algorithm="Fixed_param")
#i bet there is a way to extract those a and b values from the stan object but
#for now I am going to hard code them
aInvGam <- 4.6
bInvGam <- 22.1
#lets take a look at the distribution it produces
c_dark <- c("#8F2727")
c_dark_highlight <- c("#7C0000")

lambda <- seq(0, 25, 0.001)
dens <- lapply(lambda, function(l) dgamma(1 / l, 4.6, rate=22.1) * (1 / l**2))
plot(lambda, dens, type="l", col=c_dark_highlight, lwd=2,
     xlab="lambda", ylab="Prior Density", yaxt='n')

lambda98 <- seq(2, 20, 0.001)
dens <- lapply(lambda98, function(l) dgamma(1 / l, 4.6, rate=22.1) * (1 / l**2))
lambda98 <- c(lambda98, 20, 2)
dens <- c(dens, 0, 0)

polygon(lambda98, dens, col=c_dark, border=NA)
#lets check that it meets our needs
library(invgamma)
pinvgamma(2, aInvGam, rate = bInvGam)
pinvgamma(20, aInvGam, rate = bInvGam, lower.tail = FALSE)

#lets use this as our prior on rho

#analytic stan file: fit_covar3.stan 
#mcmc stan file: fit_covar3MCMC.stan

#analytic
fitInform <- stan(file=here(betanPath, 'fit_covar3.stan'), data=data,
            seed=5838298, refresh=1000)
drawInform <- extract(fitInform)
#visuals
plot(drawInform$f_predict[1,]~x, type = "l", ylim=c(-10, 10), 
     main = "some posterior function realizations from algorithm cov mat")
for(i in 2:200) {
  lines(x, drawInform$f_predict[i,])
}
lines(gp~x, col = "red", lwd = 2)

plot_gp_post_realizations(fitInform, data, truth,
                          "Posterior Realizations")
plot_gp_post_quantiles(fitInform, data, truth,
                       "Posterior Marginal Quantiles")
plot_gp_post_pred_quantiles(fitInform, data, truth,
                            "Posterior Predictive Marginal Quantiles")

#mcmc
# fitInformMcmc <- stan(file = "fit_covar3MCMC.stan", data = data, seed = 826, refresh = 1000,
#                 chains = 1, iter = 100) #try at lower levels first
# save(fitInformMcmc, file = "fitInformMcmc.RData")
load(here(betanPath, "fitInformMcmc.RData"))
drawInformMcmc <- extract(fitInformMcmc)
#diagnostics
#check_all_diagnostics(fitInformMcmc) #for some reason this is taking forever
#visuals
plot(drawInformMcmc$f_predict[1,]~x, type = "l", ylim=c(-10, 10), 
     main = "some posterior function realizations from algorithm cov mat")
for(i in 2:nrow(drawInformMcmc$f_predict)) {
  lines(x, drawInformMcmc$f_predict[i,])
}
lines(gp~x, col = "red", lwd = 2)

plot_gp_post_realizations(fitInformMcmc, data, truth,
                          "Posterior Realizations")
plot_gp_post_quantiles(fitInformMcmc, data, truth,
                       "Posterior Marginal Quantiles")
plot_gp_post_pred_quantiles(fitInformMcmc, data, truth,
                            "Posterior Predictive Marginal Quantiles")


#####
#section 3.3
#####

#goes through how understanding at solve the degeneracy issue above allows us to 
#take a penalized approach to the mmle and get reasonable results, not doing this section
#but it is interesting

#####
#section 4
#####

#is there a difference between the bayesian approach and the mmle?
#bayes does slightly better in capturing the true functional behaviour
#very nice explaination of why that is












