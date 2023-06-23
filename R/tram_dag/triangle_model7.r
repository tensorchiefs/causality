library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
library(tfprobability)

#######################
# Latent Distribution
latent_dist = tfd_logistic(loc=0, scale=1)
#latent_dist = tfd_normal(loc=0, scale=1)
#latent_dist = tfd_truncated_normal(loc=0., scale=1.,low=-4,high = 4)
hist(latent_dist$sample(1e5)$numpy(),100, freq = FALSE, main='Samples from Latent')

source('tram_scm/bern_utils.R')
source('tram_scm/model_utils.R')
source('R/tram_dag/utils.R')

M = 30
len_theta = M + 1
bp = make_bernp(len_theta)

######################################
############# Triangle ###############
######################################
# See Sánchez-Martin, Rateike, and Valera, “VACA.” Appendix 

#Roughly Limited to [0,1] 
dgp <- function(n) {
  N = n
  flip = sample(c(0,1), N, replace = TRUE)
  x1 = flip*rnorm(N, -2, sqrt(1.5)) + (1-flip) * rnorm(N, 1.5, 1)
  x1 = x1/14+0.6
  x2 = -0.3*x1 + 0.65 + rnorm(N, 0,0.1)
  x3 = 0.5*x1 + 0.25*x2 + 0.1 + rnorm(N,0,0.1)
  data.frame(x = x1, y = x2, z = x3)
}

dat <- dgp(50000)
dat.s = dat
range(dat.s)
hist(dat.s[,1],50)
hist(dat.s[,2],50)
dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')

##### Creation of the three NNs ######
thetaNN_x = make_model(len_theta, 1)
parents_x = 0*dat.tf[,1] + 1 #set to 1
target_x = dat.tf[,1, drop=FALSE]
calc_NLL(thetaNN_x, parents_x, target_x)

thetaNN_y = make_model(len_theta, 1)
parents_y = dat.tf[,1]
target_y = dat.tf[,2, drop=FALSE]
calc_NLL(thetaNN_y, parents_y, target_y)

thetaNN_z = make_model(len_theta, parent_dim = 2)
parents_z = dat.tf[,c(1,2)]
target_z = dat.tf[,3, drop=FALSE]
calc_NLL(thetaNN_z, parents_z, target_z)

thetaNN_l = list(thetaNN_x, thetaNN_y, thetaNN_z)
parents_l = list(parents_x, parents_y, parents_z)
target_l = list(target_x, target_y, target_z)

###### Training Step #####
train_step = function(thetaNN_l, parents_l, target_l){
  optimizer = tf$keras$optimizers$Adam(learning_rate=0.0001)
  with(tf$GradientTape() %as% tape, {
    NLL1 = calc_NLL(thetaNN_l[[1]], parents_l[[1]], target_l[[1]])
    NLL2 = calc_NLL(thetaNN_l[[2]], parents_l[[2]], target_l[[2]]) 
    NLL3 = calc_NLL(thetaNN_l[[3]], parents_l[[3]], target_l[[3]]) 
    NLL = NLL1 + NLL2 + NLL3
  })
  #Creating a list for all gradients
  n = 3
  tvars = list(
    thetaNN_l[[1]]$trainable_variables,
    thetaNN_l[[2]]$trainable_variables,
    thetaNN_l[[3]]$trainable_variables
  ) 
  #Calculation of the gradients
  grads = tape$gradient(NLL, tvars)
  for (i in 1:n){
    optimizer$apply_gradients(
      purrr::transpose(list(grads[[i]], tvars[[i]]))
    )  
  }
  return(list(NLL=NLL, NLL1 = NLL1, NLL2 = NLL2, NLL3 = NLL3))
}


epochs = 1000
loss = rep(NA, epochs)
loss1 = rep(NA, epochs)
loss2 = rep(NA, epochs)
loss3 = rep(NA, epochs)

if(FALSE){
  for (e in 1:epochs){
    l = train_step(thetaNN_l, parents_l, target_l)
    loss[e]  = l$NLL$numpy()
    loss1[e] = l$NLL1$numpy()
    loss2[e] = l$NLL2$numpy()
    loss3[e] = l$NLL3$numpy()
    if (e %% 10 == 0) {
      print(e)
      print(l)
      }
    }
    # Saving weights after training
    thetaNN_l[[1]]$save_weights(paste0("R/tram_dag/triangle_model1_weights.h5"))
    thetaNN_l[[2]]$save_weights(paste0("R/tram_dag/triangle_model2_weights.h5"))
    thetaNN_l[[3]]$save_weights(paste0("R/tram_dag/triangle_model3_weights.h5"))
    plot(loss, type='l')
    plot(loss1, col='red', type='l')
    plot(loss2, col='green', type='l')
    plot(loss3, col='blue', type='l')
} else{
  thetaNN_l[[1]]$load_weights("R/tram_dag/triangle_model1_weights_logis.h5")
  thetaNN_l[[2]]$load_weights("R/tram_dag/triangle_model2_weights_logis.h5")
  thetaNN_l[[3]]$load_weights("R/tram_dag/triangle_model3_weights_logis.h5")
}




### Distribution for X
hist(dat.s[,1], freq = FALSE,100, main='X (Observed Data)')
parents_x = 0*dat.tf[,1] + 1

#Evaluation via p(x) and 
target_grid_R = seq(0,1,length.out=length(parents_x))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
px = predict_p_target(thetaNN_l[[1]], parents = parents_x, target_grid = target_grid)
lines(target_grid_R, px, col='red') 

#Evaluation via sampling
sample_from_target(thetaNN_x, parents_x[1:3])
x_samples = sample_from_target(thetaNN_x, parents_x)
lines(density(x_samples$numpy()), col='green')
hist(x_samples$numpy(), freq=FALSE,10000,xlim = c(0,1))

# Ploting h for x (given 1)
doX =parents_x
doX_tensor = 0*dat.tf[,1] + doX #set to doX
target_grid_R = seq(-1,2,length.out=length(doX_tensor))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
h = predict_h(thetaNN_l[[2]], doX_tensor, target_grid)
plot(target_grid_R,h, type='l')


### Marginal Distribution for Y
hist(dat.s[,2], freq = FALSE,100)
parents_y = x_samples
#py = predict_p_target(thetaNN_l[[2]], parents = parents_y, target_grid = target_grid)
#lines(target_grid_R, py)
y_samples = sample_from_target(thetaNN_y, parents_y)
lines(density(y_samples$numpy()),col='green')
summary(y_samples$numpy())
quantile(y_samples$numpy(), seq(0,0.1,0.005)) #Approx 1% smaler 0
hist(y_samples$numpy(), freq=FALSE,1000,xlim=c(0,1))


### Marginal Distribution for Z
hist(dat.s[,3], freq = FALSE,100)
lines(density(dat.s[,3]),col='red')
parents_z = tf$concat(list(x_samples, y_samples), axis=1L)
z_sample = sample_from_target(thetaNN_z, parents_z)
z = z_sample$numpy()
z=pmax(pmin(z, 1), 0)
lines(density(z),col='pink')
hist(z_sample$numpy(), 10000, freq = FALSE, xlim = c(0,1), ylim=c(0,10))

######################
# doX 

#Samples from Z give X=doX
doX_triangle = function(doX, rep=1){
  zs = NULL 
  for(i in 1:rep){
    doX_tensor = 0*dat.tf[,1] + doX #set to doX
    ## X--> Y
    ys = sample_from_target(thetaNN_l[[2]], doX_tensor)
    parents_z = matrix(c(doX_tensor$numpy(), ys$numpy()), ncol=2)
    #parents_z = tf$concat(list(doX_tensor, ys), axis=1L)
    tmp = sample_from_target(thetaNN_l[[3]], parents_z)
    zs = c(zs, tmp$numpy())
  }
  return (zs)
}

z_do_xup = doX_triangle(0.5)
hist(z_do_xup,1000,xlim=c(0,1))
plot(ecdf(z_do_xup))
summary(z_do_xup)

z_do_down = doX_triangle(0.3)
hist(z_do_down,1000, xlim=c(0,1))
mean(z_do_xup) - mean(z_do_down)
#median(z_do_xup) - median(z_do_down)
(0.5 - 0.3*0.25)*0.2







