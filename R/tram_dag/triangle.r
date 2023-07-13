source('R/tram_dag/utils.R')
library(R.utils)
#######################
# Latent Distribution
latent_dist = tfd_logistic(loc=0, scale=1)
#latent_dist = tfd_normal(loc=0, scale=1)
#latent_dist = tfd_truncated_normal(loc=0., scale=1.,low=-4,high = 4)
hist(latent_dist$sample(1e5)$numpy(),100, freq = FALSE, main='Samples from Latent')

M = 30
len_theta = M + 1
bp = make_bernp(len_theta)

######################################
############# Triangle ###############
######################################
# See Sánchez-Martin, Rateike, and Valera, “VACA.” Appendix 

#Roughly Limited to [0,1] 
dgp_scaled <- function(n) {
  N = n
  flip = sample(c(0,1), N, replace = TRUE)
  x1 = flip*rnorm(N, -2, sqrt(1.5)) + (1-flip) * rnorm(N, 1.5, 1)
  x1 = x1/14+0.6
  x2 = -0.3*x1 + 0.65 + rnorm(N, 0,0.1)
  x3 = 0.5*x1 + 0.25*x2 + 0.1 + rnorm(N,0,0.1)
  dat.s = data.frame(x = x1, y = x2, z = x3)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  return(dat.tf)
}

#TRIANGLELIN from VACA2 (eqs.76-78)
#f1(u1) = u1 + 1 
#f2(x1, u2) = 10 · x1 − u2 
#f3(x1, x2, u3) = 0.5 · x2 + x1 + u3

dgp <- function(n) {
  x <- rnorm(n, 0, 1) + 1
  y <- 10 * x - rnorm(n, 0, 1) 
  z <- 0.5 * y + 1 * x + rnorm(n, 0, 1) 
  dat.s =  data.frame(x = x, y = y, z = z)
  dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  return(dat.tf)
} 

dat.tf_u <- dgp(50000)
dat.tf = scale_df(dat.tf_u)

##### Creation of deep transformation models ######
# We need on for each variable in the SCM
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

####### This needs to be created ##########
thetaNN_l = list(thetaNN_x, thetaNN_y, thetaNN_z)
parents_l = list(parents_x, parents_y, parents_z)
target_l = list(target_x, target_y, target_z)

###### Training Step #####
epochs = 1000
loss = rep(NA, epochs)
optimizer = tf$keras$optimizers$Adam(learning_rate=0.001)

if(FALSE){
  for (e in 1:epochs){
    l = train_step(thetaNN_l, parents_l, target_l, optimizer=optimizer)
    loss[e]  = l$NLL$numpy()
    if (e %% 10 == 0) {
      print(e)
      print(l)
      }
    }
    # Saving weights after training
    thetaNN_l[[1]]$save_weights(paste0("R/tram_dag/triangle_model1_weights_scaled.h5"))
    thetaNN_l[[2]]$save_weights(paste0("R/tram_dag/triangle_model2_weights_scaled.h5"))
    thetaNN_l[[3]]$save_weights(paste0("R/tram_dag/triangle_model3_weights_scaled.h5"))
    plot(loss, type='l')
} else{
  thetaNN_l[[1]]$load_weights("R/tram_dag/triangle_model1_weights_scaled.h5")
  thetaNN_l[[2]]$load_weights("R/tram_dag/triangle_model2_weights_scaled.h5")
  thetaNN_l[[3]]$load_weights("R/tram_dag/triangle_model3_weights_scaled.h5")
}

#######################
# Below ad-hoc evaluation

parents_x = 0*dat.tf[,1] + 1 #We use 

#Evaluation via sampling
x_samples = sample_from_target(thetaNN_x, parents_x)
mean(x_samples$numpy() < 0)
hist(dat.tf$numpy()[,1], freq = FALSE,100)
lines(density(x_samples$numpy()), col='green')
#hist(x_samples$numpy(), freq=FALSE,10000,xlim = c(0,1))

### Distribution for Y
### Marginal Distribution for Y
hist(dat.tf$numpy()[,2], freq = FALSE,100)
parents_y = x_samples
y_samples = sample_from_target(thetaNN_y, parents_y)
mean(y_samples$numpy() < 0)
lines(density(y_samples$numpy()),col='green')
summary(y_samples$numpy())
quantile(y_samples$numpy(), seq(0,0.1,0.005)) #Approx 1% smaler 0

### Marginal Distribution for Z
hist(dat.s[,3], freq = FALSE,100)
lines(density(dat.s[,3]),col='red')
parents_z = tf$concat(list(x_samples, y_samples), axis=1L)
z_sample = sample_from_target(thetaNN_z, parents_z)
z = z_sample$numpy()
z=pmax(pmin(z, 1), 0)
lines(density(z),col='green')
hist(z,100)


############################### Do X ########################
#Samples from Z give X=doX
doX_triangle = function(doX){
    doX_tensor = tf$expand_dims(0*dat.tf[,1] + doX,1L) #set to doX
    ## X--> Y
    ys = sample_from_target(thetaNN_l[[2]], doX_tensor)
    parents_z = tf$concat(list(doX_tensor, ys), axis=1L)
    zs = sample_from_target(thetaNN_l[[3]], parents_z)
    return(matrix(c(doX_tensor$numpy(),ys$numpy(), zs$numpy()), ncol=3))
}



dat_do_xup = unscale(dat.tf_u, doX_triangle(doX=0.75))$numpy()
dat_do_xdown = unscale(dat.tf_u, doX_triangle(doX=0.25))$numpy()

x_0 = dat_do_xdown[1,1]
x_1 = dat_do_xup[1,1]

(mean(dat_do_xup[,2])  - mean(dat_do_xdown[,2])) / (x_1-x_0) #Should be 10.0
(mean(dat_do_xup[,3])  - mean(dat_do_xdown[,3])) / (x_1-x_0) #Should be 6.0







