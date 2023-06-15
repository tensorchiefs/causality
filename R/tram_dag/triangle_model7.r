library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
library(tfprobability)

source('tram_scm/bern_utils.R')
source('tram_scm/model_utils.R')
source('R/tram_dag/utils.R')


M = 30
len_theta = M + 1
bp = make_bernp(len_theta)

#### Testing with geysir ####
if (FALSE){
  thetaNN = make_model(len_theta, 1)
  target = tf$constant(as.matrix(utils_scale(geyser$waiting)), dtype = 'float32')
  parents = 0*target + 1 #set to 1
  calc_NLL(thetaNN, parents, target)
  
  epochs = 1e3
  loss = rep(NA, epochs)
  for (e in 1:epochs){
    l = train_step(thetaNN, parents, target)
    loss[e] = l$numpy()
    if (e %% 10 == 0) {
      print(e)
      print(l)
    }
  }
  plot(loss, xlim=c(0,1e4), ylim=c(-0.6,0.1))
  ys = seq(0,1,length.out=length(geyser$waiting))
  target_grid = tf$cast(matrix(ys, ncol=1), tf$float32)
  ps = predict_p_target(thetaNN, parents = parents, target_grid = target_grid)
  hist(target$numpy(),30, freq = FALSE)
  lines(ys, ps$numpy())
  
  ############# Root Finding test
  a = tf$Variable(c(10.,1.0,3.0))
  object_fkt = function(x){
    tf$math$cos(a*x)
  }
  object_fkt(0.0)
  tfp$math$find_root_chandrupatla(object_fkt, low = 0, high = 4)
  object_fkt(3.2986722)
  cos(1.5707964)
  
  ####### Root Finding for sampling#####
  z = rlogis(299)
  theta_tilde = thetaNN(parents)
  theta_tilde = tf$cast(theta_tilde, dtype=tf$float32)
  theta = to_theta(theta_tilde)
  #y_i = tf$Variable(matrix(0.4, nrow=1,ncol=1), dtype=tf$float32)
  object_fkt = function(y_i){
    return(tf$reshape((eval_h(theta, y_i = y_i, beta_dist_h = bp$beta_dist_h) - z), c(299L,1L)))
  }
  res = tfp$math$find_root_chandrupatla(object_fkt, low = 0, high = 1)
  hist(res$estimated_root$numpy())
  
  y_i = tf$Variable(matrix(0.5636344, nrow=1,ncol=1), dtype=tf$float32)
  eval_h(theta, y_i = y_i, beta_dist_h = bp$beta_dist_h)
  
  quantile(target$numpy())
  eval_h(theta, y_i = 0.5076923, beta_dist_h = bp$beta_dist_h)
  
  h_dash = eval_h_dash(theta, target, beta_dist_h_dash = bp$beta_dist_h_dash)
  pz = tfd_logistic(loc=0, scale=1)
}


#### x->y ####
if (FALSE){
dgp <- function(n) {
  x <- seq(0.25,0.75, length.out=n)
  y <- 0.42 * x + 0.1*runif(n,0.1,0.4)
  data.frame(x = x, y = y)
}

dat <- dgp(42)
dat.s = dat
#dat.s = apply(dat, 2, utils_scale)
dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
  
thetaNN_x = make_model(len_theta, 1)
parents_x = 0*dat.tf[,1] + 1 #set to 1
target_x = dat.tf[,1, drop=FALSE]
calc_NLL(thetaNN_x, parents_x, target_x)

thetaNN_y = make_model(len_theta, 1)
parents_y = dat.tf[,1]
target_y = dat.tf[,2, drop=FALSE]

thetaNN_l = list(thetaNN_x, thetaNN_y)
parents_l = list(parents_x, parents_y)
target_l = list(target_x, target_y)

train_step = function(thetaNN_l, parents_l, target_l){
  optimizer = tf$keras$optimizers$Adam(learning_rate=0.0001)
  with(tf$GradientTape() %as% tape, {
    NLL = calc_NLL(thetaNN_l[[1]], parents_l[[1]], target_l[[1]]) +
      calc_NLL(thetaNN_l[[2]], parents_l[[2]], target_l[[2]])
  })
  #Creating a list for all gradients
  n = 2
  tvars = list(
    thetaNN_l[[1]]$trainable_variables,
    thetaNN_l[[2]]$trainable_variables
  ) 
  #Calculation of the gradients
  grads = tape$gradient(NLL, tvars)
  for (i in 1:n){
    optimizer$apply_gradients(
      purrr::transpose(list(grads[[i]], tvars[[i]]))
    )  
  }
  return(NLL)
}

epochs = 1e3
loss = rep(NA, epochs)
for (e in 1:epochs){
  l = train_step(thetaNN_l, parents_l, target_l)
  loss[e] = l$numpy()
  if (e %% 10 == 0) {
    print(e)
    print(l)
  }
}

plot(loss)
target_grid_R = seq(0,1,length.out=length(dat$x))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
ps = predict_p_target(thetaNN_l[[1]], parents = parents_l[[1]], target_grid = target_grid)
hist(target_l[[1]]$numpy(),30, freq = FALSE)
lines(target_grid_R, ps$numpy())


plot_dist = function(dox){
  parents_x = 0*dat.tf[,1] + dox
  ps = predict_p_target(thetaNN_l[[2]], parents = parents_x, target_grid = target_grid)
  plot(target_grid, ps)
}

plot(dat.tf[,1], dat.tf[,2])
lm(y ~ x,as.data.frame(dat.s))
plot_dist(dox=0.3)
plot_dist(dox=0.6)

for (x in dat.s[,1]){
  parents_x = 0*dat.tf[,1] + x
  ps = predict_p_target(thetaNN_l[[2]], parents = parents_x, target_grid = target_grid)
  plot(target_grid, ps)
  abline(v=x)
}
hist(target_l[[2]]$numpy(),30, freq = FALSE)
lines(target_grid_R, ps$numpy())
}

############# Chain ###############
if (FALSE){
dgp <- function(n) {
  x <- seq(0.25,0.75, length.out=n)
  y <- 0.9 * x + 0.1*runif(n,0.1,0.2)
  z <- -1.5 * y + 1.0 + runif(n,0.2,0.3)
  data.frame(x = x, y = y, z = z)
}

dat <- dgp(42)
dat.s = dat
#dat.s = apply(dat, 2, utils_scale)
dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')

thetaNN_x = make_model(len_theta, 1)
parents_x = 0*dat.tf[,1] + 1 #set to 1
target_x = dat.tf[,1, drop=FALSE]
calc_NLL(thetaNN_x, parents_x, target_x)

thetaNN_y = make_model(len_theta, 1)
parents_y = dat.tf[,1]
target_y = dat.tf[,2, drop=FALSE]

thetaNN_z = make_model(len_theta, 1)
parents_z = dat.tf[,2]
target_z = dat.tf[,3, drop=FALSE]

thetaNN_l = list(thetaNN_x, thetaNN_y, thetaNN_z)
parents_l = list(parents_x, parents_y, parents_z)
target_l = list(target_x, target_y, target_z)

train_step = function(thetaNN_l, parents_l, target_l){
  optimizer = tf$keras$optimizers$Adam(learning_rate=0.001)
  with(tf$GradientTape() %as% tape, {
    NLL = 
      calc_NLL(thetaNN_l[[1]], parents_l[[1]], target_l[[1]]) +
      calc_NLL(thetaNN_l[[2]], parents_l[[2]], target_l[[2]]) +
      calc_NLL(thetaNN_l[[3]], parents_l[[3]], target_l[[3]]) 
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
  return(NLL)
}

epochs = 1e3
loss = rep(NA, epochs)
for (e in 1:epochs){
  l = train_step(thetaNN_l, parents_l, target_l)
  loss[e] = l$numpy()
  if (e %% 10 == 0) {
    print(e)
    print(l)
  }
}
plot(loss)
hist(dat.s[,1], freq = FALSE)
parents_x = 0*dat.tf[,1] + 1
target_grid_R = seq(0,1,length.out=length(parents_x))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
py = predict_p_target(thetaNN_l[[1]], parents = parents_x, target_grid = target_grid)
lines(target_grid_R, py)

####### Root Finding for sampling #####
doX_chain = function(doX){
  zs = NULL 
  for(i in 1:10){
    parents_x = 0*dat.tf[,1] + doX #set to doX
    ## X--> Y
    ys = sample_from_target(thetaNN_l[[2]], parents_x)
    tmp = sample_from_target(thetaNN_l[[3]], ys)
    zs = c(zs, tmp$numpy())
  }
  return (zs)
}

#Eval Individual
parents_x = 0*dat.tf[,1] + 1
d = sample_from_target(thetaNN_l[[2]], parents_x)
target_grid_R = seq(0,1,length.out=length(parents_x))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
py_doX0 = predict_p_target(thetaNN_l[[2]], parents = parents_x, target_grid = target_grid)
plot(target_grid_R, py_doX0)


hist(d$numpy())
z_dox0 = doX_chain(0.4)
hist(z_dox0,100)
abline(v=mean(z_dox0))
z_dox1 = doX_chain(0.6)
hist(z_dox1, 100)
abline(v=mean(z_dox1))
0.9*(-1.5)*0.2
mean(z_dox1) - mean(z_dox0)
}

############# Triangle ###############
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
abline(v=0.5, col='red')

#dat.s = apply(dat, 2, utils_scale)
dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')

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


epochs = 500
loss = rep(NA, epochs)
loss1 = rep(NA, epochs)
loss2 = rep(NA, epochs)
loss3 = rep(NA, epochs)

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
plot(loss, type='l')
plot(loss1, col='red', xlim=c(0,200), type='l')
plot(loss2, col='green', xlim=c(0,200), type='l')
plot(loss3, col='blue', xlim=c(0,200), type='l')


par(mfrow=c(2,2))
plot(loss, type='l')
### Distribution for X
hist(dat.s[,1], freq = FALSE,100, main='X')
parents_x = 0*dat.tf[,1] + 1
target_grid_R = seq(0,1,length.out=length(parents_x))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
px = predict_p_target(thetaNN_l[[1]], parents = parents_x, target_grid = target_grid)
lines(target_grid_R, px, col='red')
x_samples = sample_from_target(thetaNN_x, parents_x)
lines(density(x_samples$numpy()), col='green')

### Marginal Distribution for Y
hist(dat.s[,2], freq = FALSE,100)
parents_y = x_samples
#py = predict_p_target(thetaNN_l[[2]], parents = parents_y, target_grid = target_grid)
#lines(target_grid_R, py)
y_samples = sample_from_target(thetaNN_y, parents_y)
lines(density(y_samples$numpy()))

### Marginal Distribution for Z
hist(dat.s[,3], freq = FALSE,100)
parents_z = tf$concat(list(x_samples, y_samples), axis=1L)
z_sample = sample_from_target(thetaNN_z, parents_z)
lines(density(z_sample$numpy()))
par(mfrow=c(1,1))

####### Sampling #####
doX = 2.0
doX_tensor = 0*dat.tf[,1] + doX #set to doX
## X--> Y
ys = sample_from_target(thetaNN_l[[2]], doX_tensor)
target_grid_R = seq(0,1,length.out=length(doX_tensor))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
h = predict_h(thetaNN_l[[2]], doX_tensor, target_grid)
plot(target_grid_R,h, type='l')


h0 = h[1]
h1 = h[length(target_grid_R)]



zs = seq(0,1,0.001)
z = rlogis(1e4)
hist(z,100)
plot(zs, dlogis(zs))
hist(ys$numpy(),100)
qqPlot(ys$numpy())

parents_z = matrix(c(doX_tensor$numpy(), ys$numpy()), ncol=2)
#parents_z = tf$concat(list(doX_tensor, ys), axis=1L)
tmp = sample_from_target(thetaNN_l[[3]], parents_z)
zs = c(zs, tmp$numpy())







doX_triangle = function(doX){
  zs = NULL 
  for(i in 1:10){
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
hist(z_do_xup,100)
plot(ecdf(z_do_xup))
summary(z_do_xup)
library(car)
qqPlot(z_do_xup)

z_do_down = doX_triangle(0.3)
hist(z_do_down)
mean(z_do_xup) - mean(z_do_down)
median(z_do_xup) - median(z_do_down)
(0.5 - 0.3*0.25)*0.2


#plot of Y|do(x=0.3)
do=0.8
parents_x = 0*dat.tf[,1] + 1*dox
#d = sample_from_target(thetaNN_l[[2]], parents_x)
target_grid_R = seq(0,1,length.out=length(parents_x))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
py_doX0 = predict_p_target(thetaNN_l[[2]], parents = parents_x, target_grid = target_grid)
plot(target_grid_R, py_doX0)
abline(v=-0.3*dox+0.65)
s = sample_from_target(thetaNN_l[[2]], parents_x)
mean(s) 
#0.5656284 for 0.3
#0.4309883 for 0.8
#(0.4309883 - 0.5656284) / 0.5 --> -0.2692802


#DGP
x1 = dox
x2 = -0.3*x1 + 0.65 + rnorm(1e5, 0,0.1)
lines(density(x2), col='red')
mean(x2) 
#dox = 0.8 --> 0.41005
#dox = 0.3 --> 0.5602196 
#diff (0.41005 - 0.5602196)/0.5 --> -0.3




###### Observational
parents_x = 0*dat.tf[,1] + 1 #set to 1
xs = sample_from_target(thetaNN_l[[1]], parents_x)
hist(xs$numpy(),100, freq = FALSE)
lines(density(dat.s[,1]))

ys = sample_from_target(thetaNN_l[[2]], xs)
hist(ys$numpy(),100, freq = FALSE)
lines(density(dat.s[,2]))

ys = sample_from_target(thetaNN_l[[2]], xs)
hist(ys$numpy(),100, freq = FALSE)
lines(density(dat.s[,2]))

parents_z = matrix(c(xs$numpy(), ys$numpy()), ncol=2)
zs = sample_from_target(thetaNN_l[[3]], parents_z)
hist(zs$numpy(),100, freq = FALSE)
lines(density(dat.s[,3]))



#plot of Y
hist(dat.s[,2], freq = FALSE)
target_grid_R = seq(0,1,length.out=length(parents_x))
target_grid = tf$cast(matrix(target_grid_R, ncol=1), tf$float32)
d = predict_p_target(thetaNN_l[[2]], parents = parents_y, target_grid = target_grid)
lines(target_grid_R, d, type='l')


###### do ######
dox = 0.8
x1 = dox
x2 = -0.3*x1 + 0.65 + rnorm(1e5, 0,0.1)
x3 = 0.5*x1 + 0.25*x2 + 0.1 + rnorm(1e5,0,0.1)
mean(x3) 
#0.3899373 for 0.3
#0.6026701 for 0.8
(0.6026701 - 0.3899373) / 0.5

z_dox0 = doX_triangle(0.3)
hist(z_dox0,50)
abline(v=0.64*0.25 + 0.5*0.2)
abline(v=mean(z_dox0))

z_dox1 = doX_triangle(0.8)
hist(z_dox1, 50)
abline(v=mean(z_dox1))
(0.5 - 0.3*0.25)*0.5
mean(z_dox1) - mean(z_dox0)







