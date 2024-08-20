##### Oliver's MAC ####
reticulate::use_python("/Users/oli/miniforge3/envs/r-tensorflow/bin/python3.8", required = TRUE)
library(reticulate)
reticulate::py_config()
library(tensorflow)
library(keras)

library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
source('summerof24/utils_tf.R')

#### For TFP
library(tfprobability)
source('summerof24/utils_tfp.R')


fn = 'triangle_colr_No_Scaling_2_M20.h5'
##### TEMP
dgp <- function(n_obs, doX=c(NA, NA, NA)) {
    #n_obs = 1e5 n_obs = 10
    #Sample X_1 from GMM with 2 components
    if (is.na(doX[1])){
      X_1_A = rnorm(n_obs, 0.25, 0.1)
      X_1_B = rnorm(n_obs, 0.73, 0.05)
      X_1 = ifelse(sample(1:2, replace = TRUE, size = n_obs) == 1, X_1_A, X_1_B)
    } else{
      X_1 = rep(doX[1], n_obs)
    }
    #hist(X_1)
    
    # Sampling according to colr
    if (is.na(doX[2])){
      U2 = runif(n_obs)
      x_2_dash = qlogis(U2)
      #x_2_dash = h_0(x_2) + beta * X_1
      #x_2_dash = 0.42 * x_2 + 2 * X_1
      X_2 = 1/0.42 * (x_2_dash - 2 * X_1)
    } else{
      X_2 = rep(doX[2], n_obs)
    }
    
    #hist(X_2)
    
    # Sampling according to colr
    if (is.na(doX[3])){
      U3 = runif(n_obs)
      x_3_dash = qlogis(U3)
      #x_3_dash = h_0_3(x_3) + gamma_1 * X_1 + gamma_2 * X_2
      #x_3_dash = 0.63 * x_3 -0.2 * X_1 + 1.3 * X_2
      X_3 = (x_3_dash + 0.2 * X_1 - 1.3 * X_2)/0.63
    } else{
      X_3 = rep(doX[3], n_obs)
    }
   
    #hist(X_3)
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
    dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
    
    q1 = quantile(dat.orig[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
    q3 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
    
    return(list(
      df_orig=dat.tf,  
      #min =  tf$reduce_min(dat.tf, axis=0L),
      #max =  tf$reduce_max(dat.tf, axis=0L),
      min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
      A=A))
} 

train = dgp(4000)
(global_min = train$min)
(global_max = train$max)
data_type = c('c','c','c')

#### Fitting Tram ######
df = data.frame(train$df_orig$numpy())
fit.orig = Colr(X2~X1,df)
dd = predict(fit.orig, newdata = data.frame(X1 = 0.5), type = 'density')
x2s = as.numeric(rownames(dd))
plot(x2s, dd, type = 'l', col='red')

#?predict.tram
summary(fit.orig)
confint(fit.orig) #Original 
# Fitting Tram
df = data.frame(train$df_orig$numpy())
fit.orig = Colr(X3 ~ X1 + X2,df)
summary(fit.orig)
confint(fit.orig) #Original 



MA =  matrix(c(0, 'ls', 'ls', 0,0, 'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
len_theta = 20
param_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta, 
                                 hidden_features_CS = hidden_features_CS)

x = tf$ones(shape = c(2L, 3L))
param_model(1*x)
MA
h_params = param_model(train$df_orig)
# Check the derivatives of h w.r.t. x
x <- tf$ones(shape = c(2L, 3L)) #B,P
with(tf$GradientTape(persistent = TRUE) %as% tape, {
  tape$watch(x)
  y <- param_model(x)
})
# parameter (output) has shape B, P, k (num param)
# derivation of param wrt to input x
# input x has shape B, P
# derivation d has shape B,P,k, B,P
d <- tape$jacobian(y, x)
d[1,,,2,] # only contains zero since independence of batches


MA
#      [,1] [,2] [,3]
# [1,] "0"  "ls" "ci"
# [2,] "0"  "0"  "cs"
# [3,] "0"  "0"  "0" 
# check which x_i is dependent on other x_j
# k=1: cs, k=2:ls, rest of k: CI (len_theta params)
# für k=1 sollte hur bei x2-->x3 ableitung !=0 sein: ok 
# für k=2 sollte x1-->x2 abl !=0 sein: ok
# für k=3...2+len_theta sollte sollte für x1-->x3 !=0 sein: ok
for (k in 1:(2+len_theta)){ #k = 1
  print(k) #B,P,k,B,P
  B = 1 # first batch
  print(d[B,,k,B,]) #
}

# loss before training
struct_dag_loss(train$df_orig, h_params)


with(tf$GradientTape(persistent = TRUE) %as% tape, {
  h_params = param_model(train$df_orig)
  loss = struct_dag_loss(train$df_orig, h_params)
})

gradients = tape$gradient(loss, param_model$trainable_variables)
gradients

param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS)


# ######### DEBUG TRAINING FROM HAND #######
# # Define the optimizer
# optimizer <- tf$optimizers$Adam(lr=0.01)
# # Define the number of epochs for training
# num_epochs <- 10
# for (epoch in 1:num_epochs) {
#   with(tf$GradientTape(persistent = TRUE) %as% tape, {
#     # Compute the model's prediction - forward pass
#     h_params <- param_model(train$df_scaled)
#     loss <- struct_dag_loss(train$df_scaled, h_params)
#   })
#   # Compute gradients
#   gradients <- tape$gradient(loss, param_model$trainable_variables)
#   # Apply gradients to update the model parameters
#   optimizer$apply_gradients(purrr::transpose(list(gradients, param_model$trainable_variables)))
#   # Print the loss every epoch or more frequently if desired
#   print(paste("Epoch", epoch, ", Loss:", loss$numpy()))
# }


optimizer = optimizer_adam()
param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)


##### Training ####
if (file.exists(fn)){
  param_model$load_weights(fn)
} else {
  hist = param_model$fit(x = train$df_orig, y=train$df_orig, 
                         epochs = 2000L,verbose = TRUE)
  param_model$save_weights(fn)
  plot(hist$epoch, hist$history$loss)
  plot(hist$epoch, hist$history$loss, ylim=c(1.5, 1.7))
}
param_model$evaluate(x = train$df_orig, y=train$df_scaled, batch_size = 7L)
fn
len_theta
param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask


#### Checking the transformation ####
h_params = param_model(train$df_orig)
r = check_baselinetrafo(h_params)
Xs = r$Xs
h_I = r$h_I


##### X1
fit.1 = Colr(X1~1,df)
plot(fit.1, which = 'baseline only')
lines(Xs[,1], h_I[,1], col='red', lty=2, lwd=3)
rug(train$df_orig$numpy()[,1], col='blue')
#transformed_values <- predict(fit.1, newdata = seq(0, 1, 0.01))

df = data.frame(train$df_orig$numpy())
fit.21 = Colr(X2~X1,df)
temp = model.frame(fit.21)[1:2,-1, drop=FALSE] #WTF!
plot(fit.21, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X2) Colr and Our Model', cex.main=0.8)
lines(Xs[,2], h_I[,2], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,2], col='blue')

fit.312 = Colr(X3 ~ X1 + X2,df)
temp = model.frame(fit.312)[1:2, -1, drop=FALSE] #WTF!

plot(fit.312, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X3) Colr and Our Model', cex.main=0.8)
lines(Xs[,3], h_I[,3], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,3], col='blue')


# Check the derivatives of h w.r.t. x
x <- tf$ones(shape = c(10L, 3L)) #B,P
with(tf$GradientTape(persistent = TRUE) %as% tape, {
  tape$watch(x)
  y <- param_model(x)
})
d <- tape$jacobian(y, x)
for (k in 1:(2+len_theta)){ #k = 1
  print(k) #B,P,k,B,P
  B = 1
  print(d[B,,k,B,]) #
}

o = train$df_orig$numpy()
plot(o[,1],o[,2])
lm(o[,2] ~ o[,1])

##### Checking observational distribution ####
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)
par(mfrow=c(1,3))
for (i in 1:3){
  d = s[,i]$numpy()
  hist(d, freq=FALSE, 100,main=paste0("X_",i))
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  lines(density(train$df_orig$numpy()[,i]))
}
par(mfrow=c(1,1))

######### Simulation of do-interventions #####
doX=c(0.2, NA, NA)
dx0.2 = dgp(10000, doX=doX)
dx0.2$df_orig$numpy()[1:5,]

doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX)
hist(dx0.2$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx0.2$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx0.2$df_orig$numpy()[,3])  

########### Do(x1) seems to work#####

#### Check intervention distribution after do(X1=0.2)
df = data.frame(train$df_orig$numpy())
fit.x2 = Colr(X2~X1,df)
x2_dense = predict(fit.x2, newdata = data.frame(X1 = 0.2), type = 'density')
x2s = as.numeric(rownames(x2_dense))

## samples from x2 under do(x1=0.2) via simulate
ddd = as.numeric(unlist(simulate(fit.x2, newdata = data.frame(X1 = 0.2), nsim = 1000)))
s2_colr = rep(NA, length(ddd))
for (i in 1:length(ddd)){
  s2_colr[i] = as.numeric(ddd[[i]]) #<--TODO somethimes 
}
if(sum(is.na(s2_colr)) > 0){
  stop("Pechgehabt mit Colr, viel Glück und nochmals!")
}

hist(s2_colr, freq=FALSE, 100, main='Do(X1=0.2) X2')
lines(x2s, x2_dense, type = 'l', col='red')

fit.x3 = Colr(X3 ~ X1 + X2,df)
newdata = data.frame(
    X1 = rep(0.2, length(s2_colr)), 
    X2 = s2_colr)

s3_colr = rep(NA, nrow(newdata))
for (i in 1:nrow(newdata)){
  # i = 2
  s3_colr[i] = simulate(fit.x3, newdata = newdata[i,], nsim = 1)
}

s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
par(mfrow=c(1,2))
for (i in 2:3){
  d = s_dag[,i]$numpy()
  ds = dx0.2$df_orig$numpy()[,i]
  print(paste0('sim mean ',mean(ds), '  med',median(ds)))
  print(paste0('DAG mean ',mean(d), '  med',median(d)))
  hist(d, freq=FALSE, 50, 
       main=paste0("green=DGP, red=Colr, hist=Ours, 
                   Do(X1=0.2) X_",i))
  lines(density(ds), col='green', lw=2)
  if (i ==2) lines(density(s2_colr), type = 'l', col='red')
  if (i ==3) lines(density(s3_colr), col='red')
}
par(mfrow=c(1,1))



s7 = do_dag_struct(param_model, train$A, doX=c(0.7, NA, NA))
for (i in 1:3){
  d = s7[,i]$numpy()
  ds = dx7$df_orig$numpy()[,i]
  print(paste0('sim mean ',mean(ds), '  med',median(ds)))
  print(paste0('DAG mean ',mean(d), '  med',median(d)))
  hist(d, freq=FALSE, 50, main=paste0("Do(X1=0.7) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}

mean(s7$numpy()[,2]) - mean(s$numpy()[,2])
mean(s7$numpy()[,3]) - mean(s$numpy()[,3])

median(sup$numpy()[,2]) - median(sdown$numpy()[,2])
median(sup$numpy()[,3]) - median(sdown$numpy()[,3])


####################################################################################
################ Below not tested after refactoring to new DGPs (Using Tranformation models in DGP) 
############################
####################################################################################

########### Do(x2) seem to work #####
s = do_dag_struct(param_model, train$A, doX=c(NA, 0.5, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}

s = do_dag_struct(param_model, train$A, doX=c(NA, 0.7, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}


########### TODO Check the sampling (prob needs ad) #####
dox1_2=scale_value(train$df_orig, col=1L, 2) #On X2
s_dox1_2 = do_dag(param_model, train$A, doX=c(dox1_2$numpy(), NA, NA), num_samples = 5000)
s = s_dox1_2
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}
mean(s_dox1_2$numpy()[,3])
df = unscale(train$df_orig, s_dox1_2)
mean(df$numpy()[,3]) #1.39

dox1_3=scale_value(train$df_orig, col=1L, 3.) #On X2
s_dox1_3 = do_dag(param_model, train$A, doX=c(dox1_3$numpy(), NA, NA), num_samples = 5000)
mean(s_dox1_3$numpy()[,3])
df = unscale(train$df_orig, s_dox1_3)
mean(df$numpy()[,3]) #2.12

dox1_3=scale_value(train$df_orig, col=1L, 1.) #On X2
s_dox1_3 = do_dag(param_model, train$A, doX=c(dox1_3$numpy(), NA, NA), num_samples = 5000)
mean(s_dox1_3$numpy()[,3])
df = unscale(train$df_orig, s_dox1_3)
mean(df$numpy()[,3]) #0.63
t.test(df$numpy()[,3])


hist(train$df_scaled$numpy()[,1], freq=FALSE)

if(FALSE){
  x = tf$ones(c(2L,3L)) * 0.5
  # Define the MLP model
  input_layer <- layer_input(shape = list(ncol(adjacency)))
  d = layer_dense(units = 64, activation = 'relu')(input_layer)
  d = layer_dense(units = 30)(d)
  d = layer_reshape(target_shape = c(3, 10))(d)
  param_model = keras_model(inputs = input_layer, outputs = d)
  print(param_model)
  param_model(x)
  tf$executing_eagerly()  # Should return TRUE
  with(tf$GradientTape(persistent = TRUE) %as% tape, {
    theta_tilde = param_model(x, training=TRUE)
    loss = dag_loss(x, theta_tilde)
  })
  #gradients <- lapply(gradients, function(g) tf$debugging$check_numerics(g, "Gradient NaN/Inf check"))
  gradients = tape$gradient(loss, param_model$trainable_variables)
  gradients
  
  param_model$trainable_variables
  # Update weights
  optimizer.apply_gradients(zip(gradients, param_model.trainable_variables))
}




















