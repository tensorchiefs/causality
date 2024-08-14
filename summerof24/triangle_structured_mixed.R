##### Oliver's MAC ####
reticulate::use_python("/Users/oli/miniforge3/envs/r-tensorflow/bin/python3.8", required = TRUE)
library(reticulate)
reticulate::py_config()

#### A mixture of discrete and continuous variables ####
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


fn = 'triangle_mixed.h5'
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
      X_2 = 1/5. * (x_2_dash - 0.4 * X_1) # 0.39450
      X_2 = 1/5. * (x_2_dash - 1.2 * X_1) 
      X_2 = 1/5. * (x_2_dash - 2 * X_1)  # 
      
      
    } else{
      X_2 = rep(doX[2], n_obs)
    }
    
    #hist(X_2)
    #ds = seq(-5,5,0.1)
    #plot(ds, dlogis(ds))
    
    if (is.na(doX[3])){
      # x3 is an ordinal variable with K = 4 levels x3_1, x3_2, x3_3, x3_4
      # h(x3 | x1, x2) = h0 + gamma_1 * x1 + gamma_2 * x2
      # h0(x3_1) = theta_1, h0(x_3_2) =  theta_2, h0(x_3_3) = theta_3 
      theta_k = c(-2, 0.42, 1.02)
      
      h = matrix(, nrow=n_obs, ncol=3)
      for (i in 1:n_obs){
        h[i,] = theta_k + 0.2 * X_1[i] - 0.3 * X_2[i]
      }
      
      U3 = rlogis(n_obs)
      # chooses the correct X value if U3 is smaller than -2 that is level one if it's between -2 and 0.42 it's level two answer on
      x3 = rep(1, n_obs)
      x3[U3 > h[,1]] = 2
      x3[U3 > h[,2]] = 3
      x3[U3 > h[,3]] = 4
      x3 = ordered(x3, levels=1:4)
    } else{
      x3 = rep(doX[3], n_obs)
    }
   
    #hist(X_3)
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = x3)
    dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
    
    q1 = quantile(dat.orig[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
    q3 = c(1, 4) #No Quantiles for ordinal data
    
    
    return(list(
      df_orig=dat.tf, 
      df_R = dat.orig,
      #min =  tf$reduce_min(dat.tf, axis=0L),
      #max =  tf$reduce_max(dat.tf, axis=0L),
      min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
      type = c('c', 'c', 'o'),
      A=A))
} 

train = dgp(40000)
test  = dgp(10000)
(global_min = train$min)
(global_max = train$max)
data_type = train$type



#### Fitting Tram ######
df = data.frame(train$df_orig$numpy())
fit.orig = Colr(X2~X1,df)
summary(fit.orig)
confint(fit.orig) #Original
dd = predict(fit.orig, newdata = data.frame(X1 = 0.5), type = 'density')
x2s = as.numeric(rownames(dd))
plot(x2s, dd, type = 'l', col='red')

#?predict.tram
summary(fit.orig)
confint(fit.orig) #Original 

# Fitting Tram
fit.orig = Polr(x3 ~ x1 + x2,train$df_R)
summary(fit.orig)
confint(fit.orig) #Original 

MA =  matrix(c(0, 'ls', 'ls', 0,0, 'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)

hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
len_theta = 20 # Number of coefficients of the Bernstein polynomials
len_theta_max = len_theta
for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
  if (train$type[i] == 'o'){
    len_theta_max = max(len_theta_max, nlevels(train$df_R[,i]) - 1)
  }
}
param_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta_max, 
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


# loss before training
struct_dag_loss(t_i=train$df_orig, h_params=h_params)


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
  if (FALSE){ ### Full Training w/o diagnostics
    hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 200L,verbose = TRUE)
    param_model$save_weights(fn)
    plot(hist$epoch, hist$history$loss)
    plot(hist$epoch, hist$history$loss, ylim=c(1.07, 1.2))
  } else { ### Training with diagnostics
    ws <- data.frame(w12 = numeric())
    train_loss <- numeric()
    val_loss <- numeric()
    
    # Training loop
    num_epochs <- 200
    for (e in 1:num_epochs) {
      print(paste("Epoch", e))
      hist <- param_model$fit(x = train$df_orig, y = train$df_orig, 
                              epochs = 1L, verbose = TRUE, 
                              validation_data = list(test$df_orig,test$df_orig))
      
      # Append losses to history
      train_loss <- c(train_loss, hist$history$loss)
      val_loss <- c(val_loss, hist$history$val_loss)
      
      # Extract specific weights
      w <- param_model$get_layer(name = "beta")$get_weights()[[1]]
      
      ws <- rbind(ws, data.frame(w12 = w[1, 2], w13 = w[1, 3], w23 = w[2, 3]))
    }
    # Save the model
    param_model$save_weights(paste0(fn, '_normal_', num_epochs, '.h5'))
    save.image(paste0(fn, '_normal_', num_epochs, '.RData'))
    #pdf(paste0('loss_',fn,'.pdf'))
    epochs = length(train_loss)
    plot(1:length(train_loss), train_loss, type='l', ylim=c(1,1.2), main='Normal Training')
    lines(1:length(train_loss), val_loss, type = 'l', col = 'green')
    
    plot(1:epochs, ws[,1], type='l', main='Coef', ylim=c(-0.5, 3))#, ylim=c(0, 6))
    abline(h=2, col='green')
    
    lines(1:epochs, ws[,2], type='l', ylim=c(0, 3))
    abline(h=0.2, col='green')
    
    lines(1:epochs, ws[,3], type='l', ylim=c(0, 3))
    abline(h=-0.3, col='green')
    
    
    ggplot(ws, aes(x=1:nrow(ws))) + 
      geom_line(aes(y=w12, color='x1 --> x2')) + 
      geom_line(aes(y=w13, color='x1 --> x3')) + 
      geom_line(aes(y=w23, color='x2 --> x3')) + 
      geom_hline(aes(yintercept=2, color='x1 --> x2'), linetype=2) +
      geom_hline(aes(yintercept=0.2, color='x1 --> x3'), linetype=2) +
      geom_hline(aes(yintercept=-0.3, color='x2 --> x3'), linetype=2) +
      #scale_color_manual(values=c('x1 --> x2'='skyblue', 'x1 --> x3='red', 'x2 --> x3'='darkgreen')) +
      labs(title='Coefficients (triangle_structured_mixed.R)', x='Epoch', y='Coefficients') +
      theme_minimal() +
      theme(legend.title = element_blank())  # Removes the legend title
    
    
  }
}
param_model$evaluate(x = train$df_orig, y=train$df_scaled)
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


df = data.frame(train$df_orig$numpy())
fit.21 = Colr(X2~X1,df)
temp = model.frame(fit.21)[1:2,-1, drop=FALSE] #WTF!
plot(fit.21, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X2) Colr and Our Model', cex.main=0.8)
lines(Xs[,2], h_I[,2], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,2], col='blue')

fit.312 = Polr(x3 ~ x1 + x2,train$df_R)
temp = model.frame(fit.312)[1:2, -1, drop=FALSE] #WTF!

plot(fit.312, which = 'baseline only', newdata = temp, col='blue', 
     main='h_I(X3) Polr currently w/our Model', cex.main=0.8)
rug(train$df_orig$numpy()[,3], col='blue')
theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
#Thetas for intercept
theta = to_theta3(theta_tilde)
theta_base = theta[1,3,1:3] #Are all equal for the batch
points(1:3, theta_base, col='red', pch='+', cex=2)


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

##### Checking observational distribution ####
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)

plot(table(train$df_R[,3])/sum(table(train$df_R[,3])), ylab='Probability ', 
     main='Black = Observations, Red samples from TRAM-DAG',
     xlab='X3')
table(train$df_R[,3])/sum(table(train$df_R[,3]))
points(as.numeric(table(s[,3]$numpy()))/5000, col='red', lty=2)
table(s[,3]$numpy())/5000

par(mfrow=c(1,2))
for (i in 1:2){
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

# fit.x3 = Colr(X3 ~ X1 + X2,df)
# newdata = data.frame(
#     X1 = rep(0.2, length(s2_colr)), 
#     X2 = s2_colr)
# 
# s3_colr = rep(NA, nrow(newdata))
# for (i in 1:nrow(newdata)){
#   # i = 2
#   s3_colr[i] = simulate(fit.x3, newdata = newdata[i,], nsim = 1)
# }

s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
i = 2
ds = dx0.2$df_orig$numpy()[,i]
hist(ds, freq=FALSE, 50, main='X2 | Do(X1=0.2)', xlab='samples', 
     sub='Histogram Samples from DGP with do. red:TRAM_DAG')
sample_dag_0.2 = s_dag[,i]$numpy()
lines(density(sample_dag_0.2), col='red', lw=2)
m_x2_do_x10.2 = median(sample_dag_0.2)


i = 3 
d = dx0.2$df_orig$numpy()[,i]
plot(table(d)/length(d), ylab='Probability ', 
     main='X3 | do(X1=0.2)',
     xlab='X3', ylim=c(0,0.6),  sub='Black DGP with do. red:TRAM_DAG')
points(as.numeric(table(s_dag[,3]$numpy()))/nrow(s_dag), col='red', lty=2)


s_dag = do_dag_struct(param_model, train$A, doX=c(0.7, NA, NA))
i = 2
ds = dx7$df_orig$numpy()[,i]
hist(ds, freq=FALSE, 50, main='X2 | Do(X1=0.7)', xlab='samples', 
     sub='Histogram Samples from DGP with do. red:TRAM_DAG')
sample_dag_07 = s_dag[,i]$numpy()
lines(density(sample_dag_07), col='red', lw=2)
m_x2_do_x10.7 = median(sample_dag_07)
m_x2_do_x10.7 - m_x2_do_x10.2














