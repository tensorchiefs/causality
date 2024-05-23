library(keras)
library(tensorflow)
if (FALSE){
  ### Old Style Libraries### 
  source('R/tram_dag/utils_dag_maf.R') #Might be called twice
  source('R/tram_dag/utils_dag_maf.R') #Might be called twice
  source('R/tram_dag/utils.R')
} else{
  ##################################
  ##### Utils for tram_dag #########
  library(mlt)
  library(tram)
  library(MASS)
  library(tensorflow)
  library(keras)
  library(tidyverse)
  source('R/tram_dag/utils_tf.R')
  source('R/tram_dag/utils_tf.R') #Might be called twice (for oliver's mac)
  
  #### For TFP
  library(tfprobability)
  source('R/tram_dag/utils_tfp.R')
}

fn = 'ls_sigmoid_long10k_M6_dumm.h5'

##### TEMP
dgp <- function(n_obs) {
    X_1 = rnorm(n_obs)
    
    # Sampling according to colr
    p = runif(n_obs)
    x_2_dash = qlogis(p)
    #Checking if x_2_dash is from logistic
    #hist(x_2_dash, freq=FALSE,100)
    #xx = seq(-5,5,0.1)
    #lines(xx, dlogis(xx), col='red', lwd=2)
    #checked
    
    #x_2_dash = h_0(x_2) + beta * X_1
    #h_0(x_2) = 0.42 * x_2
    X_2 = 1/0.42 * (x_2_dash - 2 * X_1)
    #Sampling from 
    dat =  data.frame(x1 = X_1, x2 = X_2)
    dat.tf = tf$constant(as.matrix(dat), dtype = 'float32')
    
    ### Hack Attack using the scaled variables
    if (FALSE){
      dat.tf = dat.tf*c(10., 1.) #HACK ATTACK
      dat.tf = scale_df(dat.tf) * 0.99 + 0.005
    }
    
    #A rows from, cols to (upper triangle)
    A <- matrix(c(0, 1, 0,0), nrow = 2, ncol = 2, byrow = TRUE)
    return(list(
      df_orig=dat.tf,  
      min =  tf$reduce_min(dat.tf, axis=0L),
      max =  tf$reduce_max(dat.tf, axis=0L),
      A=A))
} 

train = dgp(2000)
(global_min = train$min)
(global_max = train$max)

# Fitting Tram
df = data.frame(train$df_orig$numpy())
plot(df)


#### COLR ####
fit.orig = Colr(X2~X1,df)
confint(fit.orig)
?Colr
summary(fit.orig)
logLik(fit.orig) / nrow(df)


#Bis jetzt alles CI
#MA =  matrix(c(0, 'ls', 'ci', 0,0,'cs',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
#MA =  matrix(c(0, 'ls', 'ls', 0,0,'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
MA =  matrix(c(0, 'ls', 0,0), nrow = 2, ncol = 2, byrow = TRUE)
hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
len_theta = 10
param_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta, 
                                 hidden_features_CS = hidden_features_CS)

x = tf$ones(shape = c(3L, 2L))
h_params_1 = param_model(1*x)
h_params_1[1,1,] #First two should be 0
h_params_1[1,2,] #First CS should be 0
MA
# Check the derivatives of h w.r.t. x
x <- tf$ones(shape = c(3L, 2L)) #B,P
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
#     [,1] [,2]
#[1,] "0"  "ls"
#[2,] "0"  "0" 
# Nur der zweite !=0 
for (k in 1:(2+len_theta)){ #k = 1
  print(k) #B,P,k,B,P
  B = 1 # first batch
  print(d[B,,k,B,]) #
}

# loss before training
h_params = param_model(train$df_orig)
struct_dag_loss(train$df_orig, h_params)
with(tf$GradientTape(persistent = TRUE) %as% tape, {
  h_params = param_model(train$df_orig)
  loss = struct_dag_loss(train$df_orig, h_params)
})

gradients = tape$gradient(loss, param_model$trainable_variables)
gradients

param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=6, hidden_features_CS=hidden_features_CS)


# ######### DEBUG TRAINING FROM HAND #######
# # Define the optimizer
# optimizer <- tf$optimizers$Adam(lr=0.01)
# # Define the number of epochs for training
# num_epochs <- 10
# for (epoch in 1:num_epochs) {
#   with(tf$GradientTape(persistent = TRUE) %as% tape, {
#     # Compute the model's prediction - forward pass
#     h_params <- param_model(train$df_orig)
#     loss <- struct_dag_loss(train$df_orig, h_params)
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
                         epochs = 500L,verbose = TRUE)
  param_model$save_weights(fn)
  plot(hist$epoch, hist$history$loss)
}
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)
fn

param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask

#### Checking Transformation Funktion #######


# Check the derivatives of h w.r.t. x
x <- tf$ones(shape = c(10L, 2L)) #B,P
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
fit = lm(o[,2] ~ o[,1])
confint(fit)

d = train$df_orig$numpy()
plot(d[,1],d[,2])
fit = lm(d[,2] ~ d[,1])
confint(fit)


# Sampling fitted model w/o intervention --> OK 
s = do_dag_struct(param_model, train$A, doX=c(NA, NA), num_samples = 5000)
for (i in 1:2){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 20, xlim=c(0.1,1.1), main=paste0("X_",i))
  lines(density(train$df_orig$numpy()[,i]))
}
library(car)
qqplot(s$numpy()[,2], train$df_orig$numpy()[,2], xlim=c(0,1))
abline(0,1)

########### Do(x1) seems to work#####
s = do_dag_struct(param_model, train$A, doX=c(0.5, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_orig$numpy()[,i]))
}

s = do_dag_struct(param_model, train$A, doX=c(0.7, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_orig$numpy()[,i]))
}


########### Do(x2) seem to work #####
s = do_dag_struct(param_model, train$A, doX=c(NA, 0.5, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_orig$numpy()[,i]))
}

s = do_dag_struct(param_model, train$A, doX=c(NA, 0.7, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_orig$numpy()[,i]))
}



########### TODO Check the sampling (prob needs ad) #####

dox1_2=scale_value(train$df_orig, col=1L, 2) #On X2
s_dox1_2 = do_dag(param_model, train$A, doX=c(dox1_2$numpy(), NA, NA), num_samples = 5000)
s = s_dox1_2
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  lines(density(train$df_orig$numpy()[,i]))
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


hist(train$df_orig$numpy()[,1], freq=FALSE)

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




















