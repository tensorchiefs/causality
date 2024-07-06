##################################
##### Utils for tram_dag #########
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


fn = 'image_dah_wo_image_2Jul.h5'

### Loading CIFAR10 data
#my_images <- dataset_cifar10()
my_images <- dataset_mnist()

## Getting Training data
n_obs = 20000
train_images <- my_images$train$x[1:n_obs,,]
dim(train_images)
train_labels <- my_images$train$y[1:n_obs]

##### TEMP
dgp <- function(n_obs, b_labels=NULL) {
  
  scale_effect_size = 10.
  
  ###### X1 
  # depending on labels of images 
  # Images <=5 are "female" the other "male" with noise
  if (is.null(b_labels)){
    b_labels = runif(n_obs, 0, 1) < 0.5
    # x1 = ifelse(b_labels + runif(n_obs,-2,2) < 5, 0, 1) 
    # sum(x1==0)/length(x1) #0.556 
    u1 = rlogis(n_obs, location =  0, scale = 1)
    #u1 = h(x1) = h_0(x1) = qlogis(0.556) = 0.2249
    x1 = ifelse(u1 > qlogis(0.556), yes=1, no=0)
    #Same as x1 = rbinom(n=n_obs, size=1, prob = 1-0.556)
  } else{
    x1 = ifelse(b_labels + runif(n_obs,-2,2) < 5, 0, 1) 
  }
  
  ####### X2 
  u2 = rlogis(n_obs, location =  0, scale = 1) 
  b2 = 0.5 * scale_effect_size
  #h_0(x2) = 0.42 * x2
  #u2 = h(x2 | x1) = h_0(x2) + b2 * x1 = 0.42 * x2 + 0.5 * x1
  x2 = (u2 - b2*x1)/0.42
  
  ####### x4
  u4 = rlogis(n_obs, location =  0, scale = 1) 
  # u4 = h(x4|x1) = h_0(x4) - 4.9 * x1 
  # u4 = h(x4|x1) = 0.92*x4 - 4.9 * x1 
  x4 = (u4 + 4.9*x1)/0.92
  
  ####### x3
  u3 = rlogis(n_obs, location =  0, scale = 1) 
  a1 = 0.2 * scale_effect_size
  a2 = 0.03 * scale_effect_size
  a4 = 1.0
  #u3 = h(x3|x1,x2,B) = h_0(x3) + eta(B) + a1*x1 + a2*x2 + a4*x4
  #h_0(x3) = 0.21 * x3 
  #etaB = sqrt(b_labels) #Effect
  x3 =  (u3 - a1*x1 - a2*x2 - a4*x4)/0.21
  
  #Orginal Data
  dat =  data.frame(x1 = x1, x2 = x2, x3 = x3, x4 = x4)
  dat.tf = tf$constant(as.matrix(dat), dtype = 'float32')
  
  # Minimal values
  q1 = quantile(dat[,1], probs = c(0.05, 0.95)) 
  q2 = quantile(dat[,2], probs = c(0.05, 0.95))
  q3 = quantile(dat[,3], probs = c(0.05, 0.95))
  q4 = quantile(dat[,4], probs = c(0.05, 0.95))
  
  
  A <- matrix(c(
      0, 1, 1, 1,
      0, 0, 1, 0,
      0, 0, 0, 0,
      0, 0, 1, 0
      ), nrow = 4, ncol = 4, byrow = TRUE)
  
  return(list(
    df_orig=dat.tf,  
    min = tf$constant(c(q1[1], q2[1], q3[1], q4[1]), dtype = 'float32'),
    max = tf$constant(c(q1[2], q2[2], q3[2], q4[2]), dtype = 'float32'),
    A=A))
} 




# compare to Colr for tabular part
n_obs=4000
#train_images <- my_images$train$x[1:n_obs,,] / 255.
train_labels <- my_images$train$y[1:n_obs]

train = dgp(n_obs=n_obs)
train = dgp(n_obs=n_obs, b_labels=train_labels)
(global_min = train$min)
(global_max = train$max)


#### Fitting Colr ######
df = data.frame(train$df_orig$numpy())
fit.orig = Colr(X2~X1,df)
dd = predict(fit.orig, newdata = data.frame(X1 = 0.5), type = 'density')
x2s = as.numeric(rownames(dd))
plot(x2s, dd, type = 'l', col='red')

?predict.tram
summary(fit.orig)
(b21.colr = coef(fit.orig)[1])
confint(fit.orig) #Original 
# Fitting Tram
df = data.frame(train$df_orig$numpy())
fit.orig = Colr(X3 ~ X1 + X2,df)
summary(fit.orig)
confint(fit.orig) #Original 

fit.orig = Colr(X3 ~ X1 + X2 + X4,df)
summary(fit.orig)
confint(fit.orig) #Original 
(b31.colr = coef(fit.orig)[1])
(b32.colr = coef(fit.orig)[2])
(b34.colr = coef(fit.orig)[3])

fit.orig = Colr(X4 ~ X1,df)
summary(fit.orig)
confint(fit.orig) #Original 

MA =  matrix(c(
  0, 'ls', 'ls', 'ls', 
  0,    0, 'ls', 0,
  0,    0,  0, 0,
  0,    0, 'cs', 0
), nrow = 4, ncol = 4, byrow = TRUE)

MA =  matrix(c(
    0, 'ls', 'ls', 'ls', 
    0,    0, 'ls', 0,
    0,    0,  0, 0,
    0,    0, 'ls', 0
), nrow = 4, ncol = 4, byrow = TRUE)



hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
hidden_features_CS = c(2,20,50,200,500,200,50,20,2)
len_theta = 20
param_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta, 
                                 hidden_features_CS = hidden_features_CS)

summary(param_model)

x = tf$ones(shape = c(2L, 4L))
param_model(1*x)
MA
h_params = param_model(train$df_orig)
# Check the derivatives of h w.r.t. x
x <- tf$ones(shape = c(2L, 4L)) #B,P
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
struct_dag_loss(train$df_orig, h_params)
param_model = create_param_model(MA, hidden_features_I=hidden_features_I, 
                                 len_theta=len_theta, hidden_features_CS=hidden_features_CS)
optimizer = optimizer_adam()
param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)


##### Training 2nd Order Custom Loop ####
if (FALSE){
#hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 30L,verbose = TRUE, batch_size = nrow(train$df_orig))
hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 30L,verbose = TRUE, batch_size=32L)
#4.15 --> 4.12
}
    

# Train the model using a custom training loop
# Custom training loop
# Separate the weights of "beta" layers
# Retrieve the specific beta weights variable

optimizer <- optimizer_adam()

if (FALSE){
  # Custom training loop w/o second order 
  #4.17 --> 4.12
  for (epoch in 1:30) {
     with(tf$GradientTape(persistent = TRUE) %as% tape, {
       # Compute the model's prediction - forward pass
       h_params <- param_model(train$df_orig)
       loss <- struct_dag_loss(train$df_orig, h_params)
     })
     # Compute gradients
     gradients <- tape$gradient(loss, param_model$trainable_variables)
     # Apply gradients to update the model parameters
     optimizer$apply_gradients(purrr::transpose(list(gradients, param_model$trainable_variables)))
     # Print the loss every epoch or more frequently if desired
     print(paste("Epoch", epoch, ", Loss:", loss$numpy()))
  }
}


beta_weights <- param_model$get_layer(name = "beta")$weights[[1]]
# Set learning rates
#0.001  # Learning rate for Hessian updates
optimizer <- tf$keras$optimizers$Adam()  # Adam optimizer for other layers)
# Wrap the custom training loop with tf_function
# Prepare the dataset with batching
batch_size <- 32
num_batches <- ceiling(nrow(train$df_orig) / batch_size)
indices <- sample(nrow(train$df_orig)) # Shuffle the indices

# Custom training loop with batches
epochs <- 200
loss_values <- numeric(epochs)  # Vector to store loss values for each epoch

# Lists to store beta weights for each epoch
betas_21 <- numeric(epochs)
betas_31 <- numeric(epochs)
betas_32 <- numeric(epochs)
betas_34<- numeric(epochs)

for (epoch in 1:epochs) {
  #epoch = 1
  batch_losses <- c()  # Vector to store loss values for the current epoch's batches
  
  for (batch_num in 1:num_batches) {
    #batch_num = 1
    start_index <- (batch_num - 1) * batch_size + 1
    end_index <- min(batch_num * batch_size, nrow(train$df_orig))
    if (start_index > end_index) {
      next
    }
    batch_indices <- indices[start_index:end_index]
    if (length(batch_indices) == 0) {
      next
    }
    batch_indices_tf <- tf$constant(batch_indices - 1L, dtype = tf$int32)
    batch_data <- tf$gather(train$df_orig, batch_indices_tf)
    # needs 
    loss <- train_step_with_hessian_beta(train_data = batch_data, 
                                         beta_weights = beta_weights, 
                                         optimizer = optimizer,
                                         lr_hessian = 0.1)
    # Collect the batch loss
    batch_losses <- c(batch_losses, loss$numpy())
  }
  
  # Calculate the average loss for the current epoch
  avg_epoch_loss <- mean(batch_losses)
  
  # Store the average loss for the current epoch
  loss_values[epoch] <- avg_epoch_loss
  
  # Extract and store the beta weights for the current epoch
  betas_out <- param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask
  betas_out <- betas_out[1,,]$numpy()
  betas_21[epoch] <- betas_out[1,2]
  betas_31[epoch] <- betas_out[1,3]
  betas_32[epoch] <- betas_out[2,3]
  betas_34[epoch] <- betas_out[4,3]
  
  if (epoch %% 10 == 0 || epoch == 1) {
    cat(sprintf("Epoch %d, Average Loss: %f\n", epoch, avg_epoch_loss))
    print(paste0('Betas: ', betas_out[1,2], ' colr:', b21.colr))
    print(paste0('Betas: ', betas_out[1,3], ' colr:', b31.colr))
    print(paste0('Betas: ', betas_out[2,3], ' colr:', b32.colr))
    print(paste0('Betas: ', betas_out[4,3], ' colr:', b34.colr))
  }
}


# After training, you can inspect the loss values and beta weights
print(loss_values)
print(betas_21)
print(betas_31)
print(betas_32)
print(betas_34)

# Plot the loss value
plot(loss_values, type = "l", xlab = "Epoch", ylab = "Loss", main = "Loss Value vs. Epoch")
# Plot the beta weights together with the Colr estimates in a single plot
plot(betas_21, type = "l", xlab = "Epoch", ylab = "Beta Value", main = "Beta Values vs. Epoch", ylim=c(-1,5))
lines(rep(b21.colr, epochs), col = "red")
lines(betas_31, type = "l", xlab = "Epoch", ylab = "Beta Value", main = "Beta Values vs. Epoch")
lines(rep(b31.colr, epochs), col = "red")
lines(betas_32, type = "l", xlab = "Epoch", ylab = "Beta Value", main = "Beta Values vs. Epoch")
lines(rep(b32.colr, epochs), col = "red")
lines(betas_34, type = "l", xlab = "Epoch", ylab = "Beta Value", main = "Beta Values vs. Epoch")
lines(rep(b34.colr, epochs), col = "red")



loss# Save the model weights
param_model %>% save_model_weights_hdf5('model_weights.h5')



##### Training ####
if (file.exists(fn)){
  param_model$load_weights(fn)
} else {
  hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 100L,verbose = TRUE)
  param_model$save_weights(fn)
  plot(hist$epoch, hist$history$loss)
}
plot(hist$epoch, hist$history$loss, ylim=c(1.55, 2.00))
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)
fn
len_theta
param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask

df = data.frame(train$df_orig$numpy())
colnames(df) = c('X1', 'X2', 'X3', 'X4')
confint(Colr(X3 ~ X1 + X2 + X4, df))

confint(Colr(X4 ~ ., df))



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
transformed_values <- predict(fit.1, newdata = seq(0, 1, 0.01))


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

cs_hat = h_params[,3,1]$numpy() 
plot(df$X4, cs_hat)
confint(lm(cs_hat ~ df$X4))
abline(lm(cs_hat ~ df$X4), col='red')

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
  hist(d, freq=FALSE, 50, main=paste0("green=We, red=Colr, hist DGP Do(X1=0.2) X_",i))
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




















