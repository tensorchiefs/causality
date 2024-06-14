reticulate::py_config()
if (FALSE){
  ### Old Style Libraries### 
  library(keras)
  library(tensorflow)
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
  #source('R/tram_dag/utils_tf.R') #Might be called twice (Oliver's Laptop)
  
  #### For TFP
  library(tfprobability)
  source('R/tram_dag/utils_tfp.R')
}

############## Semi-structured DAG with image (Very Special to DAG in this file) ################
semi_struct_dag_loss = function (t_i, h_params){
  
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  
  #### This function is used for the semi-structured model in which an image is included
  h_params_tabular = h_params[,,1:(dim(h_params)[3]-1)]
  h_params_image = h_params[,,dim(h_params)[3]]
  
  h_cs <- h_params_tabular[,,1, drop = FALSE]
  h_ls <- h_params_tabular[,,2, drop = FALSE]
  theta_tilde <- h_params_tabular[,,3:dim(h_params_tabular)[3], drop = FALSE]
  
  #Intercept 
  theta = to_theta3(theta_tilde)
  h_I = h_dag_extra(t_i, theta, k_min, k_max)  # batch, #Variables
  #LS
  h_LS = tf$squeeze(h_ls, axis=-1L)#tf$einsum('bx,bxx->bx', t_i, beta)
  #CS
  h_CS = tf$squeeze(h_cs, axis=-1L)  # batch, #Variables
  
  # CNN-eta(B)
  h_eta_B = h_params_image  # batch, #Variables
  
  h = h_I + h_LS + h_CS + h_eta_B
  
  #Compute terms for change of variable formula
  log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
  ## h' dh/dtarget is 0 for all shift terms
  log_hdash = tf$math$log(tf$math$abs(h_dag_dash_extra(t_i, theta, k_min, k_max)))
  
  log_lik = log_latent_density + log_hdash
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (-tf$reduce_mean(log_lik))
}

#fn = 'image_dag_etaBNot0.h5' 
#fn = 'image_dag_etaB0.h5' 
fn = 'image_dag_etaBsqrt_simple_cnn_strong_effects.h5' 
library(fields)

### Loading CIFAR10 data
#my_images <- dataset_cifar10()
my_images <- dataset_mnist()

## Getting Training data
n_obs = 20000
train_images <- my_images$train$x[1:n_obs,,] / 255.
train_labels <- my_images$train$y[1:n_obs]
dim(train_images)


if (FALSE){
  ###### Hack Attack replace the images with the first image of the label
  idx = rep(NA, 10)
  for (i in 1:10){
    idx[i] = which(train_labels == i-1)[1]
  }
  train_labels[idx] #0 1 2 3 4 5 6 7 8 9
  # Get the index of the image in which the label the first time appreas
  for (i in 1:n_obs){
    #i = 1000
    id = idx[train_labels[i] + 1]
    train_images[i,,] = train_images[id,,] 
  }
}



plot_images <- function(img_array, main_label) {
  # Convert the image data to a format suitable for rasterImage (adjust dimensions)
  # Normalize the pixel values to [0, 1]
  img_array <- img_array 
  # Prepare plotting area and plot the image
  plot(0:1, 0:1, type = "n", xlab = "", ylab = "", main = main_label, axes = FALSE)
  rasterImage(img_array, 0, 0, 1, 1)
}

# Setting up the plot area
par(mfrow = c(2, 5), mar = c(1, 1, 2, 1))
# Loop through the first 10 images
for (i in c(1:120)) {
  #i=100
  print(i)
  # Reshape each image (keeping the color channel as the last dimension)
  img <- train_images[i,,] 
  # Determine label
  label <- train_labels[i] 
  # Plot the image
  plot_images(img, main_label = paste("Label:", label))
}
# Reset par to default
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

##### TEMP
dgp <- function(n_obs, b_labels) {
    scale_effect_size = 10.
    
    # Create X1 depending on labels of images 
    # Images <=5 are "female" the other "male" with noise
    x1 = ifelse(b_labels + runif(n_obs,-2,2) < 5, 0, 1) 
    #x1 = sample(c(0,1),size=n_obs,replace = TRUE) 
    ####### x2 
    u2 = rlogis(n_obs, location =  0, scale = 1) 
    b2 = 0.5 * scale_effect_size
    #h_0(x2) = 0.42 * x2
    #u2 = h(x2 | x1) = h_0(x2) + b2 * x1 = 0.42 * x2 + 0.5 * x1
    x2 = (u2 - b2*x1)/0.42
    
    ####### x3
    u3 = rlogis(n_obs, location =  0, scale = 1) 
    a1 = 0.2  *  scale_effect_size 
    a2 = 0.03 *  scale_effect_size 
    #u3 = h(x3|x1,x2,B) = h_0(x3) + eta(B) + a1*x1 + a2*x2
    #h_0(x3) = 0.21 * x3 
    etaB = (b_labels - 4) #
    #etaB = 0 #No Effect 
    etaB = sqrt(b_labels) #Effect
    x3 =  (u3 - etaB - a1*x1 - a2*x2)/0.21
    
    #Orginal Data
    dat =  data.frame(x1 = x1, x2 = x2, x3 = x3)
    dat.tf = tf$constant(as.matrix(dat), dtype = 'float32')
    
    # Minimal values
    q1 = quantile(dat[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat[,2], probs = c(0.05, 0.95))
    q3 = quantile(dat[,3], probs = c(0.05, 0.95))
    
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    return(list(
      df_orig=dat.tf,  
      min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
    A=A))
} 


# compare to Colr for tabular part
n_obs=20000
train = dgp(n_obs=n_obs, b_labels = train_labels[1:n_obs])
global_min = train$min
global_max = train$max

df = as.data.frame(train$df_orig$numpy())
colnames(df) = c('X1', 'X2', 'X3')
df$X4 = sqrt(train_labels[1:n_obs])

lm(X4 ~ X1, df)
confint(Colr(X3 ~ X1 + X2 + X4,df))
confint(Colr(X4 ~ X1,df))

# Fitting Tram
df = data.frame(train$df_orig$numpy())
pairs(df)


fit.X2 = Colr(X2~X1,df)
summary(fit.X2)  # sollte 0.5 sein, ist auch
confint(fit.X2)

df$label = train_labels[1:n_obs] 
fit.X3 = Colr(X3 ~ X1 + X2 + label,df)
summary(fit.X3) 

fit.X3_X1 = Colr(X3 ~ X1 + X2 + label,df)
summary(fit.X3) 


# end Colr
#Bis jetzt alles CI
#MA =  matrix(c(0, 'ls', 'ci', 0,0,'cs',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
#MA =  matrix(c(0, 'ls', 'ls', 0,0,'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
MA =  matrix(c(0, 'ls', 'ls', 0,0, 'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
MA
hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
len_theta = 6
tabular_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta, 
                                 hidden_features_CS = hidden_features_CS)

#summary(tabular_model)
tabular_out = tabular_model$output
tabular_out$shape
# batch, no Variable, 1(CS) + 1(LS) + M

# params pro Variable
# first CS, second LS, third part BP-coeffs
tabular_model(train$df_orig)


###### create a cnn model
input_shape <- c(28, 28, 1)

# Define the CNN model using the functional API
cnn_input <- layer_input(shape = input_shape)

conv1 <- layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = 'relu')(cnn_input)
pool1 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv1)

conv2 <- layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu')(pool1)
pool2 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv2)

#dropout <- layer_dropout(rate = 0.25)(pool2)

flatten <- layer_flatten()(pool2)

dense1 <- layer_dense(units = 64, activation = 'relu')(flatten)
dropout2 <- layer_dropout(rate = 0.5)(dense1)

cnn_out <- layer_dense(units = 1, activation = 'linear')(dropout2)

# get shape batches, #Variables, 
# since image impacts only x3, the first 2 cols should be zero
repeated_tensor <- layer_repeat_vector(n = 3)(cnn_out)
tmp = k_reshape(k_constant(c(0,0,1)), shape=c(-1,3,1))
cnn_tensor <- layer_multiply(repeated_tensor, tmp)
# Reshape the tensor
cnn_out2 <- layer_reshape(target_shape = c(3, 1))(cnn_tensor)


# 


merged_output <- layer_concatenate(list( tabular_out, cnn_out2))
merged_output$shape
# [batch, #variable, 1(CS) + 1(LS) + M(BP) + 1(CNN) ]

# Define final model
final_model <- keras_model(inputs = list(tabular_model$input, cnn_input), 
                           outputs = merged_output)

#summary(final_model)

h_params = final_model(list(train$df_orig, train_images ))
h_params$shape
# [batch, #variable, 1(CS) + 1(LS) + M(BP) + 1(CNN) ]

# check that only third variable has an eta_B Shift (last h_params-dim)
h_params[, ,9]


# loss before training
# since images are in a source node we do not need to provide them
semi_struct_dag_loss(train$df_orig, h_params)


optimizer = optimizer_adam()
final_model$compile(optimizer, loss=semi_struct_dag_loss)
#final_model$evaluate(x = train$df_scaled, y=train$df_scaled, batch_size = 7L)
final_model$evaluate(x = list(train$df_orig, train_images), y=train$df_orig, batch_size = 7L)



##### Training ####

###### Warm start ####
if (FALSE){
  final_model$get_layer(name = "beta")$get_weights()
weights = tf$constant(
  as.matrix(c(
   0.,.5 ,0.2,
   0.,0.,0.03,
   0.,0.,0.
  ), nrow = 3), shape = c(3L,3L))
final_model$get_layer(name = "beta")$set_weights(list(weights))
final_model$get_layer(name = "beta")$get_weights()[[1]]


##### Training ####
ws = data.frame(w12=0.5, w13=0.2, w23=0.03)
for (e in 1:100){
  print(e)
  final_model$fit(x = list(train$df_orig, train_images), 
                         y=train$df_orig, 
                         epochs = 1L,verbose = TRUE)
  w = final_model$get_layer(name = "beta")$get_weights()[[1]]
  #final_model$get_layer(name = "beta")$set_weights(list(weights))
  ws = rbind(ws, c(w[1,2], w[1,3], w[2,3]))  
}
plot(0:200, ws[,1], type='l', ylim=c(-1,1))
lines(0:200, ws[,2], col='red')
lines(0:200, ws[,3], col='blue')
abline(h=0.5, col='black', lty=2)
abline(h=0.2, col='red', lty=2)
abline(h=0.03, col='blue', )
legend("topright", legend = c("w12 0.5", "w13 0.2", "w23 0.03"), col = c("black", "red", "blue"), lty = 1:1, cex = 0.8)
}

if (file.exists(fn)){
  final_model$load_weights(fn)
} else {
  hist = final_model$fit(x = list(train$df_orig, train_images), 
                         y=train$df_orig, 
                         epochs = 5000L,verbose = TRUE)
  final_model$get_layer(name = "beta")$get_weights()[[1]]
  final_model$save_weights(fn)
  pdf(paste0('loss_',fn,'.pdf'))
  plot(hist$epoch, hist$history$loss)
  #Save a pdf of the loss function containing fn using pdf()
  dev.off()
}
plot(hist$epoch, hist$history$loss, ylim=c(-0.53, -0.5))

# final_model$evaluate(x = list(train$df_scaled, train_images), 
#                      y=train$df_scaled, batch_size = 7L)
fn
len_theta
final_model$get_layer(name = "beta")$get_weights() * final_model$get_layer(name = "beta")$mask
param_model = final_model
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

h_params = final_model(list(train$df_orig, train_images))
h_params

### Checking
x1 = train$df_orig$numpy()[1:1000,1]
b = train_labels[1:1000]
lm(b ~ x1)
plot(x1 , b)
abline(lm(b ~ x1))
plot(table(x1, b), col=TRUE)

eta_hat = h_params[1:1000,3,9]$numpy() 
eta_true = sqrt(train_labels[1:1000])
plot(eta_true, eta_hat)
confint(lm(eta_hat ~ eta_true))
abline(lm(eta_hat ~ eta_true))
cor(eta_hat, eta_true)

h_params[1:1000,3,1]$numpy()#CS 0
h_params[1:1000,3,2]$numpy()#CS 0


# Sampling fitted model w/o intervention --> OK 
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  hist(d, freq=FALSE, 20, xlim=c(0.1,1.1), main=paste0("X_",i))
  lines(density(train$df_scaled$numpy()[,i]))
}

######### Simulation of do-interventions #####
doX=c(0.5, NA, NA)
dx5 = dgp(10000, doX=doX)
dx5$df_orig$numpy()[1:5,]
dx5$df_scaled$numpy()[1:5,]
s_0.5 = scaled_doX(doX, train)[1]$numpy()


doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX)
hist(dx5$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx5$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx5$df_orig$numpy()[,3])  
s_0.7 = scaled_doX(doX, train)[1]$numpy()

########### Do(x1) seems to work#####
s = do_dag_struct(param_model, train$A, doX=c(s_0.5, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.5) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}
sdown = unscale(train$df_orig, s)
hist(sdown$numpy()[,3], freq=FALSE)
median(sdown$numpy()[,3])

s = do_dag_struct(param_model, train$A, doX=c(s_0.7, NA, NA))
for (i in 1:3){
  d = s[,i]$numpy()
  d = d[d>0 & d<1]
  print(mean(d))
  hist(d, freq=FALSE, 50, xlim=c(0.1,1.1), main=paste0("Do (X3=0.7) X_",i))
  #lines(density(train$df_scaled$numpy()[,i]))
}
sup = unscale(train$df_orig, s)
hist(sup$numpy()[,3], freq=FALSE)
mean(sup$numpy()[,3])

mean(sup$numpy()[,2]) - mean(sdown$numpy()[,2])
mean(sup$numpy()[,3]) - mean(sdown$numpy()[,3])

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




















