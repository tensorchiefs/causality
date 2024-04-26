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
  source('R/tram_dag/utils_tf.R') #Might be called twice (Oliver's Laptop)
  
  #### For TFP
  library(tfprobability)
  source('R/tram_dag/utils_tfp.R')
}

fn = 'image_dag.h5'
library(fields)

### Loading CIFAR10 data
my_images <- dataset_mnist()
## Getting Training data
n_obs = 20
train_images <- my_images$train$x[1:n_obs]
train_labels <- my_images$train$y[1:n_obs]

plot_images <- function(img_array, main_label) {
  # Convert the image data to a format suitable for rasterImage (adjust dimensions)
  # Normalize the pixel values to [0, 1]
  img_array <- img_array / 255
  # Prepare plotting area and plot the image
  plot(0:1, 0:1, type = "n", xlab = "", ylab = "", main = main_label, axes = FALSE)
  rasterImage(img_array, 0, 0, 1, 1)
}

# Setting up the plot area
par(mfrow = c(2, 5), mar = c(1, 1, 2, 1))

# Loop through the first 10 images
for (i in 1:10) {
  # Reshape each image (keeping the color channel as the last dimension)
  img <- train_images[i,,,]
  # Determine label
  label <- train_labels[i, 1] + 1 # Adjusting label index for 0-based indexing in R
  # Plot the image
  plot_cifar(img, main_label = paste("Label:", label))
}

# Reset par to default
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

##### TEMP
dgp <- function(n_obs, b_labels) {
    x1 = ifelse(b_labels <= 5, 0, 1) #Images <=5 are "female" the other "male"
    
    ####### x2 
    u2 = rlogis(n_obs, location =  0, scale = 1) 
    b2 = 0.5
    #h_0(x2) = 0.42 * x2
    #u2 = h(x2 | x1) = h_0(x2) + b2 * x1 = 0.42 * x2 + 0.5 * x1
    x2 = (u2 - 0.5*x1)/0.42
    
    ####### x3
    u3 = rlogis(n_obs, location =  0, scale = 1) 
    a1 = 2
    a2 = 3
    #u3 = h(x3|x1,x2,B) = h_0(x3) + eta(B) + a1*x1 + a2*x2 
    #h_0(x3) = 0.21 * x3 
    etaB = 0.5 * b_labels
    x3 =  (u3 - etaB - a1*x1 - a2*x2)/0.21
    
    dat.s =  data.frame(x1 = x1, x2 = x2, x3 = x3)
    dat.tf = tf$constant(as.matrix(dat.s), dtype = 'float32')
    scaled = scale_df(dat.tf) * 0.99 + 0.005
    A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
    return(list(df_orig=dat.tf,  df_scaled = scaled, A=A))
} 

scaled_doX = function(doX, train){
  dat_tf = train$df_orig
  dat_min = tf$reduce_min(dat_tf, axis=0L)
  dat_max = tf$reduce_max(dat_tf, axis=0L)
  ret = (doX - dat_min) / (dat_max - dat_min)
  return(ret * 0.99 + 0.005)
}


train = dgp(n_obs, 1:n_obs )
#Bis jetzt alles CI
#MA =  matrix(c(0, 'ls', 'ci', 0,0,'cs',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
#MA =  matrix(c(0, 'ls', 'ls', 0,0,'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
MA =  matrix(c(0, 'ls', 'ls', 0,0, 'ls',0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
len_theta = 6
tabular_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta, 
                                 hidden_features_CS = hidden_features_CS)

tabular_out = tabular_model$output

tabular_model(train$df_scaled)


###### create a cnn model
input_shape <- c(28, 28, 1)

# Define the CNN model using the functional API
cnn_input <- layer_input(shape = input_shape)

conv1 <- layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu')(cnn_input)
pool1 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv1)

conv2 <- layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu')(pool1)
pool2 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv2)

dropout <- layer_dropout(rate = 0.25)(pool2)

flatten <- layer_flatten()(dropout)

dense1 <- layer_dense(units = 128, activation = 'relu')(flatten)
dropout2 <- layer_dropout(rate = 0.5)(dense1)

cnn_out <- layer_dense(units = 1, activation = 'linear')(dropout2)

repeated_tensor <- layer_repeat_vector(n = 3)(cnn_out)

# Reshape the tensor
cnn_out2 <- layer_reshape(target_shape = c(3, 1))(repeated_tensor)

merged_output <- layer_concatenate(list(cnn_out2, tabular_out))

# Define final model
final_model <- keras_model(inputs = list(cnn_input, dnn_input), outputs = merged_output)

Bis dahin sind wir gekommen


# Define the model
cnn_model <- keras_model(inputs = cnn_input, outputs = output)

# Compile the model
cnn_model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Print model summary
summary(cnn_model)





x = tf$ones(shape = c(2L, 3L))
param_model(1*x)
MA
h_params = param_model(train$df_scaled)
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
struct_dag_loss(train$df_scaled, h_params)


with(tf$GradientTape(persistent = TRUE) %as% tape, {
  h_params = param_model(train$df_scaled)
  loss = struct_dag_loss(train$df_scaled, h_params)
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
param_model$evaluate(x = train$df_scaled, y=train$df_scaled, batch_size = 7L)


##### Training ####
if (file.exists(fn)){
  param_model$load_weights(fn)
} else {
  hist = param_model$fit(x = train$df_scaled, y=train$df_scaled, 
                         epochs = 5000L,verbose = TRUE)
  param_model$save_weights(fn)
  plot(hist$epoch, hist$history$loss)
}
param_model$evaluate(x = train$df_scaled, y=train$df_scaled, batch_size = 7L)
fn
len_theta
param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask

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

d = train$df_scaled$numpy()
plot(d[,1],d[,2])
lm(d[,2] ~ d[,1])

lm(d[,3] ~ d[,1] + d[,2]) #Direct causal effect 0.28


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




















