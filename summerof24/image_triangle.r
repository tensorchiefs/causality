##### Oliver's MAC ####
reticulate::use_python("/Users/oli/miniforge3/envs/r-tensorflow/bin/python3.8", required = TRUE)
library(reticulate)

reticulate::py_config()
library(tensorflow)
library(keras)

cnn_input <- layer_input(shape = c(2L,2L))
##### END Oliver's MAC ####

reticulate::py_config()
set.seed(1)
# Attention: wd must be summerof24 in causality
fn = 'image_keras_triangle_0.6_ortho.h5' 


##################################
##### Utils for tram_dag #########
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
library(tidyverse)
source('utils_tf.R')

#### For TFP
library(tfprobability)
source('utils_tfp.R')



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

### Loading CIFAR10 data
#my_images <- dataset_cifar10()
my_images <- dataset_mnist()


# Colorize the MNIST data

colorize <- function(images, colors) {
  # images = my_images$train$x
  # colors = cols_train
  colored_images <- array(0, dim = c(dim(images)[1], 28, 28, 3))
  
  for (i in 1:dim(images)[1]) {
    colored_images[i, , , colors[i]] <- images[i, , ]
  }
  
  return(colored_images / 255) # Normalize
}

set.seed(1)
make_cols <- function(ys){
  #ys = my_images$train$y
  cols = rep(0, nrow = length(ys))
  for (i in 1:length(ys)){
    y =ys[i]
    if (y < 4){
      cols[i] = sample(x=c(1,2,3), size=1, prob = c(0.2, 0.1, 0.7))
    } else if (y < 7){
      cols[i] = sample(x=c(1,2,3), size=1, prob = c(1/4, 0.5, 1/4))
    } else {
      cols[i] = sample(x=c(1,2,3), size=1, prob = c(0.6, 0.3, 0.1))
    }
  }
  return(cols)
}

cols_train = make_cols(my_images$train$y)
train_images_colored <- colorize(images=my_images$train$x, colors=cols_train)

cols_test = make_cols(my_images$test$y)
test_images_colored <- colorize(images=my_images$test$x, colors=cols_test) 


# 
i=3
image(train_images_colored[i,,,cols_train[i]])


##### TEMP
dgp <- function(n_obs, cols, labels) {
     # n_obs = 6
    # cols = cols_train[1:6]
    # labels = my_images$train$y[1:6]
   
    ##### x1
    #x1 = cols
    
    #u1 = h_0(x1) = x1
    u1 = rlogis(n_obs, location =  0, scale = 1)
    x1 = u1
    
    ####### x2
    u2 = rlogis(n_obs, location =  0, scale = 1) 
    #a1 = 0.42
    a1 = 0.6
    #u2 = h(x2|x1,B) = h_0(x2) + eta(B) + a1*x1 
    #h_0(x2) = 0.21 * x2 
    etaB = (labels - 4) * (-0.3)#
    etaB = (labels - 4) * (0.1)#
    x2 =  (u2 - etaB - a1*x1)/0.21
    
    #Orginal Data
    dat =  data.frame(x1 = x1, x2 = x2)
    dat.tf = tf$constant(as.matrix(dat), dtype = 'float32')
    
    # Minimal values
    q1 = quantile(dat[,1], probs = c(0.05, 0.95)) 
    q2 = quantile(dat[,2], probs = c(0.05, 0.95))
 
     A <- matrix(c(0, 1, 
                   0, 0), nrow = 2, ncol = 2, byrow = TRUE)
    return(list(
      df_orig=dat.tf,  
      min = tf$constant(c(q1[1], q2[1]), dtype = 'float32'),
      max = tf$constant(c(q1[2], q2[2]), dtype = 'float32'),
      A=A))
} 

n_obs = 60000
# cols = cols_train[1:6]
# labels = my_images$train$y[1:6]
train = dgp(n_obs=n_obs, cols = cols_train, labels = my_images$train$y)

test = dgp(n_obs=10000, cols = cols_test, labels = my_images$test$y)

# compare to Colr for tabular part

(global_min = train$min)
(global_max = train$max)

test$min
test$max
############ Colr ################
library(tram)
df.train = data.frame(train$df_orig$numpy())
df.train$B = (my_images$train$y - 4.)

df.test = data.frame(test$df_orig$numpy())
df.test$B = (my_images$test$y - 4.)


fit.train = Colr(X2 ~ X1 + B, df.train)
confint(fit.train)
-logLik(fit.train)/n_obs
-logLik(fit.train, newdata = df.test)/n_obs

fit.trainX1 = Colr(X1 ~ B, df.train)
confint(fit.trainX1)

#### create Networks ####
#summary(tabular_model)xw

MA =  matrix(c(0, 'ls', 
               0,   0 ), nrow = 2, ncol = 2, byrow = TRUE)
MA
hidden_features_I = c(2,2)
hidden_features_CS = c(2,2)
len_theta = 20
tabular_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta, 
                                 hidden_features_CS = hidden_features_CS)

tabular_out = tabular_model$output
tabular_out$shape
# batch, no Variable, 1(CS) + 1(LS) + M
# params pro Variable
# first CS, second LS, third part BP-coeffs
tabular_model(train$df_orig)

###### create a cnn model ####
input_shape <- c(28, 28, 3)

# Define the CNN model using the functional API
cnn_input <- layer_input(shape = input_shape)

conv1 <- layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = 'relu')(cnn_input)
pool1 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv1)
dropout1 <- layer_dropout(rate = 0.35)(pool1)

conv2 <- layer_conv_2d(filters = 16, kernel_size = c(3, 3), activation = 'relu')(dropout1)
pool2 <- layer_max_pooling_2d(pool_size = c(2, 2))(conv2)

#dropout2 <- layer_dropout(rate = 0.35)(pool2)
flatten <- layer_flatten()(pool2)
dense1 <- layer_dense(units = 30, activation = 'relu')(flatten)



#### OZ-Layer <---   TODO CHECK #####
ORTHO = TRUE
if (ORTHO){
  oz <- reticulate::import_from_path("orthog")
  ozlayer <- oz$Orthogonalization()
  extract_columns_layer <- layer_lambda(f=function(x) x[, 1:2])
  redinfo <- extract_columns_layer(tabular_model$input)
  ortho = ozlayer(dense1, redinfo)
} else {
  print("---------- Ortho False ----------")
  ortho = dense1
}
cnn_out <- layer_dense(units = 1, activation = 'linear')(ortho)

# get shape batches, #Variables, 
# since image impacts only x2, the first column should be zero
# Depending on the #variables p, we need to repeat the tensor
repeated_tensor <- layer_repeat_vector(n = 2)(cnn_out)
tmp = k_reshape(k_constant(c(0,1)), shape=c(-1,2,1))
cnn_tensor <- layer_multiply(repeated_tensor, tmp)
# Reshape the tensor
cnn_out2 <- layer_reshape(target_shape = c(2, 1))(cnn_tensor)

merged_output <- layer_concatenate(list( tabular_out, cnn_out2))
merged_output$shape
# [batch, #variable, 1(CS) + 1(LS) + M(BP) + 1(CNN) ]

# Define final model
final_model <- keras_model(inputs = list(tabular_model$input, cnn_input), 
                           outputs = merged_output)

#summary(final_model)

h_params = final_model(list(train$df_orig, train_images_colored ))
h_params$shape
# [batch, #variable, 1(CS) + 1(LS) + M(BP) + 1(CNN) ]

# check that only third variables p has an eta_B Shift (last h_params-dim)
h_params[, ,9]


# loss before training
# since images are in a source node we do not need to provide them
semi_struct_dag_loss(train$df_orig, h_params)

optimizer = optimizer_adam()

optimizer
final_model$compile(optimizer, loss=semi_struct_dag_loss)
final_model$evaluate(x = list(train$df_orig, train_images_colored), 
                     y=train$df_orig, batch_size = 7L)
final_model

######### HESSIAN OPTIMIZATION ########
if (FALSE){
  
  if (FALSE){
    # Train the model using a custom training loop (just to check)
    # Custom training loop w/o second order 
    #4.17 --> 4.12
    for (epoch in 1:2) {
      with(tf$GradientTape(persistent = TRUE) %as% tape, {
        # Compute the model's prediction - forward pass
        h_params <- final_model(list(train$df_orig, train_images))
        loss <- semi_struct_dag_loss(train$df_orig, h_params)
      })
      # Compute gradients
      gradients <- tape$gradient(loss, final_model$trainable_variables)
      # Apply gradients to update the model parameters
      optimizer$apply_gradients(purrr::transpose(list(gradients, final_model$trainable_variables)))
      # Print the loss every epoch or more frequently if desired
      print(paste("Epoch", epoch, ", Loss:", loss$numpy()))
    }
  }
  
  beta_weights <- final_model$get_layer(name = "beta")$weights[[1]]
  
  # Custom training loop
  # Separate the weights of "beta" layers
  # Retrieve the specific beta weights variable
  ## Training in which the beta layers are trained with a 2nd order update using a Hessian
  train_step_with_hessian_beta_semi_struct <- tf_function(autograph = TRUE, 
                                                          function(train_data, train_images, beta_weights, optimizer, lr_hessian = 0.1, beta_only=FALSE) {
                                                            #train_step <- function(train_data, beta_weights) {
                                                            # train_data = train$df_orig
                                                            #with(tf$GradientTape(persistent = TRUE) %as% tape2, { # Gradients for second-order derivatives
                                                            #  with(tf$GradientTape(persistent = TRUE) %as% tape1, { # Gradients for first-order derivatives 
                                                            # if (beta_only) {
                                                            #   
                                                            # } else {
                                                            #   
                                                            # }
                                                            with(tf$GradientTape() %as% tape2, { # Gradients for second-order derivatives
                                                              with(tf$GradientTape() %as% tape1, { # Gradients for first-order derivatives 
                                                                h_params <- final_model(list(train_data, train_images))
                                                                loss <- semi_struct_dag_loss(train_data, h_params)
                                                                #hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 500L,verbose = TRUE)
                                                              })
                                                              
                                                              # Compute first-order gradients
                                                              all_weights <- final_model$trainable_weights
                                                              all_grads <- tape1$gradient(loss, all_weights)
                                                              #optimizer$apply_gradients(purrr::transpose(list(all_grads, all_weights))) #HACKATTACK
                                                              other_gradients <- all_grads[!sapply(final_model$trainable_weights, function(weight) {
                                                                identical(weight$name, beta_weights$name)
                                                              })]
                                                              beta_gradients <- all_grads[sapply(final_model$trainable_weights, function(weight) {
                                                                identical(weight$name, beta_weights$name)
                                                              })]
                                                              other_weights <- final_model$trainable_weights[!sapply(final_model$trainable_weights, function(weight) {
                                                                identical(weight$name, beta_weights$name)
                                                              })]
                                                              if (length(beta_gradients) != 1) {
                                                                stop("Current implementation only supports **one** beta layer")
                                                              }
                                                              b = beta_gradients[[1]]
                                                              bl_shape <- beta_weights$shape
                                                              hessians <- tape2$jacobian(beta_gradients[[1]], beta_weights)  
                                                            }) 
                                                            optimizer$apply_gradients(purrr::transpose(list(other_gradients, other_weights))) 
                                                            # Manipulate gradients and apply them 
                                                            # Flatten the Hessian tensor to a matrix for inversion
                                                            hessian_size <- bl_shape[[1]] * bl_shape[[2]]
                                                            hessian_flat <- tf$reshape(hessians, shape = c(hessian_size, hessian_size))  # Adjust shape as needed
                                                            # Add regularization to the Hessian
                                                            hessian_flat <- hessian_flat + tf$eye(hessian_size) * 1e-8
                                                            # Compute the inverse of the Hessian matrix
                                                            hessian_inv <- tf$linalg$inv(hessian_flat)
                                                            # DEBUG HACK ATTACK - replace with tf$linalg$inv when fixed <-------------TODO 
                                                            #hessian_inv <- tf$eye(hessian_size)  # Identity matrix of appropriate size
                                                            # Flatten the gradient for matrix multiplication
                                                            grads_flat <- tf$reshape(beta_gradients[[1]], shape = c(hessian_size, 1L))
                                                            # Compute the update using Hessian and gradient (this is the newton update rule)
                                                            beta_update <- tf$matmul(hessian_inv, grads_flat)
                                                            # Reshape the update back to the original shape
                                                            beta_update_reshaped <- tf$reshape(beta_update, shape = bl_shape)  # Adjust shape as needed
                                                            # Apply the update to the beta weights with the learning rate
                                                            beta_weights$assign_sub(lr_hessian * beta_update_reshaped)
                                                            loss
                                                          })
  optimizer <- optimizer_adam()
  
  optimizer <- tf$keras$optimizers$Adam()  # Adam optimizer for other layers)
  # Wrap the custom training loop with tf_function
  # Prepare the dataset with batching
  batch_size <- 32
  num_batches <- ceiling(nrow(train$df_orig) / batch_size)
  indices <- sample(nrow(train$df_orig)) # Shuffle the indices
  
  # Custom training loop with batches
  epochs <- 20
  loss_values <- numeric(epochs)  # Vector to store loss values for each epoch
  loss_val <- numeric(epochs)
  
  # Lists to store beta weights for each epoch
  betas_21 <- numeric(epochs)
  
  time.start = Sys.time()
  time.last = Sys.time()
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
      loss <- train_step_with_hessian_beta_semi_struct(train_data = batch_data,
                                                       train_images = train_images_colored[batch_indices,,,],
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
    betas_out <- final_model$get_layer(name = "beta")$get_weights() * final_model$get_layer(name = "beta")$mask
    betas_out <- betas_out[1,,]$numpy()
    betas_21[epoch] <- betas_out[1,2]
    
    #validation_data = list(list(test$df_orig, test_images_colored), test$df_orig)
    val_loss = final_model$evaluate(x = list(test$df_orig, test_images_colored), y=test$df_orig)
    loss_val[epoch] <- val_loss
    
    if (epoch %% 10 == 0 || epoch == 1) {
      #Print time needed for 10 epochs
      print(Sys.time())
      print(Sys.time() - time.last)
      time.last = Sys.time()
      cat(sprintf("Epoch %d, Average Loss: %f val_loss: %f)\n", epoch, avg_epoch_loss, val_loss))
      print(paste0('Betas: ', betas_out[1,2]))#, ' colr:', b21.colr))
    }
  }
  
  # After training, you can inspect the loss values and beta weights
  print(loss_values)
  print(betas_21)
  
  # Plot the loss value
  #loss_values_2 = loss_values
  #loss_val_2 = loss_val
  plot(loss_values, type = "l", xlab = "Epoch", ylab = "Loss", main = "Semi Hessian Optimization lr_b = 0.1 (two runs)")
  lines(loss_val, col='green')
  abline(h=-0.30426180)
  #lines(loss_values_2, col='black')
  #lines(loss_val_2, col='green')
  
  # Plot the beta weights together with the Colr estimates in a single plot
  plot(betas_21, type = "l", xlab = "Epoch", ylab = "Beta Value", 
       main = "Semi Hessian Optimization lr_b = 0.1",
       sub = "Green colr dashed lined need know of the DGP")
  

}



####### Normal Training ####
if (file.exists(fn)){
  final_model$load_weights(fn)
} else {
  # Initialize a data frame to store specific weights
  ws <- data.frame(w12 = numeric())
  # Initialize lists to store loss history
  train_loss <- numeric()
  val_loss <- numeric()
  
  # Training loop
  num_epochs <- 20
  for (e in 1:num_epochs) {
    print(paste("Epoch", e))
    # if (e < Inf) {
    #   # Release weights and compile with the smaller learning rate
    #   release_weights(final_model)
    #   final_model$compile(optimizer = optimizer_unfrozen, loss = semi_struct_dag_loss)
    # } else {
    #   # Freeze weights and compile with the larger learning rate
    #   freeze_weights(final_model)
    #   final_model$compile(optimizer = optimizer_frozen, loss = semi_struct_dag_loss)
    # }
    
    
    hist <- final_model$fit(x = list(train$df_orig, train_images_colored), 
                            y = train$df_orig, 
                            epochs = 1L, verbose = TRUE, 
                            validation_data = list(list(test$df_orig, test_images_colored), test$df_orig))
    
    # Append losses to history
    train_loss <- c(train_loss, hist$history$loss)
    val_loss <- c(val_loss, hist$history$val_loss)
    
    # Extract specific weights
    w <- final_model$get_layer(name = "beta")$get_weights()[[1]]
    
    ws <- rbind(ws, data.frame(w12 = w[1, 2]))
  }
  
  final_model$save_weights(paste0(fn, '_normal_', num_epochs, '.h5'))
  save.image(paste0(fn, '_normal_', num_epochs, '.RData'))
  #pdf(paste0('loss_',fn,'.pdf'))
  epochs = length(train_loss)
  plot(1:length(train_loss), train_loss, type='l', main='Normal Training w/o othogonalization')
  lines(1:length(train_loss), val_loss, type = 'l', col = 'green')
  abline(h = -0.13222937)
  abline(v = 5, col = 'red')
  
  plot(1:epochs, ws[,1], type='l', main=paste0('Normal Training w/orthogonalization ', ORTHO
                                               ), ylim=c(0, 0.7))#, ylim=c(0, 6))
  abline(h = 0.6, col = 'red')
  abline(v = 4, col = 'red')
  #Save a pdf of the loss function containing fn using pdf()
  #dev.off()
}

# final_model$evaluate(x = list(train$df_scaled, train_images), 
#                      y=train$df_scaled, batch_size = 7L)
fn
len_theta
final_model$get_layer(name = "beta")$get_weights() * final_model$get_layer(name = "beta")$mask
param_model = final_model


###### DER CODE WEITER UNTEN IST NICHT GETESTET  ######
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




















